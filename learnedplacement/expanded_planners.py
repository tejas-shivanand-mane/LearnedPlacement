"""Feasible plan construction and bounded, auditable v4 baselines.

Planners never receive held-out evaluation traces. Local search is a heuristic,
not an optimality oracle. The policy adapter exposes no realized future events.
"""
from dataclasses import replace
import hashlib
import itertools
import json
import random
import time

from .expanded import Decision, effective_case, simulate, violations
from .metrics import planning_score, summarize
from .model import seed_for
from .simulator import BudgetExpired, check_budget

METHODS = ('speed_fixed','domain_tuned','coordinate_search','random_search','local_search','random_unaware')


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def spread_shards(s, groups):
    """Greedy within-group swaps reduce shared-domain copies without regrouping.

    Group membership is preserved, but inter-group communication can change in
    the v4 topology model. Domain separation is therefore not assumed free.
    """
    groups = [list(g) for g in groups]
    c = s.workload
    def penalty():
        return sum(c.nodes[groups[a][j]].domain == c.nodes[groups[b][j]].domain
                   for a in range(len(groups)) for b in range(a) for j in range(len(groups[0])))
    for _ in range(3):
        changed = False
        for g in groups[1:]:
            for a,b in itertools.combinations(range(len(g)),2):
                before = penalty(); g[a],g[b] = g[b],g[a]
                if penalty() < before: changed = True
                else: g[a],g[b] = g[b],g[a]
        if not changed: break
    return tuple(tuple(g) for g in groups)


def layout(s, shards, replicas, kind='speed'):
    c = s.workload
    if kind == 'price':
        key = lambda n:(c.nodes[n].price_per_hour/c.nodes[n].speed,n)
    elif kind == 'risk':
        key = lambda n:(c.nodes[n].failure_per_hour/c.nodes[n].speed,-c.nodes[n].speed,n)
    else:
        key = lambda n:(-c.nodes[n].speed,n)
    # Ensure enough memory for the primary and ordinary working state.
    nodes = sorted((n for n,x in enumerate(c.nodes) if x.memory_gib >= c.state_gib/shards+c.working_gib),key=key)
    if len(nodes) < shards*replicas: return None
    nodes = nodes[:shards*replicas]
    nodes.sort(key=lambda n:(-c.nodes[n].speed,n))
    groups = tuple(tuple(nodes[i:i+shards]) for i in range(0,len(nodes),shards))
    if kind != 'speed': groups = spread_shards(s,groups)
    return Decision(groups,checkpoint_every=c.checkpoint_every,backup_every=c.backup_every)


def resources(s, p, copies=0, spares=0):
    """Greedily allocate distinct backup copies, enforcing every memory limit."""
    c = s.workload
    q = replace(p,backups=(),spares=())
    candidates = sorted(set(range(len(c.nodes)))-q.active,key=lambda n:(-c.nodes[n].speed,n))
    for n in candidates:
        if len(q.spares) >= spares: break
        trial = replace(q,spares=q.spares+(n,))
        if not violations(s,trial): q = trial
    for _ in range(copies):
        for shard in range(len(p.groups[0])):
            hosts = [g[shard] for g in q.groups] + [n for j,n in q.backups if j == shard]
            domains = {c.nodes[n].domain for n in hosts}
            choices = sorted(set(range(len(c.nodes)))-set(hosts)-set(q.spares),key=lambda n:(c.nodes[n].domain in domains,c.nodes[n].failure_per_hour,c.nodes[n].price_per_hour,n))
            for n in choices:
                trial = replace(q,backups=q.backups+((shard,n),))
                if not violations(s,trial): q = trial; break
    return q


def spaces(s, config):
    n = len(s.workload.nodes)
    return [(shards,replicas) for shards in config['choices']['shards']
            for replicas in config['choices']['replicas'] if shards*replicas <= n]


def base_plan(s, config, domain=True):
    c = s.workload
    p = layout(s,c.shards,c.replicas,'domain' if domain else 'speed')
    if p is not None and not violations(s,p): return p
    for shards,replicas in spaces(s,config):
        p = layout(s,shards,replicas,'domain' if domain else 'speed')
        if p is not None and not violations(s,p): return p
    raise ValueError('No structurally feasible layout in configured space')


def random_plan(s, config, rng, fixed_layout=None):
    choices = config['choices']
    if fixed_layout is None:
        shards,replicas = rng.choice(spaces(s,config))
        p = layout(s,shards,replicas,rng.choice(('speed','domain','price','risk')))
        if p is None: return None
        if rng.random() < .5:
            nodes = list(p.active); rng.shuffle(nodes)
            p = replace(p,groups=tuple(tuple(nodes[i:i+shards]) for i in range(0,len(nodes),shards)))
    else:
        p = fixed_layout
    updates = {key:rng.choice(choices[key]) for key in POLICY_FIELDS}
    p = replace(p,**updates)
    if violations(s,p): return None
    p = resources(s,p,rng.choice(choices['backup_copies']),rng.choice(choices['spare_count']))
    return None if violations(s,p) else p


POLICY_FIELDS = ('checkpoint_every','backup_every','checkpoint_tier','checkpoint_mode',
                 'checkpoint_retention','background_fraction','recovery','repair_every',
                 'recovery_order','source_policy')


def neighbors(s, p, config):
    """One policy/layout/resource edit at a time, followed by feasibility filtering."""
    choices = config['choices']
    for field in POLICY_FIELDS:
        for value in choices[field]:
            if value != getattr(p,field):
                q = replace(p,**{field:value})
                if not violations(s,q): yield q
    for copies in choices['backup_copies']:
        for spares in choices['spare_count']:
            q = resources(s,p,copies,spares)
            if q != p and not violations(s,q): yield q
    # Group structure changes preserve policy choices, rebuild protection.
    for shards,replicas in spaces(s,config):
        if (shards,replicas) == (len(p.groups[0]),len(p.groups)): continue
        q = layout(s,shards,replicas,'domain')
        if q is not None:
            q = replace(q,**{f:getattr(p,f) for f in POLICY_FIELDS})
            if not violations(s,q): yield q
    flat = [n for g in p.groups for n in g]
    width = len(p.groups[0])
    for i,j in itertools.combinations(range(len(flat)),2):
        ids = flat.copy(); ids[i],ids[j] = ids[j],ids[i]
        q = replace(p,groups=tuple(tuple(ids[k:k+width]) for k in range(0,len(ids),width)))
        if not violations(s,q): yield q
    for i in range(len(flat)):
        for node in set(range(len(s.workload.nodes)))-p.active-set(p.spares):
            ids = flat.copy(); ids[i] = node
            q = replace(p,groups=tuple(tuple(ids[k:k+width]) for k in range(0,len(ids),width)))
            if not violations(s,q): yield q


def plan(s, method, traces, config, prior=None, stop_at=None, initial_plan=None, score_fn=None):
    """Wall-clock and completed-candidate bounded; partial candidate scores discarded."""
    prior = prior or {}
    score_fn = score_fn or planning_score
    if method == 'random_unaware':
        if 'random_search' not in prior: raise ValueError('random_search must run first')
        return prior['random_search']['controls']['random_unaware']
    if method not in METHODS: raise ValueError('Unknown planner')
    started = time.monotonic()
    limit = min(started+config['planner_seconds'],stop_at) if stop_at is not None else started+config['planner_seconds']
    fallback = initial_plan or base_plan(s,config,domain=method != 'speed_fixed')
    if violations(s,fallback): raise ValueError('Invalid initial plan')
    best, best_score, best_summary = fallback,None,None
    healthy_best, healthy_score = fallback,None
    rng = random.Random(seed_for(config['seed'],s.workload.case_id,method))
    seen, audit = set(),[]
    candidate_limit = 1 if method == 'speed_fixed' else (config['tuned_candidates'] if method == 'domain_tuned' else config['search_candidates'])
    attempts = calls = 0
    reason = 'candidate_limit'
    iter_neighbors = iter(())
    changed = True
    aware_s = healthy_s = 0.
    try:
        while len(audit) < candidate_limit and attempts < max(100,candidate_limit*50):
            check_budget(limit)
            attempts += 1
            if not audit:
                q = fallback
            elif method == 'domain_tuned':
                # Explicit cadence alternatives are always tried before random
                # policy combinations, including the no-periodic-save choice.
                intervals = [v for v in config['choices']['checkpoint_every'] if v != fallback.checkpoint_every]
                if attempts <= len(intervals)+1:
                    q = replace(fallback,checkpoint_every=intervals[attempts-2])
                else:
                    q = random_plan(s,config,rng,fallback)
            elif method in ('coordinate_search','local_search'):
                if changed:
                    items = list(neighbors(s,best,config))
                    if method == 'local_search': rng.shuffle(items)
                    iter_neighbors = iter(items); changed = False
                q = next(iter_neighbors,None)
                if q is None:
                    if method == 'coordinate_search': reason = 'local_optimum'; break
                    q = random_plan(s,config,rng)  # restart proposal
            else:
                q = random_plan(s,config,rng)
            if q is None or q in seen or violations(s,q): continue
            seen.add(q)
            tick = time.monotonic()
            try:
                healthy = simulate(s,q,[],limit)
                calls += 1
            finally:
                healthy_s += time.monotonic()-tick
            tick = time.monotonic()
            rows = []
            try:
                for trace in traces:
                    rows.append(simulate(s,q,trace,limit))
                    calls += 1
            finally:
                aware_s += time.monotonic()-tick
            summary = summarize(s.workload,rows)
            score = score_fn(s.workload,summary)
            hscore = score_fn(s.workload,summarize(s.workload,[healthy]))
            audit.append(dict(plan=q.to_dict(),score=list(score),healthy_score=list(hscore)))
            if best_score is None or score < best_score:
                best,best_score,best_summary = q,score,summary; changed = True
            if healthy_score is None or hscore < healthy_score:
                healthy_best,healthy_score = q,hscore
    except BudgetExpired:
        reason = 'wall_clock_limit'
    if attempts >= max(100,candidate_limit*50): reason = 'proposal_limit'
    elapsed = time.monotonic()-started
    result = dict(plan=best.to_dict(),planning_s=elapsed,candidates=len(audit),simulations=calls,
                  status=reason,search_summary=best_summary,planning_point_feasible=best_summary['point_estimate_feasible'] if best_summary else None,
                  candidate_pool_sha256=digest([x['plan'] for x in audit]),candidate_audit=audit,
                  planning_time_kind='measured_search_wall_time',proposal_attempts=attempts)
    if method == 'random_search':
        overhead = max(0.,elapsed-aware_s-healthy_s)
        result['controls'] = {'random_unaware':dict(plan=healthy_best.to_dict(),planning_s=healthy_s+overhead,
                            candidates=len(audit),simulations=len(audit),status=reason,
                            candidate_pool_sha256=result['candidate_pool_sha256'],
                            planning_point_feasible=None,search_summary=None,
                            planning_time_kind='attributed_healthy_scoring_plus_proposals')}
    return result


def observation(s):
    """Serializable deployment inputs for a future policy; no trace or labels.

    Features are raw/unscaled. Fitting normalizers is a TRAIN-only future step.
    This adapter is not an RL implementation or a promise of optimal actions.
    """
    d = s.to_dict()
    d.pop('split')
    for key in ('case_id','trace_key','regime','profile'):
        d['workload'].pop(key,None)
    return d


def valid_candidates(s, config, seed, count=32):
    """Common candidate-action interface for future learned ranking policies."""
    if type(count) is not int or count <= 0: raise ValueError('count must be positive')
    rng = random.Random(seed)
    candidates = [base_plan(s,config)]
    for _ in range(count*50):
        if len(candidates) >= count: break
        q = random_plan(s,config,rng)
        if q is not None and q not in candidates: candidates.append(q)
    return candidates
