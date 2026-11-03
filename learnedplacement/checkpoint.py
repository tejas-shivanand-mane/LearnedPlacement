"""Bounded placement/protection/checkpoint search and matched controls."""
from dataclasses import replace
import hashlib
import json
import random
import time

from .metrics import planning_score, summarize
from .model import Plan, seed_for, violations
from .planners import add_backups, initial, neighbor
from .simulator import BudgetExpired, check_budget, simulate


def plan_checkpoint_method(c, method, traces, config, prior, stop_at=None):
    started = time.monotonic()
    if method in ('speed_grouped_tuned', 'domain_packed_tuned'):
        plain = initial(c, method.removesuffix('_tuned'))
        layouts = list(dict.fromkeys([plain, add_backups(c, plain)]))
        scored = []
        for layout in layouts:
            for interval in config['checkpoint_intervals']:
                p = replace(layout, checkpoint_every=interval)
                summary = summarize(c, [simulate(c, p, t, stop_at) for t in traces])
                scored.append((planning_score(c, summary), p, summary))
        _, p, summary = min(scored, key=lambda x: x[0])
        return dict(plan=p.to_dict(), planning_s=time.monotonic()-started,
                    candidates=len(scored), simulations=len(scored)*len(traces),
                    status='checkpoint_grid_on_fixed_placement', search_summary=summary,
                    planning_point_feasible=summary['point_estimate_feasible'])
    if method == 'checkpoint_joint':
        return checkpoint_search(c, traces, config, stop_at)
    parent = prior['checkpoint_joint']
    if method in ('checkpoint_unaware', 'checkpoint_fixed_search'):
        return parent['controls'][method]
    p = Plan.from_dict(parent['plan'])
    if method == 'checkpoint_fixed_ablation':
        p = replace(p, checkpoint_every=c.checkpoint_every)
    elif method == 'checkpoint_no_backup':
        p = replace(p, backups=())
    else:
        raise ValueError(f'Unknown checkpoint method: {method}')
    return dict(plan=p.to_dict(), planning_s=time.monotonic()-started,
                candidates=0, simulations=0, status='single_decision_ablation',
                parent='checkpoint_joint', parent_planning_s=parent['planning_s'], search_summary=None)


def checkpoint_search(c, traces, config, stop_at=None):
    """Score every interval for each layout; commit whole layout blocks only.

    Three selectors share completed layouts: failure-aware with a tunable
    interval, failure-unaware with a tunable interval, and failure-aware with
    the case's fixed reference interval. No scoring outcome guides proposals.
    """
    started = time.monotonic()
    local_stop = started + config['checkpoint_seconds']
    effective_stop = min(local_stop, stop_at) if stop_at is not None else local_stop
    intervals = config['checkpoint_intervals']
    neutral = replace(c, nodes=[replace(n, failure_per_hour=0.) for n in c.nodes], domain_failure_per_hour=0.)
    rng = random.Random(seed_for(config['seed'], c.trace_key or c.case_id, 'checkpoint_layouts'))
    plain = list(dict.fromkeys(initial(neutral, k) for k in
                              ('speed_grouped', 'domain_packed', 'separate', 'speed_first')))
    seeds = plain + [add_backups(neutral, p) for p in plain]
    fallback = replace(plain[0], checkpoint_every=c.checkpoint_every)
    best, scores, summaries = [fallback]*3, [None]*3, [None]*3
    pool, seen, audit = [], set(), []
    attempts = 0
    aware_s = unaware_s = fixed_s = 0.
    aware_calls = unaware_calls = fixed_calls = 0
    search_status = 'layout_limit'
    try:
        while len(pool) < config['checkpoint_layouts'] and attempts < max(100, config['checkpoint_layouts']*30):
            check_budget(effective_stop)
            layout = seeds[attempts] if attempts < len(seeds) else neighbor(neutral, rng.choice(pool or plain), rng)
            attempts += 1
            if layout in seen or violations(c, layout):
                continue
            seen.add(layout)
            block = []
            for interval in intervals:
                q = replace(layout, checkpoint_every=interval)
                tick = time.monotonic()
                try:
                    healthy = simulate(neutral, q, [], effective_stop)
                    unaware_calls += 1
                finally:
                    unaware_s += time.monotonic()-tick
                rows = []
                tick = time.monotonic()
                try:
                    for trace in traces:
                        rows.append(simulate(c, q, trace, effective_stop))
                        aware_calls += 1
                        if interval == c.checkpoint_every:
                            fixed_calls += 1
                finally:
                    elapsed = time.monotonic()-tick
                    aware_s += elapsed
                    if interval == c.checkpoint_every:
                        fixed_s += elapsed
                a, u = summarize(c, rows), summarize(c, [healthy])
                block.append((q, a, u, planning_score(c, a), planning_score(c, u)))
            # Interrupted blocks never contribute to ANY of the three selectors.
            pool.append(layout)
            for q, a, u, a_score, u_score in block:
                audit.append(dict(plan=q.to_dict(), aware_score=a_score, unaware_score=u_score))
                for i, summary, score in [(0, a, a_score), (1, u, u_score), (2, a, a_score)]:
                    if i == 2 and q.checkpoint_every != c.checkpoint_every:
                        continue
                    if scores[i] is None or score < scores[i]:
                        best[i], summaries[i], scores[i] = q, summary, score
        if len(pool) < config['checkpoint_layouts']:
            search_status = 'proposal_limit'
    except BudgetExpired:
        search_status = 'wall_clock_limit'
    wall_s = time.monotonic()-started
    shared_s = max(0., wall_s-aware_s-unaware_s)
    digest = lambda values: hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    layout_hash = digest([p.to_dict() for p in pool])
    candidate_hash = digest([x['plan'] for x in audit])
    fixed_hash = digest([x['plan'] for x in audit if x['plan']['checkpoint_every'] == c.checkpoint_every])
    results = []
    for i, (seconds, calls, name) in enumerate([(aware_s, aware_calls, 'failure_aware_tuned'),
                                               (unaware_s, unaware_calls, 'failure_free_tuned'),
                                               (fixed_s, fixed_calls, 'failure_aware_fixed')]):
        results.append(dict(plan=best[i].to_dict(), planning_s=seconds+shared_s,
                            scoring_s=seconds, shared_overhead_s=shared_s, paired_search_wall_s=wall_s,
                            planning_time_kind='attributed_scoring_plus_shared_overhead',
                            candidates=len(pool)*(1 if i == 2 else len(intervals)),
                            completed_layouts=len(pool), simulations=calls, status=search_status,
                            proposal_attempts=attempts, layout_pool_sha256=layout_hash,
                            candidate_pool_sha256=fixed_hash if i == 2 else candidate_hash,
                            objective=name, search_summary=summaries[i],
                            planning_point_feasible=summaries[i]['point_estimate_feasible'] if summaries[i] else None))
    results[0]['candidate_audit'] = audit
    results[0]['controls'] = dict(checkpoint_unaware=results[1], checkpoint_fixed_search=results[2])
    return results[0]
