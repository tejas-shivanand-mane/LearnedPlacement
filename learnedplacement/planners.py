"""Small algorithmic baselines. No solver or learned policy is used in this pilot."""
from dataclasses import replace
import hashlib
import json
import random
import time

from .metrics import planning_score, summarize
from .model import Plan, seed_for, violations
from .simulator import BudgetExpired, check_budget, simulate


METHODS = ("speed_first", "reliability_first", "separate", "domain_packed",
           "joint_greedy", "joint_local", "joint_local_no_backup")
SIMPLE_METHODS = METHODS[:4] + ("speed_grouped",)
MATCHED_METHODS = ("matched_failure_aware", "matched_failure_unaware", "matched_no_backup")
CHECKPOINT_METHODS = ("speed_grouped_tuned", "domain_packed_tuned", "checkpoint_joint",
                      "checkpoint_unaware", "checkpoint_fixed_search",
                      "checkpoint_fixed_ablation", "checkpoint_no_backup")
ALL_METHODS = METHODS + ("speed_grouped",) + MATCHED_METHODS + CHECKPOINT_METHODS


def initial(c, kind):
    eligible = [i for i, n in enumerate(c.nodes)
                if n.memory_gib >= c.shard_gib+c.working_gib]
    if kind in ("speed_first", "speed_grouped"):
        key = lambda i: (-c.nodes[i].speed, c.nodes[i].price_per_hour, i)
    elif kind in ("reliability_first", "domain_packed"):
        key = lambda i: (c.nodes[i].failure_per_hour, -c.nodes[i].speed,
                         c.nodes[i].price_per_hour, i)
    else:
        key = lambda i: ((1 + c.nodes[i].failure_per_hour*c.replacement_s/3600)
                         * c.nodes[i].price_per_hour / c.nodes[i].speed, i)
    selected = sorted(eligible, key=key)[:c.replicas*c.shards]
    if len(selected) != c.replicas*c.shards:
        raise ValueError("No structurally valid initial training layout")
    if kind in ("domain_packed", "speed_grouped"):
        if kind == "domain_packed":
            selected.sort(key=lambda i: (c.nodes[i].domain, key(i)))
        # speed_grouped retains descending speed order: similar-speed machines
        # share a group instead of spreading the fastest across bottlenecks.
        groups = [selected[i*c.shards:(i+1)*c.shards] for i in range(c.replicas)]
    else:
        groups = [[] for _ in range(c.replicas)]
        for n in selected:
            possible = [r for r in range(c.replicas) if len(groups[r]) < c.shards]
            # Spread domains within groups; domain_packed is the explicit contrast.
            r = min(possible, key=lambda r: (sum(c.nodes[k].domain == c.nodes[n].domain
                                                 for k in groups[r]), len(groups[r]), r))
            groups[r].append(n)
    return Plan(tuple(tuple(g) for g in groups))


def add_backups(c, p):
    """Separate heuristic: protect high-exposure shards, prefer domain diversity.

    This score is a heuristic, not a shard-loss probability model. The simulator
    is the common evaluator for every method.
    """
    score = lambda s: sum(c.nodes[g[s]].failure_per_hour for g in p.groups)
    for s in sorted(range(c.shards), key=lambda s: (-score(s), s)):
        domains = {c.nodes[g[s]].domain for g in p.groups}
        candidates = []
        for n, node in enumerate(c.nodes):
            q = replace(p, backups=tuple(sorted(p.backups + ((s, n),))))
            if not violations(c, q):
                candidates.append(((node.domain in domains, node.failure_per_hour,
                                    node.price_per_hour if n not in p.active else 0., n), q))
        if candidates:
            p = min(candidates, key=lambda item: item[0])[1]
    return p


def neighbor(c, p, rng):
    groups = [list(g) for g in p.groups]
    backups = dict(p.backups)
    operation = rng.randrange(4)
    if operation == 0:
        # Swap shard positions, within or across groups.
        a, b = rng.sample(range(c.replicas*c.shards), 2)
        r, s = divmod(a, c.shards)
        u, v = divmod(b, c.shards)
        groups[r][s], groups[u][v] = groups[u][v], groups[r][s]
    elif operation == 1:
        spares = sorted(set(range(len(c.nodes))) - p.active)
        if spares:
            r, s = rng.randrange(c.replicas), rng.randrange(c.shards)
            groups[r][s] = rng.choice(spares)
    elif operation == 2:
        s = rng.randrange(c.shards)
        if s in backups and rng.random() < .4:
            del backups[s]
        else:
            backups[s] = rng.randrange(len(c.nodes))
    else:
        # Coupled move: change placement, then reconstruct protection for it.
        a, b = rng.sample(range(c.replicas*c.shards), 2)
        r, s = divmod(a, c.shards)
        u, v = divmod(b, c.shards)
        groups[r][s], groups[u][v] = groups[u][v], groups[r][s]
        return add_backups(c, replace(p, groups=tuple(tuple(g) for g in groups), backups=()))
    return replace(p, groups=tuple(tuple(g) for g in groups), backups=tuple(sorted(backups.items())))


def plan_method(c, method, traces, config, prior, stop_at=None):
    started = time.monotonic()
    if method in CHECKPOINT_METHODS:
        from .checkpoint import plan_checkpoint_method
        return plan_checkpoint_method(c, method, traces, config, prior, stop_at)
    if method == "matched_failure_aware":
        return matched_search(c, traces, config, stop_at)
    if method == "matched_failure_unaware":
        return prior["matched_failure_aware"]["unaware_result"]
    if method == "matched_no_backup":
        parent = prior["matched_failure_aware"]
        p = Plan.from_dict(parent["plan"])
        return dict(plan=replace(p, backups=()).to_dict(), planning_s=time.monotonic()-started,
                    candidates=0, simulations=0, status="placement_matched_ablation",
                    parent="matched_failure_aware", parent_planning_s=parent["planning_s"],
                    search_summary=None)
    if method in SIMPLE_METHODS:
        plain = initial(c, method)
        protected = add_backups(c, plain)
        # Give simple baselines a fair chance to reject unnecessary backups.
        # This uses planning traces only, never the held-out evaluation set.
        candidates = list(dict.fromkeys([plain, protected]))
        scored = []
        for candidate in candidates:
            rows = [simulate(c, candidate, trace, stop_at) for trace in traces]
            summary = summarize(c, rows)
            scored.append((planning_score(c, summary), candidate, summary))
        _, p, summary = min(scored, key=lambda item: item[0])
        return dict(plan=p.to_dict(), planning_s=time.monotonic()-started,
                    candidates=len(candidates), simulations=len(candidates)*len(traces),
                    status="heuristic_with_protection_toggle", search_summary=summary,
                    planning_point_feasible=summary["point_estimate_feasible"])
    if method == "joint_local_no_backup":
        p = Plan.from_dict(prior["joint_local"]["plan"])
        return dict(plan=replace(p, backups=()).to_dict(), planning_s=time.monotonic()-started,
                    candidates=0, simulations=0, status="placement_matched_ablation",
                    parent="joint_local", parent_planning_s=prior["joint_local"]["planning_s"],
                    search_summary=None)
    if method not in ("joint_greedy", "joint_local"):
        raise ValueError(f"Unknown method: {method}")
    limit = config["greedy_candidates"] if method == "joint_greedy" else config["local_candidates"]
    seconds = config["greedy_seconds"] if method == "joint_greedy" else config["local_seconds"]
    local_stop = started + seconds
    effective_stop = min(local_stop, stop_at) if stop_at is not None else local_stop
    rng = random.Random(seed_for(config["seed"], c.case_id, method))
    seeds = [Plan.from_dict(prior[k]["plan"]) for k in prior
             if k in ("speed_first", "reliability_first", "separate", "domain_packed", "joint_greedy")]
    seeds += [Plan(p.groups) for p in seeds]
    best = Plan.from_dict(prior["separate"]["plan"])
    best_summary = None
    best_score = None
    seen = set()
    evaluated = simulations = attempts = 0
    status = "candidate_limit"
    try:
        while evaluated < limit and attempts < max(100, limit*30):
            check_budget(effective_stop)
            if attempts < len(seeds):
                q = seeds[attempts]
            else:
                base = Plan.from_dict(prior["separate"]["plan"]) if method == "joint_greedy" else best
                q = neighbor(c, base, rng)
            attempts += 1
            if q in seen or violations(c, q):
                continue
            seen.add(q)
            rows = []
            for trace in traces:
                rows.append(simulate(c, q, trace, effective_stop))
                simulations += 1
            summary = summarize(c, rows)
            score = planning_score(c, summary)
            evaluated += 1
            if best_score is None or score < best_score:
                best, best_score, best_summary = q, score, summary
        if evaluated < limit:
            status = "proposal_limit"
    except BudgetExpired:
        # If the whole runner expired, its caller must persist the completed
        # incumbent before returning. No partially evaluated candidate is used.
        status = "wall_clock_limit"
    return dict(plan=best.to_dict(), planning_s=time.monotonic()-started,
                candidates=evaluated, simulations=simulations, proposal_attempts=attempts,
                status=status, search_summary=best_summary,
                planning_point_feasible=best_summary["point_estimate_feasible"] if best_summary else None)


def matched_search(c, traces, config, stop_at=None):
    """Two objectives, exactly the same completed candidate prefix.

    Proposals see zero hazards, no planning outcomes, and no held-out traces.
    Both scorers finish a candidate before either incumbent can change. Thus
    a wall-clock interruption cannot give one scorer additional candidates.
    This is bounded shared-pool search, not adaptive hill climbing or an oracle.
    """
    started = time.monotonic()
    local_stop = started + config["matched_seconds"]
    effective_stop = min(local_stop, stop_at) if stop_at is not None else local_stop
    neutral = replace(c, nodes=[replace(n, failure_per_hour=0.) for n in c.nodes],
                      domain_failure_per_hour=0.)
    rng = random.Random(seed_for(config["seed"], c.trace_key or c.case_id, "matched_candidates"))
    plain_seeds = list(dict.fromkeys(initial(neutral, k) for k in
                                    ("speed_grouped", "speed_first", "separate", "domain_packed")))
    seeds = plain_seeds + [add_backups(neutral, p) for p in plain_seeds]
    limit = config["matched_candidates"]
    pool, seen, audit = [], set(), []
    best = [plain_seeds[0], plain_seeds[0]]
    summaries, scores = [None, None], [None, None]
    scoring_s, simulations = [0., 0.], [0, 0]
    attempts = 0
    search_status = "candidate_limit"
    try:
        while len(pool) < limit and attempts < max(100, limit*30):
            check_budget(effective_stop)
            q = seeds[attempts] if attempts < len(seeds) else neighbor(neutral, rng.choice(pool or plain_seeds), rng)
            attempts += 1
            if q in seen or violations(c, q):
                continue
            seen.add(q)
            tick = time.monotonic()
            healthy = simulate(neutral, q, [], effective_stop)
            scoring_s[1] += time.monotonic()-tick
            simulations[1] += 1
            tick = time.monotonic()
            rows = []
            try:
                for trace in traces:
                    rows.append(simulate(c, q, trace, effective_stop))
                    simulations[0] += 1
            finally:
                scoring_s[0] += time.monotonic()-tick
            pair = [summarize(c, rows), summarize(c, [healthy])]
            candidate_scores = [planning_score(c, s) for s in pair]
            pool.append(q)
            audit.append(dict(plan=q.to_dict(), aware_score=candidate_scores[0],
                              unaware_score=candidate_scores[1]))
            for i in range(2):
                if scores[i] is None or candidate_scores[i] < scores[i]:
                    best[i], scores[i], summaries[i] = q, candidate_scores[i], pair[i]
        if len(pool) < limit:
            search_status = "proposal_limit"
    except BudgetExpired:
        search_status = "wall_clock_limit"
    elapsed = time.monotonic()-started
    digest = hashlib.sha256(json.dumps([p.to_dict() for p in pool], sort_keys=True).encode()).hexdigest()
    # Timing is scorer work plus shared generation/bookkeeping, not a claim of
    # equal CPU effort. The deterministic healthy objective needs one simulation.
    shared_s = max(0., elapsed-sum(scoring_s))
    results = [dict(plan=best[i].to_dict(), planning_s=scoring_s[i]+shared_s,
                    scoring_s=scoring_s[i], shared_overhead_s=shared_s,
                    paired_search_wall_s=elapsed, candidates=len(pool), simulations=simulations[i],
                    status=search_status, proposal_attempts=attempts,
                    candidate_pool_sha256=digest,
                    objective="failure_aware" if i == 0 else "failure_free",
                    search_summary=summaries[i],
                    planning_point_feasible=summaries[i]["point_estimate_feasible"] if summaries[i] else None)
               for i in range(2)]
    results[0]["unaware_result"] = results[1]
    results[0]["candidate_audit"] = audit
    return results[0]
