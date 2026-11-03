"""Small algorithmic baselines. No solver or learned policy is used in this pilot."""
import random
import time

from .metrics import planning_score, summarize
from .model import Plan, seed_for, violations
from .simulator import BudgetExpired, check_budget, simulate


METHODS = ("speed_first", "reliability_first", "separate", "domain_packed",
           "joint_greedy", "joint_local", "joint_local_no_backup")


def initial(c, kind):
    eligible = [i for i, n in enumerate(c.nodes)
                if n.memory_gib >= c.shard_gib+c.working_gib]
    if kind == "speed_first":
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
    if kind == "domain_packed":
        selected.sort(key=lambda i: (c.nodes[i].domain, key(i)))
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
            q = Plan(p.groups, tuple(sorted(p.backups + ((s, n),))))
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
        return add_backups(c, Plan(tuple(tuple(g) for g in groups)))
    return Plan(tuple(tuple(g) for g in groups), tuple(sorted(backups.items())))


def plan_method(c, method, traces, config, prior, stop_at=None):
    started = time.monotonic()
    if method in ("speed_first", "reliability_first", "separate", "domain_packed"):
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
        return dict(plan=Plan(p.groups).to_dict(), planning_s=time.monotonic()-started,
                    candidates=0, simulations=0, status="placement_matched_ablation",
                    parent="joint_local", parent_planning_s=prior["joint_local"]["planning_s"],
                    search_summary=None)
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
