"""Monte Carlo estimates, explicitly distinguished from feasibility proofs."""
import math
import statistics


def wilson_lower(hits, count, z=1.6448536269514722):
    """One-sided 95% Wilson bound for an independently evaluated fixed plan."""
    if not count:
        return 0.
    p = hits / count
    return (p + z*z/(2*count) - z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))) / (1 + z*z/count)


def summarize(c, rows):
    n = len(rows)
    if not n:
        return {"traces": 0}
    completed = sum(r["completed"] for r in rows)
    hits = sum(r["deadline_hit"] for r in rows)
    costs = [r["cost_usd"] for r in rows]
    cost_mean = statistics.mean(costs)
    # Normal approximation for the Monte Carlo mean; not a distribution-free bound.
    cost_upper = cost_mean + 1.6448536269514722 * statistics.stdev(costs) / math.sqrt(n) if n > 1 else None
    lower = wilson_lower(hits, n)
    complete = completed == n
    point = complete and cost_mean <= c.cost_budget_usd and hits/n >= c.required_success
    screening = complete and cost_upper is not None and cost_upper <= c.cost_budget_usd and lower >= c.required_success
    times = sorted(r["observed_s"] for r in rows)
    return dict(traces=n, completed_fraction=completed/n, censored_traces=n-completed,
                mean_completion_s=statistics.mean(times) if complete else None,
                restricted_mean_s=statistics.mean(times),
                p95_observed_s=times[max(0, math.ceil(.95*n)-1)],
                mean_cost_usd=cost_mean if complete else None,
                mean_observed_cost_usd=cost_mean,
                approximate_cost_mean_upper95=cost_upper if complete else None,
                deadline_success=hits/n, deadline_success_lower95=lower,
                point_estimate_feasible=point, statistical_screen_passed=screening,
                mean_lost_steps=statistics.mean(r["lost_steps"] for r in rows),
                mean_rollbacks=statistics.mean(r["rollbacks"] for r in rows),
                mean_phase_seconds={k: statistics.mean(r["phase_seconds"][k] for r in rows)
                                    for k in rows[0]["phase_seconds"]})


def planning_score(c, summary):
    # Point estimates only during candidate search; final evidence uses fresh traces.
    # Censored observations never qualify as feasible or as expected completion time.
    violation = (max(0., summary["mean_observed_cost_usd"]/c.cost_budget_usd-1)
                 + max(0., c.required_success-summary["deadline_success"])/c.required_success)
    return (1-summary["completed_fraction"], violation,
            summary["restricted_mean_s"], summary["mean_observed_cost_usd"])
