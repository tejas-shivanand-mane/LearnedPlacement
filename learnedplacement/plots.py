"""Plot saved results only. Importing this module never starts simulations."""
from pathlib import Path


def plot_results(output, active_nodes=8, profile=None, reference_method=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    path = Path(output)/"summary.csv"
    if not path.exists():
        print("No completed comparisons yet; run or resume the experiment first.")
        return None
    df = pd.read_csv(path)
    df = df[df.active_nodes == active_nodes].copy()
    if profile is not None and "profile" in df:
        df = df[df.profile == profile].copy()
    if df.empty:
        print(f"No completed results for {active_nodes} active nodes yet.")
        return None
    reference_method = reference_method or ("separate" if "separate" in set(df.method) else "speed_grouped_tuned")
    ref = df[df.method == reference_method][["case_id", "mean_completion_s", "mean_cost_usd"]]
    paired = df.merge(ref, on="case_id", suffixes=("", "_reference"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for method, group in paired.groupby("method", sort=False):
        valid = group.dropna(subset=["mean_completion_s", "mean_cost_usd", "mean_completion_s_reference", "mean_cost_usd_reference"])
        axes[0].scatter(valid.mean_cost_usd/valid.mean_cost_usd_reference,
                        valid.mean_completion_s/valid.mean_completion_s_reference,
                        label=method, alpha=.7, s=30)
    axes[0].scatter([1], [1], color="black", marker="x", s=80)
    axes[0].set(xlabel=f"Cost / {reference_method} cost", ylabel=f"Time / {reference_method} time",
                title="Paired cases: lower and left is better")
    # Ablations inherit a searched placement; construction time is not deployment
    # planning latency. Their parent latency remains available in plans.json.
    timed = df[~df.method.isin(["joint_local_no_backup", "matched_no_backup", "checkpoint_fixed_ablation", "checkpoint_no_backup"])]
    timing = timed.groupby("method", sort=False).planning_s.median()
    axes[1].scatter(timing.clip(lower=1e-6)*1000, range(len(timing)), s=45)
    axes[1].set_yticks(range(len(timing)), timing.index)
    axes[1].set(xscale="log", xlabel="Median planning time (ms, log scale)", title="Planning cost")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    suffix = f"-{profile}" if profile is not None else ""
    fig.savefig(Path(output)/f"time-cost-n{active_nodes}{suffix}.png", dpi=160)
    fig2, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    methods = list(df.method.unique())
    for i, method in enumerate(methods):
        values = df[df.method == method].deadline_success_lower95
        ax.scatter(values, [i]*len(values), s=30, alpha=.7)
    ax.set_yticks(range(len(methods)), methods)
    ax.set(xlabel="One-sided 95% lower bound on deadline success",
           title="Each dot is one case; inspect configured targets in the table", xlim=(0, 1.01))
    ax.grid(alpha=.2)
    fig2.savefig(Path(output)/f"deadline-n{active_nodes}{suffix}.png", dpi=160)
    plt.show()
    # Feasibility is shown explicitly; plots do not silently exclude failed screens.
    return df[["case_id", "method", "mean_completion_s", "mean_cost_usd", "deadline_success",
               "deadline_success_lower95", "statistical_screen_passed", "censored_traces",
               "planning_s", "candidates", "backup_gib"]]


def plot_failure_effects(output, active_nodes=8, profile="reference",
                         selected="matched_failure_aware", baseline="matched_failure_unaware"):
    """Saved-result decomposition: total difference = healthy + failure penalty."""
    import matplotlib.pyplot as plt
    import pandas as pd
    path = Path(output)/"summary.csv"
    if not path.exists():
        print("No saved results yet.")
        return None
    df = pd.read_csv(path)
    if "failure_free_completion_s" not in df or "profile" not in df:
        print("This plot requires a v2 run with failure-free measurements.")
        return None
    df = df[(df.active_nodes == active_nodes) & (df.profile == profile)]
    paired = df[df.method == selected].merge(df[df.method == baseline], on="case_id", suffixes=("_a", "_b"))
    paired = paired.dropna(subset=["mean_completion_s_a", "mean_completion_s_b",
                                  "failure_free_completion_s_a", "failure_free_completion_s_b"])
    if paired.empty:
        print("No complete, uncensored pairs for this selection yet.")
        return None
    paired["total_difference_s"] = paired.mean_completion_s_a-paired.mean_completion_s_b
    paired["healthy_difference_s"] = paired.failure_free_completion_s_a-paired.failure_free_completion_s_b
    paired["failure_penalty_difference_s"] = paired.total_difference_s-paired.healthy_difference_s
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, .4*len(paired))), sharey=True, layout="constrained")
    for ax, column, title in zip(axes, ["total_difference_s", "healthy_difference_s", "failure_penalty_difference_s"],
                                 ["Total time", "Failure-free time", "Additional time from failures"]):
        ax.scatter(paired[column], range(len(paired)), s=35)
        ax.set(xlabel="Difference (s); negative favors selected", title=title)
        ax.grid(alpha=.2)
    axes[0].set_yticks(range(len(paired)), paired.case_id)
    fig.suptitle(f"{selected} vs {baseline}")
    fig.savefig(Path(output)/f"failure-effects-n{active_nodes}-{profile}-{baseline}.png", dpi=160)
    plt.show()
    return paired[["case_id", "total_difference_s", "healthy_difference_s", "failure_penalty_difference_s",
                   "statistical_screen_passed_a", "statistical_screen_passed_b"]]
