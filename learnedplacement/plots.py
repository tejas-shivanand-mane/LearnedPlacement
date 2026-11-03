"""Plot saved results only. Importing this module never starts simulations."""
from pathlib import Path


def plot_results(output, active_nodes=8):
    import matplotlib.pyplot as plt
    import pandas as pd
    path = Path(output)/"summary.csv"
    if not path.exists():
        print("No completed comparisons yet; run or resume the experiment first.")
        return None
    df = pd.read_csv(path)
    df = df[df.active_nodes == active_nodes].copy()
    if df.empty:
        print(f"No completed results for {active_nodes} active nodes yet.")
        return None
    ref = df[df.method == "separate"][["case_id", "mean_completion_s", "mean_cost_usd"]]
    paired = df.merge(ref, on="case_id", suffixes=("", "_reference"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for method, group in paired.groupby("method", sort=False):
        valid = group.dropna(subset=["mean_completion_s", "mean_cost_usd", "mean_completion_s_reference", "mean_cost_usd_reference"])
        axes[0].scatter(valid.mean_cost_usd/valid.mean_cost_usd_reference,
                        valid.mean_completion_s/valid.mean_completion_s_reference,
                        label=method, alpha=.7, s=30)
    axes[0].scatter([1], [1], color="black", marker="x", s=80)
    axes[0].set(xlabel="Cost / separate planner cost", ylabel="Time / separate planner time",
                title="Paired cases: lower and left is better")
    timing = df.groupby("method", sort=False).planning_s.median()
    axes[1].scatter(timing.clip(lower=1e-6)*1000, range(len(timing)), s=45)
    axes[1].set_yticks(range(len(timing)), timing.index)
    axes[1].set(xscale="log", xlabel="Median planning time (ms, log scale)", title="Planning cost")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(Path(output)/f"time-cost-n{active_nodes}.png", dpi=160)
    fig2, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    methods = list(df.method.unique())
    for i, method in enumerate(methods):
        values = df[df.method == method].deadline_success_lower95
        ax.scatter(values, [i]*len(values), s=30, alpha=.7)
    ax.set_yticks(range(len(methods)), methods)
    ax.set(xlabel="One-sided 95% lower bound on deadline success",
           title="Each dot is one case; inspect configured targets in the table", xlim=(0, 1.01))
    ax.grid(alpha=.2)
    fig2.savefig(Path(output)/f"deadline-n{active_nodes}.png", dpi=160)
    plt.show()
    # Feasibility is shown explicitly; plots do not silently exclude failed screens.
    return df[["case_id", "method", "mean_completion_s", "mean_cost_usd", "deadline_success",
               "deadline_success_lower95", "statistical_screen_passed", "censored_traces",
               "planning_s", "candidates", "backup_gib"]]
