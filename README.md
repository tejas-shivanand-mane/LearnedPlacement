# LearnedPlacement

A local, CPU-only experiment for **joint placement and protection of sharded
training state**. It models a fixed-work HSDP-style job. It needs no Ray, GPU,
Z3, cloud credentials, or paid service. It does not train a learned policy or
execute real distributed training.

## Run the second experiment

```bash
cd LearnedPlacement
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[notebook]'
python -m jupyter lab notebooks/PlacementPilot.ipynb
```

For a fresh checkout, clone
`https://github.com/tejas-shivanand-mane/LearnedPlacement.git`, enter the directory,
and create `.venv` with `python3 -m venv .venv` first. Python 3.10+ is required.

**Restart the notebook kernel after pulling.** Run the updated notebook in order.
Its default output is `results/pilot_v2`; previous results remain in `pilot_v1`.
The Run/resume cell starts the experiment. Settings are visible/editable.
Plotting cells read saved results only. No experiment runs on import or in CI.

The standard-library command-line equivalent is:

```bash
python -m learnedplacement.runner --config configs/pilot_v2.json --output results/pilot_v2 --max-minutes 30
python -m learnedplacement.runner --output results/pilot_v2 --status
```

Each invocation has a **30-minute cooperative wall-clock limit**, including
planning and evaluation. A brief final save may follow the limit. This is not a
promise that the whole experiment finishes in 30 minutes. Repeat the same
cell/command to resume. You can change the invocation time limit when resuming.

## Why v2

The first pilot suggested that grouping machines with similar speeds accounted
for much of the time improvement; additional backups did not demonstrate a
benefit. V2 adds controls to separate normal throughput from failure effects.
It does not assume that failure-aware placement or extra backups will win.

The main question is: **Does accounting for failures when choosing a plan reduce
total completion time under failures, beyond a strong speed-based placement?**

| Method | What it does |
|---|---|
| `speed_first` | Select fastest machines; spread domains within groups. |
| `separate` | Select by a price/speed/risk heuristic; then consider extra backups. |
| `domain_packed` | Select reliable machines; pack domains within groups. |
| `speed_grouped` | Select fastest machines, then group adjacent speeds to avoid mixed-speed bottlenecks. |
| `matched_failure_aware` | Rank a shared candidate pool using failure planning traces. |
| `matched_failure_unaware` | Rank the exact same pool assuming no failures. |
| `matched_no_backup` | Preserve the aware plan's exact placement and remove its extra backups. |

Each of the first four baselines scores its plain plan and a heuristic protected
variant on the planning traces, allowing it to reject unhelpful backups.
`speed_grouped` is therefore not the strictly failure-unaware control; the
matched search provides that control.

The matched pair uses risk-neutral seeds and random valid moves. Candidate
generation sees **zero failure hazards, no scoring outcomes, and no held-out
traces**. It seeds speed-grouped, speed-spread, price/speed and domain-packed
layouts, with and without heuristic backups, then proposes swaps, machine
replacements and backup changes. Both scorers finish a candidate before either
incumbent can change. An interrupted candidate is excluded from both choices.
The saved audit lists completed candidates and both objective scores; a pool
hash verifies pairing. If none completes, a valid speed-grouped fallback is
marked with zero candidates and unknown planning feasibility.

This is bounded shared-pool search, **not adaptive hill climbing or an exact
optimizer**. The control isolates scoring under failures within this pool; it
cannot establish the best achievable benefit of failure-aware search.
Budgets match candidate choices, not CPU effort: the aware objective uses 32
traces per candidate; the deterministic unaware objective needs one failure-free
simulation. Each records scorer time plus shared generation/bookkeeping time.
The pair's actual combined wall time is also recorded; the two attributed times
must not be added as a wall-time measurement. Ablation construction time excludes
its parent's search and is omitted from the planning-latency plot.

## Configuration

`configs/pilot_v2.json` and the notebook expose:

| Setting | Default |
|---|---|
| Active machines / candidate machines | 8 / 12 or 16 / 24 |
| Groups × shards | 2 × 4 or 4 × 4 |
| Machine regimes | homogeneous, cheap/risky, fast/risky, correlated outages |
| Instances per size/regime | 2 seeded instances |
| Profiles | reference, tight, recovery_stress |
| Total cases | 48 |
| Held-out evaluations | 48 × 7 × 256 = 86,016 |
| Planning traces | 32, separate from held-out traces |
| Matched search limit | 64 completed candidates or 30 s for the pair per case |
| Work | 240 steps; 3 s unit-speed compute per healthy step |
| State / working memory | 16 GiB over 4 shard positions / 4 GiB per active machine |
| Extra backup storage | 8 GiB; at most one extra snapshot per shard |
| Backup / durable checkpoint interval | 20 / 80 committed steps |
| Required deadline success | 0.95 |

The three profiles are:

| Profile | Deadline factor | Cost factor | Durable storage | Replacement | Reinitialization |
|---|---:|---:|---:|---:|---:|
| `reference` | 1.75 | 1.50 | 2 GiB/s | 30 s | 4 s |
| `tight` | 1.15 | 1.00 | 2 GiB/s | 30 s | 4 s |
| `recovery_stress` | 1.15 | 1.00 | 0.25 GiB/s | 120 s | 16 s |

These are **synthetic sensitivity assumptions**, not measured hardware/service
parameters. Tight budgets reduce the slack observed in v1 without guaranteeing
feasibility. The stress profile tests a joint increase in recovery costs: an
8× slower durable tier and 4× longer replacement/reinitialization. It cannot
attribute a change to one of those factors individually. Checkpoint cadence is
fixed across profiles, not optimized to favor backups. No profile guarantees
that extra protection pays for itself. Calibrate against real measurements
before making systems performance claims.

Hardware is identical across profiles for each size/regime/instance. Profiles
reuse failure-seed keys, coupling outage-start randomness; replacement durations
and horizons can differ. Within each case, every frozen plan sees exactly the
same held-out event stream. Proposal randomness is also reused across profiles,
though time caps can shorten the completed prefix differently.

Machine hazards remain the original synthetic stress rates: 0.12/hour ordinary,
2/hour risky, and 0.2/hour plus 1 domain outage/hour in the correlated regime.
`model.make_case()` contains the editable generator. The notebook previews case
budgets before running. A smaller fresh run can select only the reference
profile; configuration changes always require a new output directory.

## Objective, constraints and measurements

The objective is **average total completion time including failures**, not
failure-free training time or recovery time alone. It includes useful work,
backup/checkpoint overhead, detection, waiting, recovery, reduced throughput and
recomputation. Cost is a constraint and final tie-breaker.

Candidates must satisfy structural assignment, per-node memory (including
backup staging), and extra backup storage constraints. Search ranks by:

1. Smaller noncompletion fraction at the finite simulation horizon.
2. Smaller normalized cost/deadline-success constraint violations.
3. Lower restricted mean completion time.
4. Lower observed mean cost.

No feasible candidate is guaranteed. Planning uses point estimates; frozen plans
are evaluated independently on 256 new traces. The final statistical screen
requires no censoring, a one-sided 95% Wilson lower bound on deadline success
at least 0.95, and an approximate one-sided 95% normal upper bound on mean cost
within budget. These are separate per-plan screens, not simultaneous guarantees,
distribution-free cost bounds, or guarantees under model error.

Budgets use a planner-independent unit-speed, $1/hour-per-machine reference,
including that profile's checkpoint overhead. The table's factors multiply its
time and cost. The simulation horizon is 8× reference time. Monetary cost is
reserved capacity × elapsed time, including downtime and backup-only machines;
persistent storage and egress fees are excluded.

Each selected plan also gets one **failure-free simulation of the exact same
plan**, retaining checkpoint and backup costs. This enables:

```
failure penalty = average completion time under failures - failure-free time
```

For two plans, total time difference equals failure-free difference plus failure
penalty difference. The penalty includes all failure-induced changes, not just
time labeled `recovery`. It is a counterfactual model difference, not a separate
simulator phase. Paired bootstrap reports include intervals for total and penalty
differences; subtracting the deterministic healthy difference shifts the interval.

Inspect both plans' feasibility screens, candidate counts and costs. A faster
infeasible plan is not a constrained-planning win. Null means indicate censoring;
restricted means and observed costs are retained, and censored cases are not
silently called completions. Diagnostic plots require uncensored pairs; the full
results table retains all cases. No test-set plan selection is performed.

## Saved output and resume

Under `results/pilot_v2/`:

- `manifest.json`, `source/`: exact config, source hashes/snapshot and environment.
- `status.json`: completed/paused/error state and current comparison.
- `cases/<case>/case.json`, `plans.json`: inputs, plans, scores, timing and candidate audit.
- `<method>.observations.json`: individual outcomes and paired replay seeds.
- `<method>.failure_free.json`: full failure-free observation for the frozen plan.
- `<method>.summary.json`, root `summary.csv`/`summary.json`: completed evaluations.
- Paired reports and figures generated by the notebook; its final cell exports a compact analysis ZIP.

Completed work is skipped. Observations save every 8 traces and on clean
interrupt/pause. A killed process may lose its last unsaved batch. A globally
interrupted planner restarts on resume. The `.run.lock` prevents competing
writers; remove a stale lock only after confirming the old process stopped.

Changed config or core source requires a **new output directory**. Existing v1
runs can still be plotted/read; they cannot resume under changed source hashes.
`configs/pilot.json` retains the first experiment's settings and legacy planners
for a fresh historical-style run, not a bit-identical v1 source reproduction.
Source snapshots in exported archives preserve the original implementation.

## Correctness and limitations

```bash
python -m unittest discover -s tests -v
```

Tests use small deterministic fixtures, including state loss, rollback,
version-consistent recovery, backup benefits/costs, shared candidate limits,
interruption, resume and diagnostic accounting. They do not run the full pilot.

See [MODEL.md](MODEL.md) for the state and performance contract. This is a
synthetic HSDP-style surrogate with fixed blocking recovery policies, simplified
collectives, and no actual tensors or convergence validation. It is not an
implementation or comparison of Oobleck, Bamboo, Parcae or LACHESIS, and does not
establish novelty or real-world speedups. Learning should be tested only after
repeatable headroom over strong cheap baselines is established.

## Compact sharing export

`export_run(OUTPUT)` now creates `pilot_v2.analysis.zip` containing one
`analysis.json`. It includes the original manifest/config/source hashes, case
inputs, selected plans and planning metrics, completed summaries, failure-free
observations, and paired mean/variance/standard-error statistics for every
available method pair. Paired intervals in this bundle use a two-sided 95%
normal approximation; they are not the bootstrap intervals from `paired_report`.
Partial runs compare only the shared saved trace prefix. Censored pairs compare
restricted time and observed cost and are explicitly labeled.

Raw per-trace observations, full candidate audits, source copies and figures
remain local. The bundle supports routine comparisons and planning the next
experiment, but cannot support arbitrary raw-distribution reanalysis or exact
replay. Use `export_run(OUTPUT, full=True)` only when a full debugging archive is
needed. Neither export changes local results or executes simulations. Compact
exports cannot be used as resumable run directories.

For an already completed run, pull the update, restart the notebook kernel, and
run only setup, settings and the final export cell. No experiment rerun or
plot/comparison-cell execution is needed.
