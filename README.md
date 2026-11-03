# LearnedPlacement

A local, CPU-only pilot for **joint machine placement and protection of sharded
training state**. This first experiment compares inexpensive planners in a
synthetic HSDP-style simulator. It does **not** train an RL model or execute real
distributed training. It needs no Ray, GPU, Z3, cloud credentials, or paid service.

## Run locally

Python 3.10 or newer:

```bash
git clone https://github.com/tejas-shivanand-mane/LearnedPlacement.git
cd LearnedPlacement
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook]'
python -m jupyter lab notebooks/PlacementPilot.ipynb
```

If you already cloned it, use `git pull` inside that checkout. Open the notebook
and run its cells in order. Settings are visible and editable. The **Run/resume**
cell is the only cell that starts the experiment. Plotting cells read saved data
and can be run independently after setup. There is no automatic experiment or
CI benchmark.

The core simulator and command-line runner use only the Python standard library:

```bash
python -m learnedplacement.runner --config configs/pilot.json --output results/pilot --max-minutes 30
python -m learnedplacement.runner --output results/pilot --status
```

The default is **24 cases × 7 planners × 256 held-out failure traces**, plus
planning evaluations. A 30-minute **cooperative wall-clock limit** pauses work;
it is not a promise that the full pilot finishes in 30 minutes. Rerun the same
cell/command to resume. A brief final save can occur after the limit.

To run the small correctness checks (not the pilot):

```bash
python -m unittest discover -s tests -v
```

## The question

Does choosing machines, replica groups, shard positions and additional backup
locations together improve completion time over deciding placement first and
protection afterward, under cost and reliability constraints?

We first measure headroom using heuristics and bounded search. The simulator and
structured plans can later support imitation learning or RL. No optimal labels
are required; no result in this pilot is claimed to be optimal.

## Default experiment

| Setting | Value |
|---|---|
| Training layouts | 2 groups × 4 shards; 4 groups × 4 shards |
| Active machines | 8 or 16 |
| Candidate machine pool | 12 or 24 |
| Regimes | homogeneous, cheap/risky, fast/risky, correlated-domain outages |
| Instances per size/regime | 3 independent seeds |
| Training state | 16 GiB across 4 shard positions, replicated across groups |
| Working memory | 4 GiB per training machine |
| Workload | 240 global optimizer steps; 3 s unit-speed compute per healthy step |
| Extra backup budget | 8 GiB; up to one additional snapshot per shard |
| Backup refresh | Every 20 committed steps, including initial version 0 |
| Persistent checkpoint | Every 80 committed steps; initial version 0 is durable |
| Planning/evaluation traces | 32 / 256, disjoint seed streams |
| Joint-greedy budget | At most 16 candidates and 5 s per case |
| Joint-local budget | At most 64 candidates and 15 s per case |

All values are assumptions for this pilot, **not measured cloud/GPU parameters**.
`configs/pilot.json` controls the workload, trace counts, limits and random seed.
`model.make_case()` contains the short, editable machine generator. The notebook
shows a generated case before starting the run.

Homogeneous instances have the same hardware; their failure traces differ by
instance. Other regimes also have small seeded speed variations. The synthetic
hazards are 0.12/hour for ordinary machines, 2/hour for risky machines, and
0.2/hour plus 1 domain event/hour in the correlated stress regime. Those rates
intentionally expose failure behavior in a short simulated job. They must not
be presented as an empirical failure distribution. Run sensitivity studies and
calibrate against real measurements before drawing systems conclusions.

## Planners

| Name | Method |
|---|---|
| `speed_first` | Select fast machines; spread domains within groups. |
| `reliability_first` | Select reliable machines; spread domains within groups. |
| `separate` | Select by a fixed price/speed/risk score, then choose backups separately. |
| `domain_packed` | Select reliable machines, then pack domains within groups. |
| `joint_greedy` | Score baseline seeds and valid one-move alternatives around `separate`. |
| `joint_local` | Score seeds, then hill-climb using swaps, replacement machines, backup edits and coupled moves. |
| `joint_local_no_backup` | Keep the selected `joint_local` placement exactly, remove only additional backups. |

Each simple baseline compares **no extra backups** against its separate backup
heuristic using the same planning traces. This prevents learning/search from
winning merely because a baseline is forced to pay for useless redundancy.
The backup heuristic prioritizes exposed shards and favors hosts outside the
domains containing their existing replicas, with reliable/cheap hosts preferred.
It is a heuristic score, not a probability theorem.

`joint_greedy` and `joint_local` evaluate complete candidates with the common
simulator. Candidate limits and time limits both apply. Only fully evaluated
candidates are retained. Invalid proposals are rejected with a finite attempt
cap. If time expires before any full candidate is scored, the recorded fallback
is the separate baseline; it is **not assumed feasible**.

These are original pilot baselines, not implementations of Oobleck, Bamboo,
Parcae, Chameleon, or LACHESIS. Comparing them does not establish superiority to
those systems. The first pilot does not enforce equal planning budgets across
all methods; it records the actual candidate counts, simulations, and latency.
For a paper, compare quality–latency curves and stronger tuned baselines.

## Execution and recovery semantics

See [MODEL.md](MODEL.md) for the exact surrogate, its limitations, and the state
consistency assumptions. The most important points are:

* Existing replica groups already contain equivalent training state and are
  available for peer recovery. Extra backups are genuinely additional.
* A machine failure destroys its primary and backup state. Other local shards
  of that group survive until their hosts fail; they can be used as recovery
  sources even when the group cannot execute.
* All shards used for a restored model must have the **same committed version**.
  The simulator never silently combines incompatible versions.
* Healthy groups continue with a fixed sample count per optimizer step. Fewer
  groups do more work each. Rejoining groups are synchronized before execution.
* Detection, restoring groups and snapshot writes block progress in this first
  model. These conservative policies are fixed across planners.
* Failures occur at continuous simulated times, including during training,
  snapshot writes, and recovery. Failed atomic operations cannot commit.
* If the whole run exceeds its configured simulation horizon, it is recorded
  as **censored**, not as completing at the horizon.

## Objective and constraints

The planning objective is expected completion time under:

1. Unique machine assignment and fixed group dimensions.
2. Per-node memory, including a staging buffer for each extra backup.
3. A total extra backup budget.
4. An expected monetary-cost budget.
5. A deadline-success target (default 0.95).

For the default generator, budgets are derived from a fixed unit-speed, $1/hour
reference configuration. The reference includes the shared checkpoint policy.
Deadline = 1.75 × reference time; cost budget = 1.5 × reference cost. The finite
simulation horizon is 8 × reference time.

Planning uses point estimates from its 32 traces. Candidate ranking first
penalizes noncompletion and normalized constraint violations, then completion
time and cost. A candidate may still be infeasible; this is explicitly reported.

Final reporting uses 256 **new** traces. `statistical_screen_passed` requires no
censoring, a one-sided 95% Wilson lower bound on deadline success above the
target, and an approximate one-sided 95% upper bound on mean cost below budget.
The cost bound uses a normal approximation to the sample mean. These are
separate per-plan screens, **not joint guarantees, simultaneous guarantees
across all cases, or guarantees under model error**. Do not select/retrain on
these held-out results and then call them untouched test data. With 32 traces,
even zero deadline misses would not certify 95% success at this confidence;
that is why planning does not use the final confidence screen.

Cost is allocated capacity × wall time, including downtime and backup-only
machines. Replacements retain their slot's price. This is a reserved-capacity
proxy, not exact spot-instance billing. Persistent storage/egress fees are
excluded; persistent I/O time is included.

## Saved output, reproducibility and resume

Under `results/pilot/`:

* `manifest.json`: exact config, source hashes, Python/platform metadata.
* `source/`: a snapshot of the simulator and planners used for that run.
* `status.json`: current case/method and whether finished, paused or errored.
* `cases/<case>/case.json`: all machine/workload assumptions.
* `cases/<case>/plans.json`: selected plans, planning metrics and scores.
* `cases/<case>/<method>.observations.json`: per-trace outcomes and replay seeds.
* `cases/<case>/<method>.summary.json`: completed evaluation results.
* `summary.json`, `summary.csv`: completed comparisons across cases.
* Plot and paired-comparison files when their notebook cells are run.

Repeated invocations skip completed work. Observations save every 8 traces and
on a clean interrupt/time pause. A killed process can lose the last unsaved
batch. The `.run.lock` file prevents competing writers: after a killed kernel,
confirm the old process has stopped before deleting that file. Normal pauses
remove it automatically. Changed configuration or core source requires a **new
output directory**, so incompatible results cannot silently mix.

Fixed seeds make cases and traces reproducible. Time-limited search can examine
different numbers of candidates on different machines; saved plans are the
authoritative output. A saved observation can be reproduced with `generate_trace`
and its `trace_seed`. Resume preserves completed plans. A planner interrupted
by the overall runner budget is restarted on resume rather than reported as a
full-budget result.

The notebook exports a ZIP containing configuration, source, plans and results.
Share that ZIP for analysis. No data are uploaded automatically.

## Interpretation

Use the time–cost scatter **and** feasibility table. A lower time with higher
cost is a trade-off, not an unconditional improvement. A failed reliability or
cost screen is not silently excluded from the table. Censored runs have null
mean completion/cost fields; their restricted mean time and observed cost are
reported separately. Do not average only successful traces.

Paired bootstrap intervals compare plans on identical exogenous failure traces
within a case. They quantify Monte Carlo variability conditional on that case
and simulator, not model uncertainty or all-workload generalization. The
no-backup ablation's tiny generation latency excludes its parent's search;
`parent_planning_s` is recorded in `plans.json` and should be charged if that
placement is used in deployment.

No neural policy is trained in this version. If joint choices show repeatable
headroom over cheap baselines, the next experiment can test whether learning
recovers those choices quickly on unseen workloads. If existing replicas make
extra backups unnecessary, that is an intended possible result.
