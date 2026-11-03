# LearnedPlacement

A CPU-only simulator for placement and protection of sharded training state.
**Pilot v3 adds checkpoint frequency as a planning decision**, alongside machine
placement and extra backup locations. It compares joint search with inexpensive
placements whose checkpoint intervals are also tuned.

No Ray, GPU, cloud account, Z3, actual model training or learned planner is used.
The purpose is to test whether this decision space offers meaningful headroom
before investing in a learned policy. It does not reproduce Oobleck or Bamboo.

## Run locally

From an existing checkout:

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[notebook]'
python -m jupyter lab notebooks/PlacementPilot.ipynb
```

For a fresh checkout, clone
`https://github.com/tejas-shivanand-mane/LearnedPlacement.git`, enter the directory,
and create `.venv` with `python3 -m venv .venv` first. Python 3.10+ is required.

Restart the notebook kernel after pulling and run the cells in order. The new
run name is **`pilot_v3`**, preserving older results. Settings are editable.
Only Run/resume starts the experiment; plots and exports read saved files.

The standard-library command-line equivalent is:

```bash
python -m learnedplacement.runner --config configs/pilot_v3.json --output results/pilot_v3 --max-minutes 30
python -m learnedplacement.runner --output results/pilot_v3 --status
```

Each invocation pauses at a **30-minute cooperative wall-clock limit**, followed
by a brief final save. The whole experiment may take multiple invocations.
Rerun the same cell/command to resume. Do not change config or core source while
resuming; a changed experiment requires a new output directory.

## What v3 tests

V2 found that speed grouping captured most placement gains, and additional
backups were not selected by its joint planner. V3 tests a specific missing
trade-off: more frequent durable checkpoints cost time when healthy but can
reduce rollback after state loss. It also adds a single-replica layout, where
another full replica is not already available for recovery.

| Setting | Default |
|---|---|
| Active machines / candidate pool | 4 / 6, 8 / 12, or 16 / 24 |
| Replica groups × shards | 1 × 4, 2 × 4, or 4 × 4 |
| Regimes | homogeneous, cheap/risky, fast/risky, correlated failures |
| Instances per size/regime/profile | 2 seeded instances |
| Durable-storage profiles | 2 GiB/s (`reference`), 0.25 GiB/s (`slow_storage`) |
| Total cases / methods | 48 / 8 |
| Held-out evaluations | 48 × 8 × 256 = 98,304 |
| Planning traces | 32, distinct from held-out traces |
| Workload | 240 global steps; 3 s unit-speed compute per healthy step |
| State / working memory | 16 GiB over 4 shard positions / 4 GiB per active machine |
| Extra backup storage / refresh | 8 GiB / every 20 committed steps |
| Checkpoint grid | Every 20, 40, 80, or 240 committed steps |
| Joint search | At most 32 layouts × 4 intervals = 128 candidate plans |
| Joint search wall-clock cap | 30 s per case, shared by its three selectors |

Interval 240 means **no periodic checkpoint before this 240-step job completes**.
Version 0 is initially durable under every policy. Final output persistence is
excluded equally for all methods. Extra backup refresh remains fixed at 20 steps;
this pilot does not tune it, parallelism degree, or runtime recovery actions.

The storage profiles change only the durable-bandwidth input; their derived
reference budgets change accordingly. Hardware, failure-start seeds, replacement
and reinitialization costs are otherwise paired across profiles. The two
profiles are not independent replications. V3 uses a fresh seed (20261104),
rather than reusing the held-out v2 traces for another comparison.

Failure assumptions remain 0.12/hour for ordinary machines, 2/hour for risky
machines, and 0.2/hour plus 1 event/domain/hour for the risky machines in the
correlated regime. Replacement is 30 s and reinitialization 4 s. These are
**synthetic stress assumptions, not measured platform parameters**. The storage
range is an exploratory 8× I/O sensitivity; it is not evidence of any particular
storage service's speed. No failure rates were increased to make protection win.

## Methods and controls

| Method | Decision rule |
|---|---|
| `speed_grouped` | Fastest machines grouped by speed; checkpoint every 80 steps; compare plain/protected plans. |
| `speed_grouped_tuned` | Same speed-based placement; compare plain/protected plans at every checkpoint interval (up to 8 candidates). |
| `domain_packed_tuned` | Reliable-machine, domain-packed placement; compare plain/protected plans at every interval (up to 8 candidates). |
| `checkpoint_joint` | Choose placement, protection and checkpoint interval using failure planning traces. |
| `checkpoint_unaware` | Choose from the exact same candidate plans, scoring without failures. |
| `checkpoint_fixed_search` | Choose from the same layouts using failure traces, restricted to reference interval 80. |
| `checkpoint_fixed_ablation` | Keep the joint plan's exact placement/backups; reset only its interval to 80. |
| `checkpoint_no_backup` | Keep the joint plan's exact placement/interval; remove only its extra backups. |

All choices are frozen before held-out evaluation. The simple baselines may
reject backups; they are not forced to incur unnecessary protection overhead.

Joint-search layout generation receives zero hazards and sees no scoring
outcomes. Seeds cover speed-grouped, domain-packed, price/speed and speed-spread
placements with and without heuristic backups. Random valid swaps, replacements
and backup edits expand the pool. This is bounded shared-pool search, not an
adaptive optimizer, exhaustive search or proof of optimality.

For each layout, every configured interval must finish both healthy and failure
scoring before any selector can use the layout. If a local timeout interrupts a
block, the whole block is discarded for all three selectors. The aware and
unaware selectors therefore see identical candidate plans; the fixed-interval
selector sees the same layouts but only their interval-80 versions. Pool hashes,
completed counts and full candidate score audits are saved locally. With zero
completed layouts, a valid speed-grouped reference plan is returned with unknown
planning feasibility, not claimed as an optimized result.

Candidate effort differs by control: the fixed selector has one eligible plan
per layout, versus four for the tunable selectors. Failure-aware scores use
32 traces; healthy scores need one deterministic simulation. `planning_s` is
attributed scoring time plus shared bookkeeping/generation. Scores for the fixed
selector are reused from the joint evaluation; it is **not an independently
timed deployment benchmark**. `paired_search_wall_s` records actual shared wall
time. Never sum attributed times as elapsed time. Ablations inherit their
parent's search and are excluded from latency plots.

## Objective and feasibility

The objective remains **total completion time including failures**, including
training, checkpoint/backup I/O, detection, waiting, restoration, slower progress
with fewer replicas, and repeated work after rollback. Memory/assignment/backup
storage violations are rejected before evaluation.

Candidates are ranked by noncompletion fraction, then normalized deadline/cost
constraint violations, then restricted mean time, then observed cost. This is
not equal-weight time/cost minimization, and the search need not find a feasible
plan. The same objective is used by cheap tuned baselines.

Budgets use a fixed unit-speed, $1/hour-per-active-machine reference with the
case's **reference interval 80**, not the chosen plan's interval. Deadline is
1.75× reference time, cost budget is 1.5× reference cost, and simulation horizon
is 8× reference time. The reference includes checkpoint overhead at the profile's
storage speed. Thus choosing a different interval cannot relax its own budget.
These slack factors avoid v2's zero-overhead cost allowance, but do not prove
feasibility of every case.

Final screens use 256 new traces: no censoring, a one-sided 95% Wilson lower
bound on deadline success at least 0.95, and an approximate one-sided 95% normal
upper bound on mean cost within budget. They are separate per-plan screens, not
simultaneous or real-world guarantees. Planning uses point estimates instead.
Monetary cost is allocated capacity × elapsed time, including downtime and
backup-only machines; storage and egress fees are excluded.

## How to interpret the run

Start with joint search versus **both cheap tuned baselines**. A win over a
fixed-80 baseline alone may simply mean its checkpoint interval was poor.
Next compare joint versus fixed search to hold layout exploration constant,
and joint versus unaware to isolate scoring under failures. The two exact-plan
ablations isolate changing checkpoint cadence and removing extra backups.

Each plan has a failure-free observation using its own checkpoint interval and
backup policy. Total time difference is decomposed into healthy-time difference
and failure-penalty difference. A longer interval can reduce normal overhead
without improving recovery; the diagnostic prevents conflating those effects.
Compare chosen intervals, lost steps, phase times, costs and feasibility too.

Censored runs have null mean completion/total cost; their restricted time and
observed costs remain available. Paired bootstrap intervals in the notebook
reflect Monte Carlo variability within the synthetic case, not model uncertainty
or population-wide conclusions. No held-out outcomes are used to choose plans.

## Compact sharing and saved data

The final notebook cell calls `export_run(OUTPUT)` and writes
`results/pilot_v3.analysis.zip`. It includes config/source hashes, case inputs,
selected plans (including checkpoint intervals), summaries, healthy observations,
and paired statistics for all available method pairs. It omits raw observations,
full search audits, images and source copies, which remain on your machine.

The compact paired mean intervals use a normal approximation, not the notebook
bootstrap. Partial runs compare the shared saved trace prefix and label it as
partial. Censored pairs retain all observations and use restricted quantities.
Compact bundles support routine analysis; they are not resumable archives or
sufficient for arbitrary raw-distribution reanalysis. Use
`export_run(OUTPUT, full=True)` only for full troubleshooting data.

The output directory retains a manifest, source snapshot, status, case inputs,
plans, per-trace observations, healthy results, summaries and optional plots.
Observations save every 8 traces and on clean interruption/time pause. A globally
interrupted planner restarts on resume. After a killed kernel, remove a stale
`.run.lock` only after confirming that the old process has stopped.

Existing v1/v2 runs can still be read and exported with the updated exporter.
To export one, set `RUN_NAME` to its existing directory and run only setup,
settings and export. Do not invoke Run/resume against old source hashes.
Historical configs remain in `configs/pilot.json` and `configs/pilot_v2.json`;
use new directories for any fresh reruns with current code.

## Correctness and scope

```bash
python -m unittest discover -s tests -v
```

Tests use small deterministic fixtures, not the full experiment. They cover
checkpoint overhead and state-loss savings, version consistency, inherited
intervals, matched layout blocks, interrupted search, resume and compact export.

See [MODEL.md](MODEL.md) for the simulator contract. This remains a fixed-work
HSDP-style surrogate with simplified collectives and blocking recovery. It
cannot establish real-system speedups, convergence correctness, novelty, or
superiority to existing systems. It adds one meaningful decision dimension;
it does not implement pipeline-template reconfiguration or redundant computation
in pipeline bubbles. Learning remains a later experiment if useful headroom
persists against the stronger baselines.
