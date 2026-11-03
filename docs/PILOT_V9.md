# V9: frozen RL initialization for local search

## Question and comparison

Can existing RL policies obtain better plans with less search? V8's fixed-pool
ranking did not test this. No new models are trained, no hyperparameters are
selected on these results, and a positive result is not assumed.

| Family | Initialization | Refinement |
|---|---|---|
| Heuristic | Fast eligible machines, speed grouping, shard/domain spreading, default policies | Randomized first-improvement search |
| Random | Uniform choice from the 64-plan feasible pool | Same search |
| MLP / Set / GNN | Highest score in the same pool, using frozen V8 direct-selection checkpoint | Same search |
| Portfolio | Speed plan, domain-aware plan, checkpoint/protection tuning, coordinate search | Same randomized search after that initialization |

The first five families isolate initialization with the same feasible-neighbor
generator, acceptance objective, and search seed schedule. The RNG does not
include the model seed or method label; different starting plans still lead to
different neighborhoods and RNG consumption. Random initialization has its own
RNG so it does not consume search randomness. The portfolio is a separate strong
pipeline baseline, not an initialization-only ablation. Its initialization follows
V8, but its final phase continues to the requested budgets instead of stopping
at V8's original total candidate cap. Do not label it the identical V8 reference.

All available model seeds are retained (two if that is what the source contains).
Direct checkpoints were selected previously for direct plan quality, which matches
this initialization task. Top-eight checkpoints and historical V5/V7 policies
are not added. No checkpoint is selected using V9 outcomes. Source model hashes,
prior checkpoint selection records, source hashes, dependency versions and config
are saved; incompatible simulator/feature sources are rejected.

## Workloads and budgets

Default hardware seed 20261130 differs from training and V8 evaluation. Each
hardware instance has one test-designated split; models are already frozen.
This is a new exploratory held-out cohort, not a claim of an untouched final
paper test if these results subsequently inform method development.

Balanced pilot: one deterministically rotated profile for each of six reference
sizes and four regimes = 24 cases. Each of the six profiles occurs four times.
Available machine counts are 6, 12, 24, 32, 64, 96. `case_mode="full"` yields
144 cases, all profiles of every hardware instance. Profiles include slow IO,
low bandwidth, high latency, combined low bandwidth/high latency and variable
repair, as well as reference. Budget anchors remain unchanged in adverse profiles.
The balanced design is a cheap pilot; size and profile effects cannot be fully
separated within it. Homogeneous draws can coincide despite distinct seeds.

Two search seeds (101, 202), 32 planning traces, 256 evaluation traces, horizon
factor 32 (inherited from the actual source config, not an assumed stock config).
With two model seeds this is nine individual methods, 18 trajectories/case,
432 trajectories in the 24-case pilot. Do not treat model/search seeds or failure
traces as independent workload samples for significance testing.

Time checkpoints: 1, 2, 5, 10, 25 seconds. Unique candidate checkpoints:
8, 16, 32, 64. Budgets are measured from the same trajectory, including
initialization. A run continues until **both** largest budgets are satisfied,
or its proposal space is exhausted. Thus some large-case trajectories run longer
than 25 seconds to obtain 64 candidate evaluations. There is no overall 30-minute
limit. The default pilot entails roughly 3 worker-hours of search if every
trajectory takes about 25 seconds, or about 45 minutes on four workers before
evaluation and any longer trajectories. This is workload arithmetic, not a
measured runtime prediction. Full mode has six times as many cases.

## Timing and fairness

All methods use CPU inference and the same simulator, objective and planning
failure traces per case. Cold model loading and common trace generation are
outside the timer; fallback construction, pool generation, encoding, inference,
neighborhood construction and all planning simulations are inside it. Timing is
warm-model planning latency, not cold deployment startup. One Torch thread per
worker, four case workers by default; keep workers fixed, avoid concurrent jobs.
Method/search-seed order is reproducibly shuffled per case to reduce ordering
bias. CPU contention can still affect latency. No GPU training or inference is
performed; parallel CPU simulations dominate this experiment.

Planning caches are private to each trajectory. Repeated plans within a search
reuse that search's result, but no method receives another method's cached
simulations for free. Candidate count means unique plans fully scored on all
planning traces. It does not count feature extraction, neighbor generation,
or cached repeated proposals, so report count and time curves together.

Each completed candidate records its completion timestamp and the best plan
known then. A candidate completing at 5.1 seconds is excluded from the 5-second
snapshot even if it started earlier. Before any candidate finishes, the common
structurally feasible speed fallback is selected and flagged `unscored_fallback`.
If even fallback construction is unfinished, the plan is unavailable and matched
coverage is reported; no later plan is silently inserted. Candidate-count
snapshots include initialization scores and use the same trajectory, without
looking at later evaluation outcomes. Search termination before a budget carries
forward its last available plan.

The best planning objective is nonincreasing. The held-out evaluation objective
need not be: optimizing sampled failure traces can overfit those samples.
Time curves are not rerun at every budget. Some wall-clock fluctuation is normal.

## Evaluation and outputs

Only after all searches for a case complete are independent evaluation traces
generated, under a distinct seed namespace. No evaluation outcome enters search.
Selected plans shared across budgets or methods are evaluated once per case;
this safe evaluation-only cache reduces work without changing planning timing.
All selected plans are checked for structural feasibility.

The objective remains:

```
restricted_mean_time / deadline
+ 20 * max(mean_observed_cost / cost_budget - 1, 0)
+ 20 * max(required_success - deadline_success, 0)
+ 100 * (1 - completed_fraction)
```

Weights come from the trained source config. Completion uses the restricted mean,
so unfinished runs contribute the simulation horizon rather than disappearing.
Planning time is excluded from this objective. Export cost, deadline success,
completion fraction and statistical-screen rate alongside it.

The notebook plots evaluation objective and completion time against both budgets.
It averages model and search seeds within cases, then weights cases equally.
Only complete matched cases at a budget enter plots; coverage is printed. Use
profile/size filters to inspect tradeoffs; avoid selecting favorable subsets as
headline evidence. No error bars or bottom captions. The compact analysis ZIP
includes selected plans, summaries, search progress, provenance and hashes, but
not model weights or raw observations. Results remain synthetic, with no proof
of global optimality or real distributed-training improvement.

## Running and resuming

1. Pull `main`, restart the notebook kernel and open `PlacementPilot.ipynb`.
2. Point `SOURCE` to the existing `pilot_v8_tuning_w4` (or V8 models) directory.
   An analysis ZIP is insufficient. This never resumes V8 training or reselects
   its checkpoints, so a new source file in this commit does not invalidate
   existing frozen models.
3. Choose settings before starting; inspect the model/case count printed by the
   configuration cell. Run the experiment cell, then plotting/export cells.
4. Rerun unchanged to resume. Change output folder if config, workers, dependencies,
   model bundle or source changes. Earlier outputs are not overwritten.

Every completed trajectory is saved. An interrupted trajectory restarts from
scratch; replaying it with a warm restored cache would corrupt time comparisons.
Evaluation saves every `save_every_traces` traces and resumes cached work. A hard
kill can lose the current trajectory or a few evaluation traces. Parallel
interrupts cancel queued cases and signal workers to stop between simulator
traces. A currently running trace or neighborhood construction must finish;
confirm processes stopped before clearing a stale `.run.lock`. No simultaneous writers to one output are allowed.
Full research execution is left to the user; automated tests use tiny synthetic
networks and workloads, never the user's trained weights or full benchmark.
