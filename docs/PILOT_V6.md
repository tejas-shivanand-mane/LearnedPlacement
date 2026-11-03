# V6: frozen learning guides bounded search

This experiment asks whether an existing learned ranking can reduce planning
work while preserving completion time, cost and deadline success under failures.
It does not train new policies or run real distributed training. No Ray, GPU,
cloud resources, exact optimizer or solver is required.

## Run

Pull main, restart the notebook kernel and open `notebooks/PlacementPilot.ipynb`.
Set `SOURCE_RUN` to the completed **v5 run directory** containing
`learning/frozen_models.json` and `learning/seed-*.json`. The compact analysis ZIP
contains no model weights and cannot be used. Automatic discovery works only if
there is one distinct completed model set. Every configured seed is retained;
there is no selection by validation/test quality. Models are copied into the new
run, hash checked and never updated. Original v5 outputs remain untouched.

`STAGE="small"`, `SPLIT="validation"` is the first run. Then use `STAGE="scale"`
with a new output name. Reserve `SPLIT="test"` until design choices are frozen.
The default fresh seed is 20261107, different from v5's 20261106.

| Stage | Available machines | Reference active machines |
|---|---|---|
| small | 6, 12, 24 | 4, 8, 16 |
| scale | 32, 64, 96 | 24, 48, 64 |

The scale stage adds replica counts 6, 8, 12, 16 (shard choices remain 2, 4, 8),
so plans can use more than 50 active machines. Total active count must fit the
available pool. Fixed work per case is anchored to its reference active count;
this is a growing-workload experiment, not a strong-scaling benchmark for one
identical training job. Three failure domains remain fixed at all sizes; do not
interpret this as a calibrated datacenter topology. Larger sizes are an explicit
extrapolation test for policies trained on small instances.

There is **no mandatory wall-clock cutoff**. `MAX_MINUTES=None` runs until done.
A positive value is an optional per-invocation limit; values above 30 are allowed.
Rerun the same cell/configuration/output to resume. Completed candidate summaries
and evaluation observations persist. An interrupted candidate retries in full.
Source/config/model changes require a new output directory. Worker count also
must remain fixed within a run because it affects timing.

`WORKERS=4` is the notebook default (or fewer if fewer CPUs are available).
Per-case detailed progress is saved in `partitions/<case>/worker.log`.
Independent cases run in separate CPU processes; NumPy/BLAS uses one thread per
spawned worker. The event simulator is Python and does not use CUDA. Moving its
small neural scorer to a GPU would not accelerate the dominant event simulation
work; GPU support is deliberately not a dependency. For isolated latency claims,
repeat with `WORKERS=1` in a new output. Parallel contention affects wall-budget
search quality as well as measured latency. Cold model loading and shared trace
generation are outside per-method planning timing.

A notebook interrupt may wait for active process jobs to reach a checkpoint or
finish. Do not start another run in the same directory while workers remain
alive. After a forced kernel/process kill, check for remaining processes before
removing stale `.run.lock` files (root and affected partitions).

## Methods and budgets

Defaults: three policy seeds, 64 common candidates, top-k in {4,8}, local candidate
budget 32, local time budget 5 seconds. These are **comparison budgets**, not an
experiment runtime cap. Edit their lists before starting a new run. There are 21
methods, 48 cases per stage/split and 256 evaluation traces: 258,048 logical
method-trace observations. Identical selected plans share held-out evaluations
within each case, saving simulator calls without changing outcomes. Planning
scores are not shared between methods. No full experiment was run by the agent.

- `rl_<seed>`: frozen deterministic argmax, zero planning simulations.
- `rl_<seed>_top4/top8`: score the same pool using RL, then simulate only the
  highest-ranked 4/8 candidates using 32 planning traces each.
- `random_top4/top8`: uniform random ordering without replacement over that same
  pool; top-4 is a prefix of top-8. All cases have reproducible random subsets.
- `pool_search`: simulate the entire common pool, not a global optimality oracle.
- `rl_<seed>_local_c32/s5`: initialize from the RL argmax, then improve locally.
- `portfolio_local_c32/s5`: construct the heuristic portfolio and continue local
  search, all under the SAME TOTAL budget as the corresponding learned method.
- `local_reference`: an un-timed reference pipeline: speed baseline, up to 24
  domain-tuned proposals, up to 64 coordinate-search proposals, then up to 64
  local-search proposals. This preserves the v5 strategy, but uses the v6 engine
  with score caching, no unnecessary healthy simulations and deterministic new
  proposal seeds. It is not a byte-identical rerun of v5.

Candidate budgets count unique fully simulated candidates across the entire
initialization/search pipeline. Every candidate gets 32 identical planning
failure traces. Cached duplicate plans consume no new simulation budget. For a
small budget the heuristic pipeline may exhaust it during initialization, which
is part of the end-to-end cost being compared. Each time budget includes pool
construction, features, ranking, heuristic initialization, proposal generation,
scoring, saving and resume overhead. Finite operations can overshoot a time limit;
`budget_overrun_s` reports this. If no candidate finishes, a structurally valid
unscored fallback is explicitly marked. Attempted simulator calls (including
incomplete work) and completed calls are both reported.

Local moves vary checkpoint/recovery policy, backup/spare allocation, shard/group
counts, assignment swaps and machine substitutions. Improvements are accepted
immediately; random proposals are attempted after exhausting a neighborhood.
The same deterministic local RNG seed is used across budgets for each strategy.
Neither local search nor RL guarantees an optimum or probabilistic feasibility.

All failure-aware selectors minimize the same v5 objective:

```
restricted_mean_time / deadline
+ 20 * max(mean_observed_cost / budget - 1, 0)
+ 20 * max(required_deadline_success - observed_deadline_success, 0)
+ 100 * (1 - completed_fraction)
```

Structural constraints are hard checked; cost/deadline requirements are soft
penalties. A plan can have a better score while being slower. Search sees 32
planning traces, never the 256 held-out evaluation traces. Final evaluation uses
a separate namespace. All methods are paired on the same evaluation trace seeds.

## Analysis

Use execution time under failures, observed cost, deadline/cost screens,
censoring, planning time and execution-plus-planning time. Compare top-k RL with
random-k at equal simulations; compare learned versus heuristic local search at
equal total candidate and wall budgets. Full-pool and local-reference results
show a higher-work quality reference. Keep all three policy seeds visible.
Report each machine size and split separately. Dot/line plots contain no vertical
error bars or bottom captions. Complete-case filtering avoids partial-run bias.

`export(OUTPUT)` creates only the compact `.analysis.zip`: manifests, source/model
hashes, training summary provenance, selected plans, planning diagnostics, paired
summary statistics and execution environment. Weights, raw observations, detailed
candidate scores and source snapshots stay in the local run. Intervals are paired
normal approximations, not simultaneous bounds. Profiles from one hardware
instance are dependent; homogeneous instances may be identical despite new seeds.
A new random seed alone is not proof of broad generalization. Do not tune on test
results and then call the same tests an untouched final evaluation.
