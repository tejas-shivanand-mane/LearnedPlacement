# V7: representative training, checkpoint validation and modest tuning

The v5 policies were trained only on 6/12/24-machine pools. V6 demonstrated useful
ranking transfer to 32/64/96 machines, but inconsistent benefits from RL-started
local search. V7 tests whether representative training, duration and a small
hyperparameter comparison improve that ranking. It is still a contextual bandit
choosing complete feasible plans, not a runtime recovery controller.

## What runs

1. Train **nine policies**: three settings × seeds 11/29/47, from scratch on mixed
   sizes. Keep old v5 policies untouched. Actual NumPy planner training runs on CPU;
   the distributed workload and failures are simulated.
2. Validate saved policies at **2,048, 4,096, 8,192 and 16,384 episodes**. Validation
   never updates weights, optimizer state or reward baselines. Save recent-window
   training reward as well as cumulative reward; validation curves determine
   whether more training helps held-out decisions.
3. Select **one setting AND episode shared across all three seeds**, using the mean
   top-eight validation objective across seeds. Retain every seed. The final
   checkpoint is not automatically the best, and this modest sweep does not
   establish optimal tuning or convergence.
4. Freeze original and selected policies together, then compare them on fresh
   full validation cases. The test split is a separate explicit notebook setting.

| Setting | Hidden units | Learning rate | Entropy weight |
|---|---:|---:|---:|
| baseline | 32 | 0.001 | 0.01 |
| wider | 64 | 0.001 | 0.01 |
| conservative | 64 | 0.0003 | 0.03 |

The last setting changes learning rate and exploration together. This is a small
configuration comparison, not an ablation identifying their separate effects.
Each episode samples a context and a feasible action, then obtains eight failure
rollouts. All settings use identical context/trace seed rules; sampled actions
can differ. There are 64 candidate actions. Eight deterministic candidate pools per training
case are sampled across episodes, shared across settings/seeds; their plans and
feature matrices are cached in each worker. This avoids regenerating candidates
and features on every update. Failure traces remain fresh per episode. It changes
the candidate proposal protocol from v5's fresh pool per episode, so training
differences are not solely attributable to model size. Pools are regenerated
deterministically after restart; the cache stays in memory, not in exports.
No solver or optimal labels are used.

## Data separation and work budget

Training hardware seed: **20261108**. Four training instances for each of six
sizes and four regimes, with four profiles, produce **384 training cases**.
Available machine counts are **6, 12, 24, 32, 64, 96**; reference active counts
are **4, 8, 16, 24, 48, 64**. Reference work scales with this count; this is not
strong scaling of one identical fixed job. Shard choices 2/4/8 and replica choices
1/2/4/6/8/12/16 allow more than 50 active machines. Three failure domains remain
fixed and synthetic; this is not a calibrated datacenter workload.

The fifth hardware instance is validation. For inexpensive checkpoint checks,
select one profile per size/regime by a fixed rotating rule: **24 cases** balanced
across sizes and profiles. All profiles of a hardware instance share its split.
No test instances are generated during tuning. Homogeneous cases can be identical
across instances; new seeds alone do not prove broad distributional generalization.

At each checkpoint, validate both direct argmax and RL top eight. Top-eight
selection uses **16 planning traces**, and selected plans are evaluated with **64
separate validation traces**. Validation pools/traces stay fixed across settings,
seeds and checkpoints. Identical direct/reranked actions reuse evaluation work.
The default selects on the mean top-eight penalized objective; completion time,
cost, censoring and constraint screens are also exported and plotted. With only
64 validation traces, confidence-bound screens are coarse. They are diagnostics,
not a feasibility guarantee or a model-selection constraint.

Fresh full comparison hardware seed: **20261109**, with one validation and one
test instance per size/regime and all four profiles. Each split has **96 cases**.
It uses **32 planning traces and 256 paired held-out evaluation traces**, matching
previous full evaluations. The cheaper checkpoint selector is an approximation
of this deployment procedure; the fresh comparison checks that its choice holds
up. Keep test untouched until decisions are final. If you inspect test and then
tune again, reserve new final test instances/traces.

Default maximum training work is **147,456 episodes × 8 = 1,179,648 simulated
training rollouts**, plus checkpoint validation. This is a larger run than v5;
there is no measured full-run duration yet. Jobs run concurrently on CPU and
reuse saved work. No full research experiment was run by the agent.

## Objective

The v5 reward is unchanged so that changing the model/data is distinguishable
from changing the optimization task:

```
reward = -(restricted_mean_time / deadline
         + 20 * max(mean_observed_cost / budget - 1, 0)
         + 20 * max(required_success - deadline_success, 0)
         + 100 * (1 - completed_fraction))
```

Completion time under failures is reported prominently, but cost/deadline terms
can favor a slower plan. Structural constraints are hard checked; stochastic
budget/deadline requirements remain soft penalties. The optional configuration
`selection_metric="completion_time"` selects the checkpoint by time instead,
without changing training reward, but it may choose more constraint violations.
Choose this before starting a new run; default is `"objective"`.

## Comparison methods

- Original small-trained policies: model IDs 11, 29, 47.
- Selected mixed-size policies: IDs 1011, 1029, 1047 (the added 1000 identifies
  the family; actual training seeds remain 11, 29, 47).
- Direct and top-eight inference for all six models, on identical pools.
- Random top eight and full 64-candidate simulation search.
- **Analytical top eight**: a fixed nonlearned ranking using the simulator's
  healthy-step algebra, checkpoint/backup overhead, prices, hazard rates, spare
  counts and domain-based redundancy. No event loop or failure rollouts are used
  for ranking. Simulate its top eight on the same planning traces as RL top eight.
- The longer heuristic/local reference, using the existing v6 engine.

The analytical ranker is a transparent first-order approximation, not a published
system reproduction or rigorous expected-time estimator. It approximates async
checkpoint overhead and redundant recovery and must itself be assessed rather
than assumed optimal. All ranking time is included in measured planning time.
Equal candidate counts need not imply equal latency because plans can differ in
simulation cost. Report both. The original local-search budget sweeps remain in
the v6 notebook; this follow-up concentrates on ranking quality.

## Notebook, resuming and outputs

Pull main, restart the kernel, open `notebooks/PlacementPilot.ipynb`. It has
separate cells for training, read-only progress/plots, freezing the comparison
source, running fresh validation and exporting. `PlacementPilotV6.ipynb` preserves
the previous workflow. `SOURCE_RUN` points to the completed original v5 run
containing model checkpoints; an analysis ZIP cannot supply model weights.

`MAX_MINUTES=None` means **no mandatory timeout**. `WORKERS=4` uses separate CPU
processes, with one BLAS thread per process. GPU is not required: simulation
rather than the tiny MLP is the main work. Worker count is editable before the
run; keep it fixed on resume. No claimed GPU acceleration is present.

States save every **32 complete episodes**, at every validation checkpoint and on
handled interruptions. A forced process kill can lose up to 31 unsaved episodes;
replaying them is deterministic. Checkpoint weights, validation candidate scores
and validation observations are saved separately. Interrupted validation resumes
without retraining its checkpoint. Completed jobs are skipped on rerun. An
interrupt may wait for active workers; confirm processes stopped before removing
stale `.run.lock` files. Detailed job progress is in `jobs/<setting>-seed-<n>/`.
Config/source/NumPy changes require a new tuning output directory. Increasing the
maximum checkpoint is a configuration change, not an in-place extension.

Default folders:

- `results/pilot_v7_tuning_w4`: training, checkpoint validation and selection.
- `results/pilot_v7_models_w4`: immutable original/selected comparison bundle.
- `results/pilot_v7_validation_w4`: fresh full validation comparison.

Share both compact analysis ZIPs from the final export cell. The tuning ZIP
contains learning curves, case-level checkpoint metrics, source hashes and the
selection rule/results. The evaluation ZIP contains paired method comparisons,
selected plans, model hashes and timing metadata. Raw rollouts and model/optimizer
weights stay local. Source checks prevent mixing changed models into old results.
Timing under multiple workers includes CPU contention; use a separate serial run
for isolated latency claims. Interval/screen caveats from v6 still apply.
