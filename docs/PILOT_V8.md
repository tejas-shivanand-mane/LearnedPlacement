# V8: structure-aware candidate ranking

V7's tuning slightly improved average top-eight completion time, did not
materially improve large-case ranking, and worsened direct selection. V8 tests
whether representing individual resources and placement relationships helps.
It does not assume a GNN is better or that longer training solves the problem.

## Controlled comparison

Train three encoders with seeds **11, 29, 47**, retaining every seed:

| Encoder | Inputs and computation |
|---|---|
| MLP | The existing 60 log-transformed summary features, one tanh hidden layer |
| Set | Machines, shards, replica groups and failure domains; shared node transformations and mean pooling by type |
| GNN | The same node attributes and pooling, plus two layers of typed message passing |

Machines connect to their failure domain and active replica group. Separate
relations connect machines to their primary or backup shards. Reverse relations
allow information to flow both ways (eight directed relation types). No machine,
shard, domain or candidate IDs enter model features. The graph uses membership
edges rather than a dense machine-pair adjacency matrix.

Both structured encoders receive the original global features plus the same
node attributes, including one-hop neighbor summaries by relation. The set
baseline can therefore see local placement attributes, but cannot propagate
messages through the explicit edges. This is a **DeepSets-style set encoder,
not a Set Transformer**. Both use a small MLP scoring head. The GNN can combine
relationships over multiple hops; improvement is a hypothesis, not a guarantee.
The representations describe the initial complete plan, not future failures.

Defaults: 64 hidden units, two set/GNN layers, Adam learning rate 0.0003,
entropy weight 0.03, gradient norm limit 5. Parameter counts are reported;
matching hidden width does **not** match parameter counts. The MLP-to-structured
comparison changes both information and architecture. The set-to-GNN comparison
more directly tests message passing with otherwise shared attributes.

All architectures use identical case/pool/trace seed rules and feasible
64-candidate pools. Actions are complete plans. REINFORCE uses a previous-return
per-case EMA baseline, not optimal labels or a solver. V8 makes one update after
16 on-policy episodes; the policy is fixed while collecting this batch. The
three V8 encoders share this procedure. Frozen V5/V7 models used different
training procedures and data, so comparisons with them are historical controls,
not architecture-only ablations. No policy receives held-out evaluation outcomes.

## Latency and bandwidth

The simulator already includes a uniform **per-message latency** and separate
bandwidth parameters. Its collective cost is:

```
communication = volume / effective_bandwidth + 2 * (participants - 1) * latency
```

Within-group shard communication and across-group same-shard synchronization
both contribute. Network transfers for checkpointing, peer backups and recovery
also add latency to endpoint service time. A cross-domain factor reduces
bandwidth; all messages use the same latency in this pilot. This is not a model
of individual links, switches, congestion, overlap of compute/collectives, or
measured NCCL behavior. Domain membership represents shared failures as well as
the bandwidth penalty used in this simplified simulator.

| Profile | Latency | Changed resource |
|---|---:|---|
| reference | 1 ms | None |
| slow_io | 1 ms | Remote storage bandwidth × 0.125 |
| slow_network | 1 ms | Cross-domain bandwidth factor × 0.5 |
| variable_repair | 1 ms | Lognormal repair duration |
| high_latency | 10 ms | Latency only |
| slow_network_high_latency | 10 ms | Latency and cross-domain bandwidth |

The 10 ms setting is a synthetic sensitivity case, not a measured datacenter
value. Hardware and deadline/cost anchors are unchanged across profiles.
Every encoder receives latency, NIC bandwidth and the cross-domain factor.
Candidate pools are shared across methods within each case; the existing pool
seed includes the profile, so pools need not be identical across profiles.

## Diagnostic before training

On **48 validation cases** (six sizes × four regimes × two adverse network
profiles), score all 64 candidates with 16 planning failure traces and record
healthy reference times. Nominate at most three plans: best penalized objective,
fastest observed time, and fastest planning-point-feasible plan if one exists.
Confirm these on **128 separate failure traces**. Identical nominations share
confirmation work. The diagnostic is resumable and parallel across cases.

Export all planning candidate summaries and the confirmation results. Planning
feasibility is a noisy point estimate, and confirmation covers only nominated
plans. Finding a passing plan is useful evidence. Finding none is **not** proof
that the pool or problem is infeasible. Healthy times are not rigorous lower
bounds: changing live replica counts can also change communication overhead.
Multiple screens are exploratory and not multiplicity corrected. No model is
updated using diagnostic outcomes. Test data is never loaded by this stage.

## Training and selection

Hardware seed **20261120**. Four training instances × six sizes × four regimes
× six profiles = **576 training cases**. Pools contain 6/12/24/32/64/96 machines;
reference active counts are 4/8/16/24/48/64. Work scales with reference count; this
is not strong scaling of one fixed job. Three failure domains remain fixed.

The fifth hardware instance supplies validation. A deterministic rotation picks
one profile per size/regime, yielding **24 checkpoint-validation cases** (four
per profile). All profiles of a hardware instance share a split. Homogeneous
instances can still coincide across seeds; this is not broad OOD evidence.

Evaluate checkpoints at **2,048 / 4,096 / 8,192 episodes**, with eight failure
rollouts per training episode. Total maximum training budget is **73,728 episodes
and 589,824 rollouts**, plus validation and diagnostic work. There are 27
architecture/seed/checkpoint evaluations. The richer encoders add computational
work; no full-run duration or GPU speedup has been measured.

Each checkpoint uses 16 planning traces for top-eight search and 64 independent
validation traces for selected plans. Direct selection uses no planning
simulation. Pick one checkpoint **per architecture and deployment mode**,
by the mean validation objective across all seeds. Retain every seed; do not
select the best seed. `selection_metric="completion_time"` is an explicit
alternative to choose before starting, with its cost/deadline tradeoffs.

The reward remains:

```
-(restricted_mean_time / deadline
  + 20 * max(mean_observed_cost / budget - 1, 0)
  + 20 * max(required_success - deadline_success, 0)
  + 100 * (1 - completed_fraction))
```

This is completion time under failures with soft cost/deadline penalties, not
pure completion-time minimization. Structural constraints are hard checked.
No stochastic deadline/cost or optimality guarantee is claimed.

## Execution and resume

Install `.[notebook,architecture]` into the notebook's Python environment. The
architecture extra adds PyTorch; older NumPy-only pilots do not require it.
Use a CUDA-capable PyTorch build compatible with the GPU to enable GPU training.
`DEVICE="auto"` chooses CUDA when PyTorch reports it available; `"cpu"` works
without a GPU. No separate graph-learning package is needed.

One process owns the active network and optimizer. CPU simulator workers return
losses for the sampled on-policy actions; they never initialize CUDA. Jobs run
sequentially with shared simulator workers, avoiding nine competing GPU
contexts. Candidate scores are computed in graph chunks (default eight plans).
A bounded CPU LRU cache (64 entries) retains deterministic candidate pools and
features. Eight pool versions per training case are reused; failure traces are
fresh every episode. This bounds memory instead of caching every graph pool.

PyTorch model/optimizer state is saved every 32 episodes, at checkpoints and on
handled interruptions. Only complete batches commit. An interrupted batch
replays deterministically on CPU under the tested setup. Forced kills can lose
unsaved work. CUDA scatter/reduction order can produce small nondeterministic
differences. Backend/worker/source/dependency checks prevent changing these
mid-run. Checkpoint validation saves its candidate scores and observations, so
interrupted validation resumes without retraining. Completed jobs are skipped.

There is **no mandatory timeout**. `MAX_MINUTES=None` is the default. An optional
positive limit pauses work; an interrupt may wait for active workers. Do not
remove a run lock until all old processes have stopped. Configuration, source
or dependency changes require a new output folder; old results stay intact.

## Fresh comparison and outputs

Freeze selected policies and, by default, the original and selected V7 MLPs
from `results/pilot_v7_models_w4`. This folder contains local model weights;
the compact analysis ZIP does not. Setting `LEGACY_SOURCE=None` explicitly
omits historical models. Each architecture's direct and top-eight methods
receive its separately selected checkpoints.

Fresh comparison seed **20261121** generates separate validation and test
instances. The notebook defaults to validation. Each split has **144 cases**,
32 planning traces per candidate, and 256 paired held-out evaluation traces.
With historical models, there are **34 methods**; without them, 22. Baselines
are full-pool search, random top-eight, analytical top-eight and the local-search
reference. All plans come from the common pool except the local-search baseline.
A ranker cannot select a better plan absent from its pool.

Evaluation uses CPU inference for every model, with four case workers by
default. This avoids GPU contention and gives the old/new models a common
inference backend. Planning time includes pool construction, graph/features,
inference and candidate simulation; cold model load and shared trace generation
are excluded. Run separate serial measurements for isolated latency claims.
GPU training does not imply a GPU inference benchmark.

Inspect execution time, cost, deadline success, censoring and combined screens
by size and profile, plus planning time. All new models train on all configured
sizes. These results do not establish extrapolation to unseen sizes/topologies;
that requires a separate distribution-shift experiment. Reserve the test split
until architecture/training decisions are frozen.

Use `notebooks/PlacementPilot.ipynb`; V7 is preserved as
`notebooks/PlacementPilotV7.ipynb`. Default output folders are
`pilot_v8_tuning_w4`, `pilot_v8_models_w4` and `pilot_v8_validation_w4` under
`results/`. Share the tuning and evaluation `.analysis.zip` files. The training
ZIP includes the diagnostic, learning curves, per-case checkpoint metrics and
selection. No weights, optimizer state, cached graphs or raw rollouts are exported.

Only tiny fixture tests, encoder forward/backward checks and notebook syntax
checks were run during implementation. No full research experiment was run.
