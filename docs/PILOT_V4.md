# Pilot v4: broader planning before reinforcement learning

This experiment asks whether joint resource, placement, checkpoint and recovery
choices improve **completion time for the same amount of training work**, subject
to memory, backup-storage, monetary-budget and deadline-success requirements.
It runs locally on a CPU. There are no tensors, GPU workers, Ray processes, cloud
requests, Z3 calls or learned policies. The simulator remains an uncalibrated
analytical surrogate; this is a headroom experiment, not a systems evaluation.

## Decisions actually implemented

| Decision | Default choices |
|---|---|
| Shards per replica group | 2, 4, 8 |
| Replica groups | 1, 2, 4, subject to available machines |
| Machine selection | Any feasible subset of the common candidate pool |
| Group membership and shard positions | Swaps, replacements and structured layouts |
| Extra state copies | Greedy allocations targeting 0, 1 or 2 copies per shard; actual counts limited by memory and 32 GiB backup budget |
| Reserved spare machines | 0, 1 or 2 if available and feasible |
| Backup refresh | Every 10, 20 or 80 committed steps |
| Checkpoint interval | Every 20, 40, 80 or 240 committed steps |
| Checkpoint destination | Durable remote store or node-local disk |
| Checkpoint execution | Synchronous or asynchronous |
| Checkpoint history | 1 or 2 retained snapshots, plus durable initial version 0 |
| Background bandwidth reservation | 25% or 50% during asynchronous saves |
| Recovery admission | Eager, or defer while at least one group can still train |
| Repair decision boundary | Every 1 or 20 committed steps, or immediately if no group can train |
| Recovery order | Fastest recoverable group first, or fewest missing shards first |
| Source selection | Greedy endpoint-load balancing, or first available compatible source |

The number of active machines is now a decision. Reference counts 4/8/16 only
set workload size, candidate-pool size (6/12/24), and planner-independent budgets.
The planner may allocate fewer or more active machines from that pool. Machines
have different speed, price, memory, bandwidth, failure rate and domain inputs.
The risk/price regimes are synthetic; there is no live cloud-price lookup or
claim that a particular node is an actual spot/on-demand product.

A protection request is a target, not a promise of equal copies for all shards.
The greedy allocator may protect only some shards when space runs out. Actual
(shard, host) copies, memory consumption and spare counts are saved for analysis.

## Conditions and experiment matrix

There are three reference sizes, four regimes (homogeneous, cheap/risky,
fast/risky, correlated), three hardware-instance seeds, and four profiles:

1. `reference`: heterogeneous RAM/NICs; fixed repair duration.
2. `slow_io`: remote storage bandwidth is reduced eightfold.
3. `slow_network`: cross-domain bandwidth is halved.
4. `variable_repair`: repair duration is lognormal, with coefficient of variation
   1 and the same mean as the fixed-duration case.

Profiles change one condition at a time. Node capabilities, Poisson failure
start streams and budget anchors are shared across a hardware instance's
profiles. These profiles are **paired sensitivities, not independent samples**.
Node/domain failure rates remain the previous synthetic regime values.

Default: **144 cases × 6 methods × 256 held-out traces = 221,184 evaluations**,
plus planning simulations. This is the full matrix, not a promised 30-minute
sweep. Each invocation is capped at 30 minutes and resumes from saved work.
For an initial subset, before starting a NEW run, set `active_nodes=[4,8]` and
`expanded_profiles=expanded_profiles[:2]` (48 cases). Keep the three instance
splits if preparing data for later learning.

Instance 0 is tagged `train`, instance 1 `validation`, and instance 2 `test`.
All profiles of an instance remain in the same split. This pilot trains nothing;
these tags establish the interface for the next phase. Three instance seeds are
not enough for a final generalization claim. If test results inform design,
reserve fresh final-test instances/seeds before the paper evaluation.

## Baselines and what their comparisons mean

| Method | Behavior |
|---|---|
| `speed_fixed` | Speed-grouped reference resource count, fixed checkpoint interval; no extra copies/spares. Diagnostic reference. |
| `domain_tuned` | Same resource dimensions; greedily rearrange shard positions across domains without changing group membership. Tune checkpoint/protection/recovery policies over up to 24 valid candidates. |
| `coordinate_search` | Start at the domain-aware reference. Try feasible single changes in a fixed order and accept improvements; up to 64 scored candidates. |
| `random_search` | Explore the full expanded space, including the same domain-aware seed; choose using failure planning traces. Up to 64 candidates. |
| `local_search` | Start at the same domain-aware seed, shuffle feasible single-edit neighborhoods, accept improvements and try random restarts when exhausted. Up to 64 candidates. |
| `random_unaware` | Select using healthy execution only from the **exact completed candidate pool** scored by `random_search`. |

These are in-repository baselines, **not reimplementations of Oobleck, Bamboo,
Parcae, Gemini, TierCheck or RLFTWS**. No planner is called optimal. Local search
can miss coordinated improvements and the finite candidate grids are not the
entire feasible space. Backup host allocation is heuristic, not exhaustive.

Every ordinary planner has a 20-second per-case cap in addition to its candidate
limit. Global interruption discards an unfinished planner and retries it on
resume. A local timeout keeps its best fully scored candidate; zero candidates
means a feasible reference fallback with unknown statistical feasibility.
Incomplete candidate scores never enter either matched selector. The full
candidate audit is saved locally, but omitted from compact analysis exports.

Planner timings measure search wall time, including generation, validation and
healthy/failure scoring. `random_unaware` has **attributed** healthy-scoring and
proposal cost from the shared search, not an independent deployment timing.
Candidate counts, timing kind, and pool hashes must accompany comparisons.
Timed caps can produce different candidate counts on different computers.

## Model semantics

- **Fixed work:** the reference per-step compute work is proportional to
  `base_step_s × reference_shards × reference_replicas`. A candidate's time is
  this work divided by its shards per group and the summed slowest-member speeds
  of its live groups. Changing resource count or sharding cannot reduce the
  amount of work. All methods commit the same 240 global optimizer steps.
- **Communication:** serialized computation plus within-group shard collective
  time plus between-group same-shard synchronization. Bottleneck NICs, latency
  and domain penalties are charged. This is not a packet/fabric simulator;
  there is no tensor/pipeline overlap model. Within-group shard permutation can
  now change inter-group communication costs.
- **Memory:** full model state is divided equally among shard positions. Each
  primary and spare reserves its shard plus fixed working memory. Each backup
  reserves two shard buffers. Asynchronous checkpoint sources reserve another
  frozen shard buffer. Checkpoint sources are primaries, not reusable backup
  buffers. Local history has a separate disk-capacity check.
- **Checkpoints:** full snapshots only; no incremental/compressed format yet.
  Version 0 is initially available in remote storage for every plan. Remote
  saves are globally durable on atomic completion. Local saves place one copy
  of each shard on a selected current primary's disk. Local disks and memory
  are lost on node loss; remote storage is assumed available.
- **Asynchronous saves:** one globally consistent snapshot may be in flight.
  It commits only after all transfers finish; a participating node failure
  cancels it. Frozen staging prevents later steps overwriting that snapshot.
  Save duration is divided by reserved bandwidth share. Foreground transfers
  and communication are conservatively slowed by the remaining share for any
  operation that starts while a save is pending, even if it finishes mid-operation.
  This is a deliberate conservative approximation, not measured overlap.
  Due save opportunities are skipped while another save is pending; aborted
  saves retry at a subsequent interval. Final output/checkpoint drain is excluded
  equally, so a pending final save need not complete before the job finishes.
- **Recovery:** sources must share one committed global version. Recovery can
  combine surviving primaries, backups, local checkpoints and remote snapshots.
  It never stitches incompatible versions. Detection, transfer and reinit time
  are charged. Group restoration blocks training while underway; this is not
  FT-HSDP's asynchronous catch-up. Defer lets healthy groups continue and only
  restores when all groups stop. Eager can restore at configured step boundaries.
- **Spares:** reserved idle machines can replace currently down primary slots.
  The fastest available compatible spare is chosen deterministically, then state
  transfer and reinit are charged. A spare is consumed once; retired original
  machines do not silently re-enter training. Original allocation plus spares
  stays billed for the full job, including outages. No free elastic resizing.
- **Failures:** Poisson node starts plus correlated domain starts, independent
  of the chosen plan. Overlapping outages are reference-counted. Domain members
  share repair duration. Identical traces are used across methods; runtime
  policy sees only events that have already occurred, never future downtime.
- **Cost:** reserved machine price × elapsed time, provisioned remote checkpoint
  capacity × elapsed time, and attempted remote write payload. Full attempted
  payload is conservatively billed even on interruption. These synthetic prices
  are editable and are not claims about any storage provider. Local disk is
  included in machine price. Network egress pricing is not modeled.
- **History:** retained snapshots are pruned to avoid using a future execution
  history after rollback. Consistency is version-based; data ordering/RNG replay
  and convergence are assumed, not executed or validated by this simulator.

## Objective, constraints and statistics

Structural memory, assignment and backup-capacity constraints are enforced
before simulation. Search ranks candidates lexicographically by censored
fraction, normalized cost/deadline violation, restricted mean execution time,
then cost. This is a feasibility-first heuristic, not an RL reward or proof of
stochastic feasibility. Deadline-success target is 0.95; budgets are anchored
independently of chosen plans and the search grids.

Planning uses 16 failure traces by default. Evaluation uses 256 different
traces, identical across methods. Plans are frozen before evaluation. Selection
on a small planning sample can overfit; increase that sample on a NEW run to
check stability before interpreting small gains.

Saved metrics include time, cost components, censoring, deadline success,
Wilson lower confidence bound, approximate mean-cost upper bound, phase times,
lost steps, rollbacks, checkpoint commits/aborts, spare remaps, planning time and
all chosen variables. The statistical screens are Monte Carlo evidence, not
formal reliability guarantees. Paired normal intervals in exports represent
within-case simulation uncertainty, not model error or simultaneous coverage.

`estimated_execution_plus_planning_s` adds measured planning latency to the
restricted execution mean. It is a reporting estimate: resource acquisition and
failures during planning are not simulated; deadline screening uses execution
time. For a censored run, restricted time and observed cost must not be reported
as expected total completion time or expected total cost.

## Scope staged for later work

This version broadens the HSDP-style model; it does not pretend to model every
possible training decision. Tensor/pipeline/expert parallelism, uneven layer
partitioning, microbatch/accumulation changes, incremental compression, coding,
activation recomputation, elastic acquisition/release, arbitrary migration,
nonblocking catch-up, and multi-job admission/fairness need additional semantics
and validation. Each is a documented next extension, not a cosmetic config flag.
Machine risk prediction uncertainty and non-Poisson arrival processes also need
separate sensitivity experiments.

`expanded_planners.observation()` exposes deployment inputs without future
traces, split labels or case identifiers. `valid_candidates()` exposes a
reproducible feasible candidate-action interface. These are preparation for
later supervised/RL policies, **not an RL agent or training run**. Future methods
must use the same simulator, resource budgets and held-out evaluation protocol;
report training costs separately and retain hard feasibility checks.
