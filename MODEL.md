# Simulator contract and limitations

This is a planning surrogate for a **single fixed-work HSDP-style job**. It is
not a PyTorch/torchft implementation or a calibrated performance predictor.

## State and time

An optimizer commit advances a scalar version v by one. Each live replica group
contains S distinct shards with identical logical training state to every other
live group at that version. The configured state includes model, optimizer and
necessary recovery metadata; `working_gib` is separate transient memory.

The logical model assumes a fixed deterministic sample sequence and equivalent
state transitions across changes in group count. This abstracts away floating
point reduction-order effects. Matching step numbers in a real implementation
would be insufficient without compatible lineage, optimizer and RNG state.

For live group set L, group compute speed is the minimum machine speed in the
group. Compute time is `base_step_s * configured_replica_count / sum(group_speeds)`.
This assumes ideal workload redistribution between groups and a fixed global
sample count per step. It models the cost of fewer groups, not reduced training
work or dropped samples. Arbitrary microbatch divisibility is not modeled.

Communication is a serialized surrogate:
`collective_factor * shard_gib / effective_bandwidth`.
Effective bandwidth is the minimum live-machine bandwidth, reduced by
`cross_domain_factor` if live machines span multiple domains. This is deliberately
simple: it does not model real per-layer collectives, topology congestion,
overlap, pipeline bubbles, or precise HSDP scaling. A later real-system
calibration must replace this model before interpreting absolute times.

## Failure streams

Each node has an independent Poisson outage-start process. Each failure domain
has a separate shared Poisson process that starts an outage on every member.
Individual and domain outages may overlap; a node is available only after all
its outstanding outages end. Outage durations are fixed replacement delays.
Events are generated for the entire case horizon, not conditioned on a plan.
Events for unused machines do not incur time or cost for a plan.

Failures invalidate all volatile state on that machine. Replacement brings up
an empty machine in the same logical slot with the same properties. This pilot
does not model repair that preserves memory, acquiring different replacement
types, provider capacity queues, or migration to alternative slots at runtime.

## Atomic operations

Training steps, backup refresh, durable checkpoint writes and group restores
commit only on successful completion. Failure of a participating source or
destination aborts the operation; spent time remains charged. Failure of an
unrelated node does not abort it. Detection pauses are charged after an ongoing
operation completes or aborts. Simultaneous events are processed with down
before up; overlaps are reference-counted.

Training state stays at version v during the step; at commit all participating
groups advance together. This models the ability to discard an incomplete step.
A real system would need a correct step protocol to provide this behavior.

Backups retain their previous snapshot until a complete refresh commits. Their
host memory accounts for two shard buffers, although the storage budget counts
one retained logical snapshot. Extra snapshots refresh synchronously at a fixed
cadence. Unavailable backup hosts are skipped at a refresh; they catch up at a
later refresh. Primary state, existing replicas, and extra backups are all
eligible sources with appropriate versions.

Durable storage holds one complete committed version, initialized to zero.
Checkpoint writes use atomic replace and an implicit staging object. This tier
is assumed available and outside the additional in-memory backup budget.
Its size, fees, failures and metadata service are not optimized in the pilot.

## Recovery

At every decision boundary the simulator first checks whether every shard of
the current committed version has an available source anywhere. Sources can
include leftover shards from different damaged groups. If not, it finds the
**newest common version** present for every shard, counting the durable
checkpoint. It rolls back to that version, charges lost committed steps, and
invalidates snapshots from the abandoned future. It never stitches mismatched
versions to claim recovery.

If no complete physical group is available, the job waits for replacement
events. Any physically available but stale group is restored eagerly; this
blocks training and charges a fixed reinitialization cost plus data transfer.
Restored groups join only at a commit boundary. Groups not yet restored keep
their stale local shards and do not participate in updates.

Transfer requests use the same version for every source. A greedy source choice
accounts for accumulated source/destination load. Each endpoint's transfer time
is summed; independent endpoints overlap. Cross-domain traffic costs more.
Durable storage is one shared endpoint. No tensor traffic or real connections
are created. This is a shared-endpoint bandwidth approximation, not a network
simulator. Restoring several groups in a batch is atomic in this model.

Detection, rejoin, refresh and recovery-mode choices are fixed for all planners.
The planner only chooses the initial machines, group layout, shard positions and
additional backup hosts. It cannot see the sampled future failures.

## Statistical contract

Every planner gets the same planning trace set. Those traces are not reused for
reporting. Every frozen plan is evaluated on the same separately seeded test
traces, including unused-node events. Confidence screens apply to each frozen
plan under this model, not to a policy trained over the test set.

If the simulation horizon is reached before the job completes, that observation
is right-censored. It fails the deadline criterion (the horizon is beyond the
deadline). The reported cost is only cost accumulated by the horizon. Expected
completion and expected total cost are not reported as finite estimates if
any evaluated trace is censored. The planner uses a restricted mean as a search
proxy and explicitly prioritizes lower noncompletion fractions.

The default risk rates and performance numbers are synthetic. A useful pilot
outcome is whether different planners behave sensibly and whether there is
headroom worth investigating. It is not evidence of real-world speedups,
guaranteed deadline reliability, or novelty over existing systems.
