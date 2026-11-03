# Pilot v5: learned plan selection with a longer simulation horizon

## What this experiment tests

Can a learned policy choose a good complete resource/placement/protection plan
without simulating candidate plans at deployment? Compare its quality, budget
and deadline behavior, and full decision latency against search and heuristics.

This version trains a real **one-step contextual-bandit reinforcement-learning
policy** using REINFORCE. A context is a workload and cluster; an action is one
structurally valid complete plan; the simulator supplies a stochastic reward.
A small neural network scores candidates. This is not a multi-step controller
that replans during failures, not optimal-plan imitation, and not full PPO/DQN.
The candidate generator limits what all candidate-ranking methods can choose.

The training workload itself remains synthetic: no tensors, GPUs, Ray, cloud
calls or Z3 are involved. Only the small planner networks undergo ML training,
using NumPy on the CPU. The v4 workload/recovery model and its limitations still
apply; see [PILOT_V4.md](PILOT_V4.md).

## Longer horizon, fresh experiment

`simulation_horizon_factor` is **32**, increased from 8. With the default
reference workload this is approximately 25,228 simulated seconds (7 hours),
instead of 6,307 seconds. It applies to every method and training/planning/test
rollout. Deadlines and cost budgets are unchanged. A longer horizon reduces
censoring but cannot guarantee that every workload finishes. It is separate from
the **30-minute real wall-clock cap per invocation**.

V5 uses fresh scenario and trace seeds, a new output directory `results/pilot_v5`,
and recomputes all comparisons. Do not mix v4/v5 statistics: the horizon,
objective, scenario split and some baselines differ. V4's notebook is preserved
as `PlacementPilotV4.ipynb` and its results remain readable.

## Shared objective

Let `T` be restricted mean execution time, `D` the deadline, `C` mean observed
cost, `B` budget, `p` the observed fraction finishing by the deadline, `q` the
fraction finishing by the horizon, and `p_required=0.95`. All failure-aware v5
selectors minimize the same finite penalized score:

```
T / D + 20 * max(C / B - 1, 0)
      + 20 * max(p_required - p, 0)
      + 100 * (1 - q)
```

RL receives its negative as reward. This replaces v4's lexicographic
constraint-violation ordering. Completion time remains a term even when no
sampled plan satisfies the constraints. The weights are explicit configuration,
not learned or tuned on test results. The penalties do **not guarantee**
stochastic budget/deadline constraints. Report actual screens and failure rates,
not reward alone. Structural assignment, memory and backup constraints are
checked before offering an action and again after selection.

RL estimates reward from 8 independent failure traces per episode; baseline
search uses 32 planning traces. Nonlinear penalties on small samples can bias
estimates and rare failures may be missed. This is a pilot, not a proof of
constraint satisfaction. Evaluation uses 256 new paired traces per case.

## Training and splits

Default matrix: 3 reference sizes × 4 regimes × 6 hardware-instance indices ×
4 environment profiles = 288 cases. Instance indices 0–3 are TRAIN (192 cases),
index 4 is VALIDATION (48), and index 5 is TEST (48). All profiles of one instance
stay together. Synthetic homogeneous instances can have identical hardware
attributes; these splits do not establish generalization to new architectures.

Each of seeds **11, 29 and 47** trains for **2,048 episodes**, with 8 rollouts per
episode: **49,152 training workload simulations** total. Episodes sample only
TRAIN cases, use fresh feasible candidate pools and fresh failure seeds, sample
an action from the actor, and update only after all its rollouts complete.
Training samples no validation/test contexts or outcomes. The terminal policy
at the fixed episode count is frozen; no checkpoint is selected using validation
or test results. Validation and test are both reported, separately. Later design
changes informed by test results require fresh final test instances.

The actor is a shared candidate scorer: a 32-unit tanh MLP over fixed,
dimensionless log-transformed features. Features include compute/cost estimates,
resource counts, domain collisions, topology, memory, protection, checkpoint and
recovery choices. No case IDs, split labels or future failure events are input
features. Normalization is fixed analytically, so it is not fitted on held-out
samples. Candidate index is not a feature.

The update is REINFORCE with entropy regularization, an action-independent
per-context exponential moving baseline from **previous** visits, Adam, and
explicit gradient-norm clipping. All of this is in `learning.py`. Three seeds
measure training randomness; no best seed is selected using held-out outcomes.
The running training reward is a diagnostic, not a held-out performance claim.

A supervised control has the same architecture and learns reward prediction
with Huber regression from exactly the first RL seed's sampled state/action
feedback. It gets no optimal labels or extra simulator calls. It selects the
highest predicted reward at deployment. This comparison does not establish the
best achievable supervised method; broader label generation is a later baseline.

## Methods and comparison fairness

| Method | Selection |
|---|---|
| `speed_fixed` | Original fixed-resource speed grouping; reference checkpoint cadence. |
| `domain_tuned` | Domain-aware placement with checkpoint/protection/recovery policy tuning. |
| `coordinate_search` | Single-edit search, now initialized from the tuned domain-aware plan. |
| `heuristic_portfolio` | Best of speed, tuned-domain and coordinate plans on planning traces only. |
| `local_search` | Bounded local search initialized from that portfolio winner. |
| `random_search` | Failure-aware simulation search over the complete common deployment candidate pool. |
| `random_unaware` | Healthy-only selection over that same complete pool. |
| `rl_seed_11/29/47` | Each independently trained actor scores the same common pool; deterministic argmax, no simulator calls. |
| `supervised_control` | Reward regressor scores that same pool; no simulator calls. |

The common pool has up to 64 valid plans: cheap speed/domain/price/risk seeds,
checkpoint cadence variants, then random proposals. Generation is deterministic
per scenario and does not use simulation outcomes. All three actors, supervised
control and both common-pool search selectors use **identical plans**, recorded
with a shared pool hash. Pool search is not a global optimum; it is a comparison
within the offered action set. Its stochastic planning estimates may still rank
plans differently from held-out evaluation.

Common-pool search completes all candidates and saves per-candidate progress;
it is not truncated at 20 seconds, which would change the action space seen by
RL. It still obeys the global 30-minute cap. Other search methods retain their
candidate limits and per-search 20-second cap. Their local search action spaces
are not identical to the deployment pool, so comparisons are also comparisons
of proposal strategies. A statistically feasible or optimal plan is not assured.

Decision times for learned methods include candidate generation, feature
extraction, scoring and structural validation. Models are loaded once before
evaluation; cold loading is excluded. Training wall time and simulation counts
are reported separately. Parent heuristic times are included in coordinate/local
search and portfolio costs; do not sum those cumulative times again.
`random_unaware` timing is attributed **healthy scoring only**, excluding pool
generation, so it is a lower-bound control timing rather than full latency.
Planning-trace generation is shared preparation outside per-planner timing.

## Resume and outputs

The notebook's Run/resume cell performs training first, then evaluation, pausing
after at most 30 real minutes plus final serialization. It may take multiple
invocations. Every completed episode saves actor/regressor weights, both Adam
states, moving baselines, episode count, time and history. Seeds are derived
from episode indices: an interrupted episode is replayed deterministically.
Incomplete episodes never contribute a partial update. Completed pool-search
candidates and evaluation observations also persist. Interrupted work contributes
to wall time; saved simulator counts describe completed episodes/candidates.

Sources, configuration and NumPy version are fingerprinted. Changed values require
a NEW output directory. Never remove `.run.lock` while an old process is active.
The full research training/evaluation is intended to be run by the user; repository
tests use only tiny synthetic learning tasks and two-step simulator fixtures.

Default evaluation: 96 validation/test cases × 11 methods × 256 traces =
**270,336 evaluations**, in addition to training and baseline planning. A smaller
first run can use sizes `[4,8]` and the first two profiles while preserving the
split layout. The notebook prints the full selected configuration.

Share `pilot_v5.analysis.zip`. It contains case inputs, selected plans, summaries,
paired statistics, model/pool hashes and compact learning histories/timings. Raw
observations, full candidate audits, optimizer states and weights stay local.
No full training data archive is required for routine analysis.

Evaluate TEST separately from VALIDATION and show each RL seed. Report restricted
means, censoring, budget/deadline screens and total decision time alongside
completion time. The longer horizon and penalties do not justify treating
unfinished trajectories as finished. The separate
`estimated_execution_plus_planning_s` column is a reporting estimate, not a
simulation of resource acquisition or failures during planning.
