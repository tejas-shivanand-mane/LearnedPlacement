# LearnedPlacement

A local simulator for resource, placement, checkpoint and recovery planning.
**Pilot V8 compares an MLP, a set encoder and a heterogeneous GNN**, using matched
REINFORCE training budgets and separately selected direct/top-eight checkpoints.
It also diagnoses candidate-pool feasibility under adverse networks and varies
latency independently of bandwidth. Distributed workloads remain simulated;
planner networks really train. No Ray, cloud or solver is required.

## Run locally

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[notebook,architecture]'
python -m jupyter lab notebooks/PlacementPilot.ipynb
```

Python 3.10+ is required. Restart the kernel after pulling. `DEVICE="auto"` uses
CUDA when available, otherwise CPU; use a compatible CUDA PyTorch build for GPU
training. One process owns the GPU and four CPU workers simulate sampled actions.
Evaluation uses CPU inference consistently across all models. No graph framework
or additional GPU is required. **No mandatory time limit** (`MAX_MINUTES=None`).

Run the network diagnostic cell, then the train/resume cell. Defaults: 3 encoders
× 3 seeds, 2,048/4,096/8,192-episode checkpoints and pools of 6–96 machines. The
six profiles include 1 ms versus 10 ms latency and separate bandwidth reductions.
Selection uses validation only; keep final test untouched. The reward is unchanged.

The comparison cell defaults to including original and tuned MLPs from
`results/pilot_v7_models_w4`. Set `LEGACY_SOURCE` to your actual V7 models folder;
an analysis ZIP has no weights. Explicitly set it to `None` to omit historical
policies. Share the two compact ZIPs from the export cell. See
[the V8 protocol](docs/PILOT_V8.md) for budgets, architecture details, latency
assumptions, source checks and limitations. Full research runs are left for you.

V7 is preserved in `notebooks/PlacementPilotV7.ipynb`; earlier V5/V6 notebooks
also remain available. Original results are preserved. Source/configuration or
dependency changes require a new output directory; rerun unchanged to resume.

## What v5 compares

Three independent **REINFORCE contextual-bandit policies** choose a complete
feasible plan from a common candidate set. They learn from simulator rewards,
not optimal labels. After training, selection needs no simulator calls. This is
a first RL comparison, not a dynamic controller that replans during failures.

Comparators include speed grouping, a tuned domain-aware heuristic, coordinate
search, a heuristic portfolio, stronger initialized local search, matched-pool
failure-aware/unaware search, and a supervised reward-prediction control. All
failure-aware selectors use a shared penalized completion-time objective.
Structural feasibility is checked; stochastic deadline/budget guarantees are not
claimed. Train/validation/test cases and failure seeds are separated. Policies
freeze before validation/test; report all three learning seeds.

See [the complete v5 experiment](docs/PILOT_V5.md) for exact rewards, sample counts,
resume semantics, limitations and timing definitions. The simulator is still an
uncalibrated fixed-work model, not measured HSDP execution or a reproduction of
published systems. It does not yet model tensor/pipeline parallelism or multi-job
scheduling.

Share **`pilot_v5.analysis.zip`**. The compact export contains selected plans,
paired summaries and learning diagnostics; raw observations and weights stay local.

## Previous experiments and checks

V4 remains available as `notebooks/PlacementPilotV4.ipynb`, `configs/pilot_v4.json`
and `python -m learnedplacement.expanded_runner`; see [v4 documentation](docs/PILOT_V4.md).
V3 remains in `notebooks/PlacementPilotV3.ipynb` and `python -m learnedplacement.runner`;
see [v3 documentation](docs/PILOT_V3.md) and [the original model](MODEL.md).
Old results remain readable; fingerprints prevent mixing source versions.

```bash
python -m unittest discover -s tests -v
```

Learning tests require the `learning` extra; V8 tests require `architecture`. Tests use tiny deterministic fixtures
and synthetic gradient checks; full research training and evaluation are left for
the user to run.
