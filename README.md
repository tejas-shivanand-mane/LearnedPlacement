# LearnedPlacement

A local CPU simulator for resource, placement, checkpoint and recovery planning.
**Pilot v5 trains and compares RL plan selectors**, with a simulation horizon
increased from 8× to **32×** reference time. No Ray, GPU, cloud or Z3 is required.
The planner networks train; the distributed training workload remains simulated.

## Run locally

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[notebook,learning]'
python -m jupyter lab notebooks/PlacementPilot.ipynb
```

Python 3.10+ is required. For a fresh checkout, create `.venv` with
`python3 -m venv .venv` first. Restart the notebook kernel after pulling.
The new output is **`results/pilot_v5`**. Only Run/resume starts work.

```bash
python -m learnedplacement.learning_runner --config configs/pilot_v5.json --output results/pilot_v5 --max-minutes 30
python -m learnedplacement.learning_runner --output results/pilot_v5 --status
```

Training and evaluation pause after 30 real minutes per invocation, with saved
weights, optimizer states and simulation progress. Rerun unchanged to resume.
The complete experiment may require several invocations. Config/source/NumPy
changes require a new run directory. The notebook provides a smaller first-run
option. Deadlines and budgets are unchanged despite the longer horizon.

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

Learning tests require the `learning` extra. Tests use tiny deterministic fixtures
and synthetic gradient checks; full research training and evaluation are left for
the user to run.
