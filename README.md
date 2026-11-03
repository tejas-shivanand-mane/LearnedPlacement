# LearnedPlacement

A local CPU simulator for resource, placement, checkpoint and recovery planning.
**Pilot v7 trains RL policies across small and large configurations**, validates
saved checkpoints, compares three training settings and evaluates selected models
against the original small-trained models and nonlearned baselines. Workloads are
simulated; planner networks really train. No Ray, cloud, GPU or solver is required.

## Run locally

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[notebook,learning]'
python -m jupyter lab notebooks/PlacementPilot.ipynb
```

Python 3.10+ is required. Restart the kernel after pulling. The notebook has
separate train/resume and evaluate/resume cells, four configurable CPU workers,
and **no mandatory timeout** (`MAX_MINUTES=None`). Defaults: 3 settings × 3 seeds,
checkpoints at 2,048/4,096/8,192/16,384 episodes and pools of 6/12/24/32/64/96
machines. Validation chooses one setting/episode shared by all seeds. Test data
is not used during tuning. The original reward is unchanged.

After training, set `SOURCE_RUN` to your completed original v5 results folder
(the analysis ZIP has no weights). Run the fresh validation comparison with both
model families, random/analytical top-eight baselines and full/local search.
Share the two compact ZIPs from the export cell. See [the v7 design](docs/PILOT_V7.md)
for exact budgets, source checks, selection rules and limitations.

The earlier workflows remain in `notebooks/PlacementPilotV5.ipynb` and
`notebooks/PlacementPilotV6.ipynb`. Original outputs are preserved. Source or
configuration changes require a new run directory; rerun unchanged to resume.

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
