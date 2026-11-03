# LearnedPlacement

A local CPU simulator for resource, placement, checkpoint and recovery planning.
**Pilot v6 reuses frozen v5 RL policies to guide search**, compares equal planning
budgets and adds 32/64/96-machine pools. No Ray, cloud, GPU or solver is required.
The workload is simulated; no new neural training occurs in v6.

## Run locally

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[notebook,learning]'
python -m jupyter lab notebooks/PlacementPilot.ipynb
```

Python 3.10+ is required. Restart the kernel after pulling. Set `SOURCE_RUN` to
your completed v5 results folder (the analysis ZIP has no weights). Start with
`STAGE="small"`, `SPLIT="validation"`, then use `STAGE="scale"`. New output names
keep previous results intact. There is **no mandatory 30-minute limit**:
`MAX_MINUTES=None` runs until completion. Four CPU case workers are the notebook
default; worker count is editable. Rerun unchanged to resume.

See [the v6 design and limitations](docs/PILOT_V6.md) for methods, budgets,
parallel timing, fresh splits and export details. The compact `.analysis.zip`
contains only analysis inputs; raw data and frozen models remain local.
`notebooks/PlacementPilotV5.ipynb` preserves the earlier training workflow. The
v5 runner also accepts `max_minutes=None` or a value above 30 when called in
Python; its archived notebook still explicitly requests 30 minutes by default.

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
