# LearnedPlacement

A local simulator for resource, placement, checkpoint and recovery planning.
**Pilot V9 reuses existing V8 RL models to initialize local search; no training.**
Compare heuristic, random, MLP, Set and GNN starts with the same search procedure,
plus a stronger heuristic portfolio. Record quality at matched time and candidate
budgets, using independent evaluation failure traces.

## Run locally

```bash
git pull --ff-only
python -m pip install -e '.[notebook,architecture]'
python -m jupyter lab notebooks/PlacementPilot.ipynb
```

Restart the kernel after pulling. Set `SOURCE` to your completed V8 tuning or
models folder (default `results/pilot_v8_tuning_w4`); analysis ZIPs have no weights.
The settings cell checks the models before running. All available direct-policy
seeds are retained. No Ray, cloud, Z3, retraining or GPU is required.

Default: 24 balanced fresh cases, all six sizes/profiles, two search seeds,
four CPU workers, 32 planning traces, 256 independent evaluation traces.
Budgets: 1/2/5/10/25 seconds and 8/16/32/64 candidate evaluations.
There is no mandatory overall time limit. A trajectory can continue beyond
25 seconds to supply count checkpoints; later plans cannot affect earlier results.
Choose `case_mode="full"` before starting for 144 cases. Keep settings and workers
unchanged to resume; choose a new output folder for a changed experiment.

Run the experiment cell, then the plotting and compact export cells. Share the
single `.analysis.zip`. See [the V9 protocol](docs/PILOT_V9.md) for timing,
model provenance, resume details and limitations. Full research runs are left
for you. Prior outputs are preserved.

V8 training remains in `notebooks/PlacementPilotV8.ipynb` and is documented in
[the V8 protocol](docs/PILOT_V8.md). Earlier notebooks remain available.

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
