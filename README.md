# LearnedPlacement

A local CPU simulator for joint training-resource, placement, checkpoint and
recovery planning. **Pilot v4 broadens the decision space before testing RL.**
No Ray, GPU, cloud account, Z3 or actual model training is required.

## Run locally

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[notebook]'
python -m jupyter lab notebooks/PlacementPilot.ipynb
```

Python 3.10+ is required. For a fresh checkout, clone this repository and create
`.venv` with `python3 -m venv .venv` first. Restart the kernel after pulling.
The default output is **`results/pilot_v4`**; old results are preserved.
Read the editable configuration cell before running. Only Run/resume starts work.

```bash
python -m learnedplacement.expanded_runner --config configs/pilot_v4.json --output results/pilot_v4 --max-minutes 30
python -m learnedplacement.expanded_runner --output results/pilot_v4 --status
```

Each invocation stops at a cooperative 30-minute cap and saves progress. Rerun
unchanged to resume. The full 144-case matrix may need several invocations; the
notebook documents a 48-case starting subset. Plots and compact export read saved
files without starting simulations. Share **`pilot_v4.analysis.zip`**, not the raw
results directory. Changed configuration/source requires a new run directory.

## What changed

V4 chooses worker count and type (through node selection), sharding degree,
replica groups, shard positions, extra backups and their refresh rate, reserved
spares, checkpoint interval/tier/overlap/retention, and recovery admission, order,
source selection and repair boundaries. Cases vary memory, network, storage and
repair conditions. Training work is held constant across allocations.

It compares speed grouping, a stronger domain-aware tuned heuristic, coordinate
search, random search, local search and a matched failure-unaware selector.
Plans are evaluated on fresh paired failure traces. Case-level splits and a
feasible candidate interface prepare for later learning; **RL is not implemented
or trained in this pilot**. These baselines are not published-system reproductions.

See [the complete experiment description](docs/PILOT_V4.md) for variables,
objective, constraints, profiles, budgets, comparison fairness and limitations.
The simulator is an uncalibrated analytical model; it does not establish real
HSDP performance or publication novelty. Tensor/pipeline parallelism, training
batch changes, asynchronous replica catch-up and multi-job scheduling are staged
for later modeling rather than represented by unsupported flags.

## Previous experiments and checks

V3 remains available in `notebooks/PlacementPilotV3.ipynb`, `configs/pilot_v3.json`
and `python -m learnedplacement.runner`. See [v3 documentation](docs/PILOT_V3.md)
and [the original simulator model](MODEL.md). Old result files are readable;
source fingerprints intentionally prevent mixing old and updated runs.

```bash
python -m unittest discover -s tests -v
```

Tests use tiny deterministic simulations and temporary result directories.
The full research experiment is left for the user to run.
