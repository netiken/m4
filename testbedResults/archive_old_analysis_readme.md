# New Scenario Analysis

This directory collects the raw measurements from the physical testbed (`real_world.txt`) together with the simulator outputs produced by each stack (e.g. FlowSim, ns-3/UNISON, m4). The refreshed `analyzev2.py` script compares every scenario locally—no SSH hops required—and generates the plots/csv artefacts that the paper uses.

## Directory Layout
- `expirements_4/`, `expirements_12/`, `expirements/` – scenario groups. Each scenario folder must contain a `real_world.txt` plus the usual debug artefacts (plots, logs, etc.).
- `analyzev2.py` – local comparison driver.
- `generate_sweep_figures*.py`, `summarize_sweep_figures.py`, … – helper scripts kept from older workflows.

> The script auto-selects `expirements_4` if it exists; pass `--local-base` when you want a different group.

## Requirements
- Python 3.9+
- `numpy`
- `matplotlib`

Create/activate whatever environment you use for the rest of the project, then install:

```bash
python3 -m pip install --upgrade numpy matplotlib
```

## Running The Comparison

```bash
cd new_scenarios
python3 analyzev2.py --local-base expirements_4
```

By default the script will look for:

- FlowSim outputs under `../flowsim/sweeps_4/flowsim_output.txt`
- ns-3 (UNISON) outputs under `../UNISON/sweeps_4/ns3_output.txt`

Use matching `_12` directories automatically when the local base ends with `_12`. You can disable the defaults (`--no-default-sources`) and/or add extra stacks:

```bash
python3 analyzev2.py \
  --local-base expirements_4 \
  --source flowsim=../flowsim/sweeps_4:flowsim_output.txt \
  --source m4=/path/to/m4/sweeps:m4_output.txt:1.0
```

Source syntax: `name=path:filename[:rdma_scale]`. Paths are resolved relative to the repository root unless they are absolute. The optional `rdma_scale` compensates stacks that report RDMA durations on a different scale (ns-3 needs `2.0`).

### Useful Options
- `--trim` – drop warm-up samples from both ends of each series (default `20`).
- `--output-dir` – change where plots go (default `sweepfigures/`).

## Outputs
The script creates/updates:

- `sweepfigures/relative_errors.png` – overall CDF.
- `sweepfigures/cdf_<scenario>.png` – per-scenario CDFs.
- `sweepfigures/box_whiskers_all_keys.png` and `sweepfigures/box_<scenario>.png` – distribution summaries.
- `sweepfigures/sweepcsv/<scenario>/key_<client>_<op>.csv` – table aligning local + simulator durations.
- `relative_errors_*.npy` – NumPy arrays with raw error samples (one per source + combined).

All artefacts are regenerated on every run, so it is safe to rerun the script after adding new simulator outputs.
