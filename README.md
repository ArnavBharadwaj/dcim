# Digital twin-assisted workload allocation and cooling optimisation with GNNs

Research code for a workshop paper. A digital twin of a data hall whose internal state
is a directed graph; a message-passing network predicts per-rack inlet temperature; two
controllers consume that forecast.

## State of the work

| Phase | | |
|---|---|---|
| 0 | Simulator audit | **done** — gate failed, see below |
| 0b | Replacement thermal kernel | **done** |
| 1 | Data generation | **done** — 3 halls, 500 h each, gate passed |
| 2 | Baselines | **done** |
| 3 | Graph model | **done** |
| 4 | Twin properties: calibration, drift | not started |
| 5 | Controllers: placement, cooling | not started |
| 6 | Ablations | not started |

## The Phase 0 finding, which reshaped the project

SustainDC's rack inlet temperature is computed in one place,
`envs/datacenter.py:279`:

```python
rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint
```

`rack_supply_approach_temp` is a JSON constant, loaded once and never recomputed. Rack
power does not enter — not a neighbour's, not the rack's own. Driving one rack from 50%
to 100% utilisation, a 14.8 kW change, moves **no** rack's inlet temperature by a single
floating-point unit. The influence matrix is the zero matrix including its diagonal, and
`d(inlet)/d(setpoint)` is exactly 1.000000 at every rack.

Every model in the planned comparison would have tied at machine precision, and every
ablation would have been null by construction.

The response was to replace the thermal kernel with a heat-recirculation model while
keeping SustainDC for its HVAC power chain and exogenous data. **The thermal ground
truth in this project is therefore our own physics-based model, not SustainDC**, and the
paper has to say so. Details and the full list of what is whose:
[docs/thermal-model-design.md](docs/thermal-model-design.md).

## Documentation

| | |
|---|---|
| [docs/simulator-thermal-model.md](docs/simulator-thermal-model.md) | The Phase 0 audit and the measurements behind it |
| [docs/thermal-model-design.md](docs/thermal-model-design.md) | The replacement kernel: derivation, parameters, limitations |
| [docs/data-generation.md](docs/data-generation.md) | Workload replay, placement policies, the thermal gate |
| [docs/prediction-task.md](docs/prediction-task.md) | What the models predict, and why the control plan is an input |

## Setup

```bash
uv sync --extra dev
./scripts/bootstrap_external.sh    # pins SustainDC at a92b4755
./scripts/fetch_trace.sh           # Alibaba PAI GPU trace sample
```

Requires a native arm64 Python on Apple Silicon. This machine had an x86_64 Homebrew and
an x86_64 python.org build; `uv` pins its own arm64 CPython 3.11 so torch gets real
wheels and MPS.

## Running

```bash
uv run python scripts/phase0_probe_thermal_coupling.py     # audit SustainDC
uv run python scripts/phase0b_probe_new_kernel.py          # audit the replacement
uv run python scripts/phase1_generate.py --hall hall_a --seed 0
uv run python scripts/phase1_gate_figure.py
uv run python scripts/phase2_baselines.py
uv run python scripts/phase3_train_gnn.py
uv run python scripts/make_tables.py                       # tables from runs.csv
uv run pytest                                              # 144 tests
```

Every run appends one row to `results/runs.csv` with the config hash, seed, git SHA and
every metric. Tables are built from that file, not from console output.
`results/archive/` holds superseded runs kept for provenance.

## Layout

```
configs/     hydra-style configs: one per hall, per twin parameter set, per experiment
src/twin/    geometry, heat recirculation, power, thermal dynamics
src/data/    trace replay, placement policies, generation, splitting, features
src/graph/   graph construction from a hall config
src/models/  gnn, lstm, rc, lightgbm, persistence
src/eval/    metrics, runs.csv writer
scripts/     one entry point per phase
tests/       data pipeline tests
results/     runs.csv, figures
docs/        design notes
external/    pinned SustainDC checkout (gitignored)
data/        trace and generated trajectories (gitignored, regenerable)
```

## Notes for anyone picking this up

- **LightGBM runs single-threaded on purpose.** It and torch each ship an OpenMP
  runtime; on macOS arm64 they cannot both use a thread pool in one process. torch must
  be imported first and `n_jobs` must be 1. The failure mode is a segfault with no
  traceback that a shell pipeline reports as success. See `src/models/lgbm.py`.
- **Splits are by episode, never random.** Consecutive 30 s timesteps are near-identical,
  so a random split puts a sample's own near-duplicate on the other side of the boundary.
- **Normalisation statistics come from hall_a only** and are never recomputed on hall_b
  or hall_c. `assert_no_renormalisation` guards this on every path.
