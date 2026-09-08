# Phase 1: workload, trajectories, and the thermal gate

What the dataset is, how it is made, and the decisions that had to be corrected by
measurement rather than assumption.

## Workload

Source: the Alibaba PAI GPU cluster trace, `alibaba/clusterdata`,
`cluster-trace-gpu-v2020`. The repository bundles
`pai_job_duration_estimate_100K.csv`: 100,000 real jobs spanning about 290 hours, with
GPU request, CPU request, submit time and duration per job.
`scripts/fetch_trace.sh` records where it comes from; `data/` is gitignored.

| | |
|---|---|
| jobs | 100,000 |
| span | 289.8 h |
| arrivals | ~345 jobs/h, min 80, max 780 |
| duration | median 729 s, p90 13,075 s, max 535,585 s |
| GPU request | 33% ask for 1, 35% for a fraction (0.05-0.5), 15% CPU-only |

### Why the stream is bootstrapped rather than tiled

The brief asks for 500 h per hall and the trace covers 290, so the extra time has to
come from somewhere. Tiling was rejected: the split is by time, so an exact repeat
would put the same jobs in train and test, which is precisely the leak the brief warns
about.

Instead, jobs are drawn i.i.d. from the empirical pool with arrivals from a
non-homogeneous Poisson process following the trace's own hour-of-day rate profile.
This keeps the real joint distribution of GPU request, CPU request and duration, keeps
the real diurnal shape, treats every part of the timeline identically so train and test
are statistically the same, and never repeats a job at a fixed offset. It is a
resampling of real data, and the paper should describe it that way rather than as a
real replay.

Jobs are truncated at 3 hours. The trace's longest runs 149 hours; left alone, a handful
of these pin capacity for a whole episode and flatten the utilisation signal.

### Arrival rate from a target utilisation

Little's law: the expected number of busy GPU slots is the arrival rate times the mean
GPU-hours per job. Inverting it gives the rate, so "if fewer than 2% of timesteps
breach the limit, increase the load density and regenerate" is one number in
`configs/data/gen_default.yaml` rather than a manual search.

Racks hold 64 GPU slots (8 nodes x 8), so hall_a has 12,800 -- the same order as the
PAI cluster the trace came from.

| target mean utilisation | jobs/hour |
|---|---|
| 30% | 1,890 |
| 50% | 3,149 |
| 70% | 4,409 |
| 85% | 5,354 |

## Placement

Four policies, sampled per episode. SustainDC could not express any of this: its
`dc_gym.step` hardcodes one scalar utilisation for every rack, so all four would have
produced byte-identical trajectories.

| policy | behaviour | racks touched (200-rack probe) |
|---|---|---|
| `random` | uniform over racks with capacity | 170 |
| `round_robin` | cycle, skipping racks that cannot fit the job | 164 |
| `best_fit` | tightest rack that still fits | 16 |
| `corner_stack` | lowest rack index first | 13 |

`corner_stack` is deliberately bad and is in the sampling set on purpose: racks are
indexed row-major, so it piles load into one corner and produces the hot states the
brief insists the dataset must contain.

## Episodes

500 hours per hall, as 100 episodes of 5 hours at 30 s resolution. Each episode draws
its own placement policy, its own target utilisation in [0.35, 0.95], and its own
cooling schedule. A 30-minute burn-in runs unrecorded so recording starts from a
realistically occupied hall, and the twin is reset to equilibrium at that occupancy so
the first samples are not an artefact of an arbitrary starting temperature.

### The cooling schedule, and a mistake worth recording

The first version drifted the supply temperature and fan speed as a slow random walk.
Measured on the resulting data, the supply temperature moved **0.68 K over a five-hour
episode** and the fan by 0.07. The plant was effectively static within an episode, all
of the setpoint range lived *between* episodes, and the consequence showed up directly
in the prediction target:

| horizon | delta std, random walk | delta std, after the fix |
|---|---|---|
| 30 s | 0.030 K | 0.116 K |
| 60 s | 0.054 K | 0.208 K |
| 300 s | 0.201 K | 0.681 K |

A 30 s target of 0.03 K is below any real sensor's resolution; the models would have
been compared on their ability to predict numerical noise.

The fix is to schedule the cooling the way a real CRAC controller behaves: hold a
setpoint for 5 to 20 minutes with slight drift, then step it. Within-episode supply
range went from 0.68 K to the full 5.0 K, and the targets to the right-hand column
above. Implemented in `src/data/generate.py::_control_schedule`.

## Operating envelopes

The envelope is part of the hall, not a global constant, and two rounds of measurement
were needed to get it right.

**Supply temperature.** hall_a's mean recirculation rise is about 5 K. A supply setpoint
above ~21 C therefore puts every rack over the 27 C recommended limit at any fan speed,
leaving no operating decision to make. At the original 25 C cap, 67% of timesteps
breached and 7% went past the 32 C allowable limit -- a hall in permanent violation, not
a dataset. Capped at 21 C.

**Fan speed.** At the original 0.40 floor, 4.7% of timesteps passed the allowable limit.
No operator runs the fans that low at load. Floor raised to 0.55.

**hall_c needs its own envelope.** With every row facing the same way, a rack's inlet
sits directly across a mixed aisle from the next row's exhaust, so the recirculation
path is about one aisle width instead of a full row pitch. Measured at 85% utilisation
it runs roughly 12 K hotter than hall_a at identical setpoints:

| supply | hall_a max inlet | hall_c max inlet |
|---|---|---|
| 16 C, fan 1.0 | 22.0 | 25.4 |
| 18 C, fan 0.8 | 24.4 | 28.4 (at 16 C) |
| 21 C, fan 0.6 | 30.5 | 38.0 |

Real operators of uncontained halls run colder supply air and keep the fans up, because
they have to. hall_c is given 13-18 C and fan 0.60-1.0. Held to hall_a's envelope it
would sit permanently in violation and the dataset would carry no usable compliant
states. Hall configs can override `limits` as well as `recirculation`.

## The thermal gate

The brief: plot the distribution of rack inlet temperature, and if fewer than 2% of
timesteps breach the thermal limit, increase the load density and regenerate. Limit is
the ASHRAE 2021 class A1 recommended maximum, 27 C.

Results are in `results/figures/phase1_thermal_gate.png` and in `results/runs.csv`
under `phase=1`. All three halls pass with a defensible distribution: a clear violation
tail, nothing past the 32 C allowable limit.

## What is in the dataset

Per hall, `data/trajectories/<hall>.npz`:

| array | shape | |
|---|---|---|
| `util` | (T, N) | rack utilisation, % |
| `inlet` | (T, N) | rack inlet temperature, C -- the prediction target |
| `power` | (T, N) | rack power, W |
| `supply_temp`, `fan_frac` | (T,) | the two control inputs |
| `provisioning`, `crac_return` | (T,) | plant state |
| `episode_id`, `policy_id`, `time_s` | (T,) | provenance for splitting |

`<hall>_meta.json` carries the config hash, the seed, the resolved generation config
and per-episode records of policy, target utilisation, arrival rate and peak inlet
temperature.

## Splitting

By episode, which is a split by time -- episodes are laid down in order -- and which
additionally guarantees no input window ever spans a boundary. 70 / 15 / 15.

A random split would be catastrophic here and the brief says so: consecutive 30 s
timesteps are near-identical, so a random split puts a sample's own near-duplicate on
the other side of the boundary and every model looks excellent. `tests/test_split.py`
asserts every training timestep precedes every test timestep and that no window crosses
an episode boundary.
