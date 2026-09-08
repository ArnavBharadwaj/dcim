# SustainDC's rack thermal model: what it actually computes

Phase 0 gate report.
Repo probed: `HewlettPackard/dc-rl` at commit `a92b475`, cloned to `external/dc-rl/`.
Probe script: `scripts/phase0_probe_thermal_coupling.py`.
Artefacts: `results/phase0_coupling.csv`, `results/phase0_summary.json`,
`results/figures/phase0_coupling.png`.

## The one line that matters

Rack inlet temperature is computed in exactly one place in the whole repository,
`envs/datacenter.py:279`, inside
`DataCenter_ITModel.compute_datacenter_IT_load_outlet_temp`:

```python
rack_supply_approach_temp = rack.clamp_supply_approach_temp(rack_supply_approach_temp)
rack_inlet_temp = rack_supply_approach_temp + CRAC_setpoint
```

That is the entire inlet model:

    T_inlet[i] = a[i] + T_CRAC_setpoint

`a[i]` is `RACK_SUPPLY_APPROACH_TEMP_LIST[i]`, a constant read from
`utils/dc_config.json` and never written again at runtime. `clamp_supply_approach_temp`
(`datacenter.py:209`) clamps it into `[3.8, 5.3]`; in the shipped config every value is
already 5.0 or 5.3, so the clamp is inert.

Nothing else enters. Not rack power, not any neighbour's power, not IT fan speed, not
CRAC airflow, not the previous timestep, not ambient temperature.

## The four gate questions

### Is the mapping linear in neighbouring rack power?

No — it is *constant* in neighbouring rack power. There is no neighbour term of any
order. It is also constant in the rack's **own** power.

Measured (`dc_config.json`, 20 racks, CRAC setpoint 20 °C, all racks at 50 %
utilisation, one rack driven to 100 %, which changes that rack's draw by **14.8 kW**):

| quantity | measured |
|---|---|
| max \|Δ inlet\| at any *other* rack | **0.000e+00 K** |
| max \|Δ inlet\| at the *perturbed* rack | **0.000e+00 K** |
| max \|Δ outlet\| at any *other* rack | 0.000e+00 K |
| max \|Δ outlet\| at the *perturbed* rack | 4.593 K |
| float64 spacing at these magnitudes | 3.6e-15 K |

The zeros are bit-exact, not "small". A 14.8 kW swing on one rack moves no rack's inlet
temperature by a single ULP. The same holds for a 30 % → 100 % perturbation.

Sweeping one rack's utilisation across the full range with its neighbours held fixed:

| own load % | own inlet °C | own outlet °C | own power W |
|---|---|---|---|
| 0 | 25.3000 | 29.9786 | 39,852 |
| 25 | 25.3000 | 32.1358 | 46,825 |
| 50 | 25.3000 | 34.0938 | 53,797 |
| 75 | 25.3000 | 35.8846 | 60,770 |
| 100 | 25.3000 | 37.5333 | 67,742 |

Inlet is flat to four decimals; outlet is the only thing that moves.

### Are the neighbour coefficients fixed, or do they change with CRAC state?

There are no neighbour coefficients. The one coefficient that exists is the
per-rack offset `a[i]`, and it is fixed: a JSON constant, loaded once in
`utils/dc_config_reader.py:64`, never recomputed.

The CRAC setpoint enters as a pure additive shift with unit gain, identically at every
rack. Measured `d(T_inlet[i]) / d(T_setpoint)` over a 16 → 24 °C sweep:
**min 1.000000, max 1.000000** across all racks. So the setpoint cannot change the
*spatial pattern* of inlet temperatures at all — it rigidly translates the whole
profile.

CRAC fan speed is not a control input in this simulator. The action space
(`utils/make_envs_pyenv.py:124-132`) is a `Discrete` over a CRAC setpoint delta only.
`CRAC_SUPPLY_AIR_FLOW_RATE_pu` is a config constant, and `CRAC_Fan_load`
(`datacenter.py`, `calculate_HVAC_power`) is therefore a constant too, independent of
load and setpoint.

### Does air recirculation appear at all, or only a per-rack offset?

Only a per-rack offset. `grep -rni "recircul"` over the repository returns nothing. The
config comments say the approach temperatures "can be pre-computed from CFD analysis"
(`dc_config_reader.py:55-70`) and cite Sun et al. 2021 Table 5 Scenario 19 — i.e.
recirculation was baked into two scalars offline and then frozen. In the shipped config
there are exactly **two distinct values** over 20 racks: 5.3 °C for the four racks at
each end row, 5.0 °C for the middle. That is the entire spatial structure of the hall.

There is exactly one path by which one rack's load influences a hall-level thermal
quantity, and it is not spatial:

    T_CRAC_return = mean_i ( T_outlet[i] + b[i] )        # datacenter.py:531

an unweighted 1/N average over every rack, used only for HVAC power. Measured: driving
one rack from 50 % to 100 % moves the average return temperature by **+0.172 K**. This
is mean-field and all-to-all with identical 1/N weights — it carries no geometry, and it
never feeds back into any rack's inlet temperature.

### How many hops of spatial influence does the model contain?

**Zero.** Not one hop, not two. The influence matrix ∂T_inlet[i]/∂P[j] is the zero
matrix, including the diagonal.

There is also no time dynamics: `compute_datacenter_IT_load_outlet_temp` is memoryless,
with no thermal capacitance, no state carried across `step()`, and no lag.

## Two further blockers found while probing

These are not part of the four questions but they change what Phase 1 can be.

**1. The gym wrapper cannot express per-rack workload placement.** `dc_gym.step`
(`envs/dc_gym.py:175`) hardcodes:

```python
ITE_load_pct_list = [self.cpu_load_frac*100 for i in range(self.DC_Config.NUM_RACKS)]
```

Every rack always carries the identical utilisation, taken from one scalar. The
thermal kernel accepts a per-rack list, but nothing in the environment ever supplies a
non-uniform one. As shipped, SustainDC has no notion of *which* rack a job lands on —
so a placement controller has no lever, and the four placement policies in the Phase 1
brief (random / best-fit / round-robin / corner-stacking) would all produce byte-identical
trajectories.

**2. Only one of the four hall configs loads.** `dc_config_dc1/2/3.json` are stale
relative to `dc_config_reader.py`:

| config | derived NUM_RACKS | supply-temp list len | server-power list len | loads? |
|---|---|---|---|---|
| `dc_config.json` | 20 | 20 | 20 | yes |
| `dc_config_dc1.json` | 20 | 16 | 20 | no — `KeyError: CHILLER_COP_BASE` |
| `dc_config_dc2.json` | 25 | 20 | 20 | no — same KeyError |
| `dc_config_dc3.json` | 25 | 25 | 20 | no — `AssertionError` on list length |

All three carry `CHILLER_COP` where the reader expects `CHILLER_COP_BASE`, and their
list lengths disagree with their own declared geometry. Note also that
`DataCenter_ITModel.__init__` builds racks with `zip(range(num_racks), rack_CPU_config)`,
which silently truncates rather than erroring when lists are short.

## What this means for the paper

The Phase 0 gate was written to answer: *if the heat model is a fixed linear sum over
two hops, a GNN fits it perfectly and so does every baseline.* The answer is worse than
that hypothesis. The model is not a two-hop linear sum. It is a **rank-1 constant**:
one lookup plus one scalar.

Concretely, against the stock simulator:

- Persistence, the deliberately weak calibration baseline, is **exact** whenever the
  setpoint is unchanged. Zero error, not small error.
- A two-parameter linear regression on `(rack_id_onehot, setpoint)` is exact to machine
  precision. So is LightGBM. So is the RC fit. So is the LSTM. So is the GNN.
- Every entry in the Phase 2 accuracy table is 0.000 ± 0.000 at every horizon.
- Every Phase 6 ablation is null by construction: shuffling edges, removing edge
  features, dropping to one layer, and replacing attention with mean all cost exactly
  nothing, because the graph is not carrying information in the first place.
- The Phase 5 placement controller has no signal to optimise: predicted peak inlet
  temperature is identical for every candidate rack, so argmin is arbitrary.

This is not a tuning problem or a "needs more data" problem. It is a statement about
what the simulator computes.

## Options

Recorded here for the Phase 0 decision; not acted on.

**A. Replace the thermal kernel with a recirculation model.** Keep SustainDC for
workload, weather, carbon intensity, and the HVAC/chiller/cooling-tower power chain —
those parts are substantive — and substitute the inlet computation with a cross-
interference matrix, the standard formulation in the thermal-aware scheduling
literature (Tang et al.'s heat-recirculation matrix). Inlet becomes
`T_inlet = T_supply + D · P`, with `D` built from hall geometry, plus a capacitance
term for dynamics. This restores per-rack placement sensitivity and multi-hop coupling,
and makes hall_b / hall_c genuinely different halls. Cost: the thermal ground truth is
then our own model, so the paper must be honest that it evaluates a GNN against a
synthetic-but-physically-motivated twin, not against SustainDC.

**B. Switch simulator.** Something with real airflow coupling. Much larger time cost,
and most open options have the same problem in a different form.

**C. Reframe the paper.** Drop the accuracy claim, keep the control claim, and argue
about the cooling setpoint loop only, which SustainDC does support. This forfeits the
graph contribution and the title.

**A is the only option that preserves the paper as written.** It requires deciding, up
front and in writing, what is SustainDC's and what is ours.
