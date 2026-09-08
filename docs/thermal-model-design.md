# The replacement thermal model

What we built after the Phase 0 gate failed, why each piece is shaped the way it is,
and what a reviewer is entitled to push back on.

Read `docs/simulator-thermal-model.md` first. It records the measurement that forced
this: SustainDC computes `T_inlet[i] = const[i] + T_setpoint`, so the influence matrix
`dT_inlet/dP` is the zero matrix including its diagonal, and every model in the planned
comparison would tie at machine precision.

Decision taken at the gate: **Option A**. Keep SustainDC for what it does credibly,
replace the inlet computation, and say so plainly in the paper.

## What is ours and what is SustainDC's

| Piece | Source | Why |
|---|---|---|
| Rack inlet temperature | **ours** (`src/twin/recirculation.py`) | SustainDC's has no power term at all |
| Thermal dynamics | **ours** (`src/twin/thermal.py`) | SustainDC is memoryless, so horizons would be meaningless |
| Rack power and airflow | **ours** (`src/twin/power.py`) | SustainDC's curves are hand-tuned; see below |
| Hall geometry and aisles | **ours** (`src/twin/geometry.py`) | SustainDC has no geometry, only a per-rack constant |
| HVAC chain: chiller COP, cooling tower, pumps | SustainDC | substantive, physically grounded, worth keeping |
| Weather and carbon intensity series | SustainDC | real data, no reason to reimplement |

This is a large fraction of the simulator. The paper must not describe its thermal
results as "evaluated on SustainDC". The honest phrasing is that the twin is our own
physics-based model of heat recirculation, that SustainDC supplies the plant-side power
chain and the exogenous data, and that the thermal ground truth is therefore synthetic.
That is a real limitation and it should be stated in the paper rather than discovered by
a reviewer.

## Formulation

### Heat recirculation

Standard heat-recirculation account of a raised-floor hall, following Tang et al.,
*Thermal-aware task scheduling for data centers through minimizing heat recirculation*.
Let

- `a_ij` = fraction of rack *i*'s exhaust flow drawn into rack *j*'s inlet,
- `k_i = rho * c_p * f_i`, the thermal mass flow of rack *i*'s fans, in W/K.

Rack *i* raises the air it draws by `P_i / k_i`, so `T_out = T_in + K^-1 P`. The air
entering rack *j* mixes CRAC supply with other racks' exhaust:

```
k_j T_in,j  =  sum_i a_ij k_i T_out,i  +  (k_j - sum_i a_ij k_i) T_supply
```

Substituting and rearranging gives `T_in = T_supply + D P` with

```
D  =  (K - A^T K)^-1 - K^-1
   =  K^-1 [ (I - A^T)^-1 - I ]                    <- the form we compute
   =  K^-1 [ A^T + (A^T)^2 + (A^T)^3 + ... ]       <- Neumann series
```

The second form is better conditioned. The third makes the multi-hop structure
explicit: term *m* is air that passed through *m* racks before arriving, which is what
licenses `hop_decomposition` reporting "influence at *m* hops".

`tests/test_recirculation.py::test_D_satisfies_the_mixing_energy_balance` re-derives
the inlet temperature straight from the mixing equation above and checks the closed form
against it. If that algebra were wrong, nothing downstream would catch it.

### Building A from geometry

`geometric_kernel` gives an unnormalised propensity for *i*'s exhaust to reach *j*'s
inlet:

```
kernel_ij  =  exp(-d_ij / lambda)                 distance decay, exhaust face to inlet face
            * row_attenuation^(rows_crossed - 1)  cost of climbing over further rack rows
            * [w_up + (w_dn - w_up)(cos t + 1)/2] asymmetry along the return path
```

`cos t` is the cosine of the angle between the *i*-to-*j* vector and *i*'s return
direction, so racks downstream of *i* toward its nearest CRAC receive more of its air
than racks upstream. **This is the only reason the induced graph is directed**, and it
is what makes the directed-vs-undirected ablation a real question rather than a
formality. Measured asymmetry of `D` on hall_a: `sum|D - D^T| / sum|D| = 0.63`.

`cross_interference` then normalises so row *i* sums to `escape_i`:

```
A_ij  =  escape_i * kernel_ij * k_j / sum_j' (kernel_ij' * k_j')
```

Two things about that line.

**Mass is conserved at the source.** Rack *i* sheds exactly `escape_i * k_i` of
recirculating air, never more.

**The `k_j` weight is load-bearing, not cosmetic.** It makes each target's share of the
recirculating air proportional to how hard that target is pulling. Without it, a rack's
recirculated intake is fixed by geometry while its total intake grows with its fan
speed, so ramping its fans *dilutes* its own inlet. We measured that artefact on hall_a
before fixing it: driving one rack from 50% to 100% utilisation **cooled it by 0.79 K**
while its neighbours warmed by only 0.08 K. A placement controller trained against that
would have learned to stack load onto the hottest racks. Real halls run the other way --
a rack whose airflow demand outruns the cold air delivered to its aisle makes up the
deficit by ingesting exhaust. With the weight in place the same perturbation warms the
source by +0.07 K and peaks at +0.09 K on its immediate downstream neighbour.
Regression tests: `test_entrainment_weight_makes_rise_independent_of_own_airflow` and
`test_loading_a_rack_never_cools_it`.

### Leakage, and where the state dependence comes from

```
escape_i = clip( escape_base * prox_i * (1 + underprovision_gain * max(0, 1 - phi)),
                 0, escape_max )
```

- `prox_i` rises with rack *i*'s distance from its nearest CRAC return, normalised so
  the hall mean multiplier is 1. Geometric, fixed per hall.
- `phi` is the provisioning ratio, total CRAC supply flow over total rack airflow
  demand. When the racks' own fans move more air than the CRACs supply, the deficit can
  only be made up by ingesting recirculated air.

`phi` is the mechanism that puts the **cooling controller in charge of the coupling
structure** rather than just its offset. `K` also moves with rack fan speed, and `D`
scales as `1/K`. So `D` is a function of the control inputs, and no single fixed linear
operator represents the hall. Measured on hall_a at uniform full load:

| CRAC fan | mean inlet | max inlet |
|---|---|---|
| 1.0 | 25.22 | 26.45 |
| 0.9 | 25.22 | 26.45 |
| 0.8 | 25.98 | 27.40 |
| 0.7 | 27.20 | 28.92 |
| 0.6 | 28.60 | 30.67 |

Note the flat region between 1.0 and 0.9: while `phi >= 1` the fans can be turned down
for free. Below that the response is sharply nonlinear. Finding and sitting on that knee
is exactly the job of the Phase 5 cooling controller.

### Dynamics

Each rack gets a first-order lag toward its steady-state target, integrated exactly:

```
alpha = 1 - exp(-dt / tau)          tau = 60 s
T_inlet <- T_inlet + alpha * (T_steady - T_inlet)
```

Power is evaluated at the **current** inlet temperature, not the steady-state one, so
the feedback loop (warmer air, more power and more fan flow, warmer air) resolves
through time instead of through an implicit solve. `steady_state()` does run the
implicit solve, damped Picard, and is used only to start an episode in equilibrium.

Without this lag, predicting temperature at *t+H* collapses into predicting workload at
*t+H* and the 30 s / 60 s / 300 s horizons all measure the same thing. With `tau = 60 s`
at 30 s resolution, 30 s is transient and 300 s is nearly steady-state.

### Power and airflow

SustainDC's power curves are not usable as physics, and we did not adopt them:

- The CPU power ratio carries a hardcoded `+0.05` temperature slope
  (`(self.m_cpu+0.05)*inlet_temp + self.c_cpu`), about 5% of full-load power per kelvin.
  Measured server behaviour is roughly an order of magnitude below that. Using it would
  make the twin's dynamics an artefact of that one constant.
- The IT-fan curve is scaled by three constants introduced in the source with the
  comments `#1 -> 10`, `#5 -> 5` and `#100 -> 20`, and returns an airflow "ratio" above
  3 at ordinary operating points. It is a fit, not a fan law.

Ours keeps the same structure -- power rising with utilisation and with inlet
temperature, an idle floor, fan flow rising with both -- with a cube-law fan power,
which is the actual affinity law and is what SustainDC itself uses for its CRAC fans.
Rack airflow is sized from the design temperature rise, `f = P / (rho c_p dT)`, rather
than being a free parameter.

## Parameters and their provenance

Everything below lives in `configs/twin/default.yaml`. None of it is measured by us;
these are modelling choices inside published ranges, and the paper should present them
as such.

| Parameter | Value | Basis |
|---|---|---|
| `decay_length_m` | 2.5 | recirculation e-folding length, order of an aisle pitch |
| `row_attenuation` | 0.35 | cost of climbing over a further rack row |
| `weight_downstream` / `weight_upstream` | 1.0 / 0.25 | exhaust is carried toward the return and leaks along the way |
| `escape_base` | 0.30 | hall-average recirculated fraction, partial containment |
| `escape_max` | 0.55 | above this no steady state exists; a guard, not a physical value |
| `crac_distance_gain` | 0.45 | racks far from a return leak more |
| `underprovision_gain` | 1.20 | steepness of the leakage response to `phi < 1` |
| `cpu_temp_coeff_per_k` | 0.004 | leakage and internal fan response, ~0.4% of full load per K |
| `design_delta_t_k` | 12.0 | standard air-cooled rack design point |
| `time_constant_s` | 60.0 | rack-air thermal mass |
| `design_provisioning` | 1.15 | CRAC flow margin over rack demand at full load |
| rack size | 8 x 4 kW nodes, ~34 kW | GPU-dense rack, matching the paper's "AI-heavy" framing |

Calibration target was the supply-to-inlet rise, which for air-cooled halls with partial
containment is a few kelvin on average with hot spots several kelvin above that. At
uniform full load on hall_a the model gives a mean rise of **5.22 K** and a maximum of
**6.45 K** over a 20 C supply, putting the hall at the edge of the ASHRAE recommended
27 C limit at full load. That is a useful place to sit: violations are reachable without
contrivance, and the controllers have real work to do.

### A parameter that is weakly identified

`row_attenuation` barely moves anything in a contained layout: sweeping it from 0.05 to
0.9 changes hall_a's spread from 2.42 K to 2.78 K, because distance decay already kills
the multi-row paths. It does matter in an uncontained one, where a single aisle serves
both inlets and exhausts, moving hall_c's spread from 11.50 K to 6.65 K. We are keeping
it, and flagging it here rather than presenting it as a tuned quantity.

## Measured properties

From `scripts/phase0b_probe_new_kernel.py` on hall_a, 200 racks, one rack driven from
50% to 100% (a 13.9 kW change), supply 20 C, CRAC fans 100%:

| Quantity | Replacement kernel | SustainDC |
|---|---|---|
| max \|d inlet\| at the perturbed rack | 0.292 K | 0.000 K |
| max \|d inlet\| at any other rack | 0.255 K | 0.000 K |
| rack pairs coupled above 1 mK | 48.7% | 0% |
| influence beyond one hop | 29.7% | n/a, D is zero |
| `D` density | 1.000 | 0.000 |
| `D` asymmetry | 0.63 | undefined |
| `d(inlet)/d(setpoint)` | 0.878 to 0.923, varying by rack | exactly 1.000000 everywhere |

Influence decays smoothly with distance rather than vanishing: mean \|response\| is
0.066 K at one grid step, 0.019 K at four, 0.0025 K at ten.

The hop profile is 70.3 / 20.9 / 6.2 / 1.8 / 0.5 %. Three message-passing layers are
therefore justified and the layer ablation should show a real curve that flattens, which
is the honest form of that result.

### The three halls differ

At uniform full load, supply 20 C:

| Hall | racks | mean rise | max rise | spread | placement leverage |
|---|---|---|---|---|---|
| hall_a, 10x20 paired, 4 CRAC | 200 | 5.22 | 6.45 | 2.47 | 1.94 K |
| hall_b, 7x20 paired, 3 CRAC | 140 | 5.22 | 6.90 | 4.78 | 1.15 K |
| hall_c, 10x20 uncontained, 3 CRAC | 200 | 5.21 | 10.59 | 9.72 | 3.12 K |

"Placement leverage" is the peak inlet temperature when 30% of racks run at 100% stacked
in a corner, minus the peak when the same total load is spread uniformly. It is nonzero
everywhere, so the placement controller has something to optimise -- which it did not,
in SustainDC, where every candidate rack scores identically.

Note the mean rise is the same in all three. That is a property of normalising each
rack's leakage to `escape_base`: geometry decides *where* recirculated heat lands, not
*how much* recirculates. An uncontained hall should also leak more in total, so hall_c
will likely want an `escape_base` override when its config is written in Phase 1. Hall
configs can already override any recirculation parameter.

## What a reviewer will ask

- **"Your thermal ground truth is your own model."** True, and it must be said in the
  paper. The defence is that the alternative was a simulator whose influence matrix is
  identically zero, that the formulation is the standard one from the thermal-aware
  scheduling literature rather than invented here, and that the parameters sit in
  published ranges. It is not a defence to call it SustainDC.
- **"Then a GNN fits your generative model, not a data centre."** Partly true, and the
  transfer and drift experiments are the answer: hall_b and hall_c are different halls,
  and Phase 4's drift injection changes the hall after training.
- **"Is the coupling just linear?"** No. `D` depends on the control inputs through `K`
  and through `phi`, and the leakage response to `phi` has a knee. A single fixed linear
  operator cannot represent it. Whether that is *enough* nonlinearity to separate the
  GNN from LightGBM with neighbour features is an open empirical question -- Phase 2
  runs the baselines first precisely so that it is answered before the model is built.
- **"Why is the recirculation kernel that shape?"** It is a parametric choice, not a CFD
  result. Documented above, and every coefficient is in one config file.

## Known limitations

1. Two-dimensional. Real recirculation is worst over the tops of racks, and rack height
   is not modelled.
2. Supply air is uniform across the hall. Perforated-tile flow actually varies with
   plenum pressure, which is a real source of spatial structure we do not have.
3. `phi` is global. A locally starved aisle in an otherwise well-provisioned hall is not
   represented.
4. No humidity, no leakage through walls or cable openings, no external heat gain.
5. The CRAC return temperature is a single flow-weighted mixture. Per-unit returns would
   let a single CRAC failure show a localised signature; currently drift injection acts
   through total flow and its share weighting.
