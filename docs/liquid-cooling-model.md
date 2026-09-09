# The liquid-cooled twin

A direct-to-chip model for AI halls, and the reason it is a better subject for this
paper than the air-cooled one.

## Why build it

Phase 2 measured, on the air-cooled twin, that neighbouring racks' inlet temperatures
correlate at **0.9953**. Own-rack features reach R² = 0.825 on the 300 s target; adding
every neighbour feature moves it to 0.827. Air recirculation is diffuse, so every rack
in a region shares one thermal environment and a rack's own temperature already encodes
what its neighbours are experiencing. A graph model has little room to win, and the
edge-shuffle ablation was expected to come back near-null.

Liquid cooling does not work that way, and the difference is the argument.

In a direct-to-chip hall the coolant leaves a Coolant Distribution Unit, runs along a
branch manifold, and is drawn off in parallel by the racks on that branch. Racks sharing
a branch share a supply temperature, a flow budget and a CDU. Racks on a different loop
share **nothing** -- however close they stand.

So thermal coupling follows the plumbing, not the floor plan. That is exactly the
structure a Euclidean *k*-nearest-neighbour feature set cannot represent, because it
picks neighbours by distance.

## What the probe measures

`scripts/liquid_probe.py`, on hall_l1 at 85% utilisation, 34 °C facility water, pumps at
85%. Each rack is given a 5 kW step and the coolant supply response is measured at every
other rack.

| relationship | pairs | mean \|ΔT_coolant\| | mean distance |
|---|---|---|---|
| same branch (shared manifold) | 1,920 | **0.01817 K** | 3.40 m |
| same CDU, different branch | 2,048 | 0.00960 K | 4.24 m |
| different CDU | 12,288 | **0.00000 K** | 8.99 m |

Note the middle column against the right one: the *most strongly* coupled group is not
the physically closest. And the decisive comparison, restricted to racks within one row
pitch of each other:

| physically within 2.6 m | pairs | mean \|ΔT_coolant\| |
|---|---|---|
| on the same CDU | 1,232 | 0.01753 K |
| on different CDUs | **276** | **0.00000 K** |

**276 pairs of racks stand side by side and are thermally independent.** A *k*=4
neighbour feature set built on distance would pull those racks in and learn from
channels that carry no signal.

Summarised over all pairs:

- correlation of coupling with Euclidean distance: **−0.448**
- variance of coupling explained by loop topology: **0.645**

## The model

Heat splits. Cold plates take `liquid_fraction` (0.80) of rack power; the rest leaves as
air and recirculates through the air twin's kernel unchanged, with rack airflow scaled to
the air share because a liquid-cooled rack ships far smaller fans.

At each CDU, the secondary supply sits above the facility water by an approach that
degrades as the unit approaches its rated duty:

```
T_loop = T_facility + approach_min * (1 + approach_gain * Q_loop / capacity)
```

That term couples every rack on a loop to every other rack on it — block structure by
loop rather than by geography.

Along a branch:

```
m_dot,i    = m_design * pump_frac * (1 - droop * d_i)
T_supply,i = T_loop + gain * d_i + crosstalk * (upstream mixed return rise)
T_return,i = T_supply,i + Q_liquid,i / (m_dot,i * c_p)
T_case,i   = T_supply,i + case_rise_full_load * (Q_i / Q_design) * (m_design/m_dot,i)^0.4
```

### Two modelling choices that mattered, both found by measuring

**Cross-talk is what makes the graph non-trivial.** The first version had only a static
manifold gain, so perturbing any rack moved every rack on its loop by *exactly* the same
amount. Topology then explained **100%** of the coupling variance — meaning a single
categorical "which CDU" feature would capture the entire structure, and there would be no
graph problem at all. Adding supply/return manifold cross-talk, where coolant reaching a
rack has been warmed by the return of every rack upstream of it, makes the coupling
directed and ordered within a branch. Explained variance fell to 0.645, and the residual
is the part that needs a graph.

**The case-to-coolant rise is per device, not per rack.** It was first written as K per
kW of *rack* power, which made a rack with more GPUs look like it ran each GPU hotter.
Every accelerator has its own cold plate fed in parallel off the rack manifold, so the
rise is set by per-device power. Expressed now as a rise at full rack load against a
design reference.

## Operating envelope

ASHRAE liquid classes W32 through W45, measured on hall_l1:

| load | facility water | pump | coolant supply | max case | throttling |
|---|---|---|---|---|---|
| 90% | 26 °C | 1.00 | 33.2 | 60.4 | no |
| 90% | 34 °C | 1.00 | 41.3 | 69.5 | no |
| 90% | 42 °C | 1.00 | 49.5 | 78.6 | no |
| 100% | 42 °C | 0.80 | 50.0 | 84.6 | no |
| 100% | 42 °C | 0.55 | 50.1 | **91.1** | **yes** |
| 60% | 42 °C | 0.55 | 48.4 | 77.6 | no |

At the cold end the hall has easy margin; at the warm end, where a hall with no
mechanical chilling actually runs, the 85 °C case limit binds and pump speed is what
saves it.

**Two independent cooling levers with different costs**, which the air hall did not have.
Facility water temperature is slow and plant-wide. Pump speed is fast, per-loop, and moves
chip temperature through the cold plate's convective coefficient without moving the
coolant supply at the head of a branch at all.

## What this changes for the paper

1. **The accuracy experiment has room again.** The air hall's 0.995 neighbour correlation
   capped what any spatial model could add. Here the strongest coupling is between racks
   that a distance-based model would not select.
2. **The edge-shuffle ablation becomes a real test.** Shuffling edges in the air hall was
   expected to cost almost nothing. Shuffling them here should destroy the model, because
   the edges carry the plumbing and nothing else encodes it.
3. **The directed-edge ablation is no longer cosmetic.** Manifold cross-talk is strictly
   upstream-to-downstream, so reversing an edge is physically wrong rather than merely
   uninformative.
4. **Placement has a sharper lever.** Where to put a job is now partly a question of
   which loop has headroom, which is invisible to a floor-plan heuristic.

## Limitations

- **Parallel racks off a manifold, not a series cascade.** Immersion tanks and some
  rear-door designs are genuinely serial and would have stronger ordering still.
- **Flow is set by position, not solved hydraulically.** A rack drawing more does not
  starve its neighbours; a proper network solve would add another coupling path.
- **Single-phase only.** No two-phase or evaporative cold plates.
- **The CDU approach is a smooth ramp in duty**, not a heat-exchanger effectiveness
  curve with a real NTU.
- **Everything here is our model.** As with the air twin, the thermal ground truth is
  synthetic, and the paper has to say so.
