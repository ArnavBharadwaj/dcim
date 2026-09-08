# The prediction task

What the models are asked to do, and one formulation error that had to be corrected
before any of the numbers meant anything.

## The task

Given a rack's recent history and the plant's setpoint trajectory over the next *H*
seconds, predict the **change** in that rack's inlet temperature over *H*.

    inputs   utilisation and inlet-temperature history (6 lags = 3 minutes),
             current power, one-step deltas,
             current supply temperature, fan speed, provisioning ratio,
             the control plan over the horizon,
             and for the neighbour-aware baselines, the same for the k nearest racks
    target   T_inlet(t + H) - T_inlet(t)
    H        30 s, 60 s, 300 s

Delta rather than absolute temperature, because a model predicting absolute
temperature can score well by learning a constant per-hall offset -- exactly the thing
that must not transfer.

## The control plan is an input, not something to forecast

The first version of this task gave the models the plant state at *t* and asked for the
temperature at *t + H*. That is the wrong question, and measuring showed how wrong.

On hall_a, decomposing the variance of the hall-mean temperature change:

| horizon | target std | R² from control actions taken *after* t | R² from the state at t |
|---|---|---|---|
| 30 s | 0.113 K | 0.538 | 0.028 |
| 60 s | 0.203 K | 0.565 | 0.027 |
| 300 s | 0.656 K | **0.736** | 0.043 |

At 300 s, three quarters of what the models were being scored on was decided by
setpoint changes that had not happened yet at prediction time. The task was mostly
"guess what the operator will do next", which no model can do and which is not the
question a digital twin exists to answer. The symptom was visible in the results before
the cause was: LightGBM scored *worse than persistence* at 300 s.

A twin answers **"if I set the plant to this, what happens"**. So the setpoint
trajectory over the horizon is part of the input:

| feature | |
|---|---|
| `plan_d_supply` | supply temperature at t+H minus at t |
| `plan_d_fan` | fan fraction at t+H minus at t |
| `plan_mean_d_supply` | mean supply over (t, t+H] minus at t |
| `plan_mean_d_fan` | mean fan over (t, t+H] minus at t |

Expressed as changes from *t*, so the block carries no absolute setpoint and transfers
between halls with different operating envelopes. The endpoint and the mean are both
included because a setpoint that dips and returns has a different thermal effect from
one that never moved.

Future **workload** is deliberately not given. That is genuinely unknown at prediction
time, and it is what keeps the task non-trivial. It is also what Phase 5's placement
controller supplies for itself: it knows the job it is about to place, so it can ask the
twin about each candidate rack.

Every model gets this block, including the per-rack LSTM, which receives it as constant
extra channels. Giving one model more plant information than another would turn the
comparison into a statement about inputs rather than about spatial structure. The RC
baseline gets the horizon-mean supply temperature in place of the value at *t*, which is
the same information in the form its equation uses.

## What each model sees

| model | own history | neighbours | plant | graph |
|---|---|---|---|---|
| persistence | -- | -- | -- | -- |
| per-rack LSTM | yes, as a sequence | none | yes | -- |
| RC network | current state | via graph edges | yes | conductances |
| LightGBM k=2/4/6 | yes, flattened | k nearest, aggregated | yes | -- |
| graph model | yes | via message passing | yes | learned, directed |

The LSTM has no spatial input at all; that is its role. The gap between it and the
neighbour-aware models measures how much spatial information is worth, before any
question about *how* to use it.

## Splitting and seeds

Split by episode, chronologically, 70/15/15. See `docs/data-generation.md`.

Five seeds per cell, reported as mean and standard deviation, never a single best run.
The seed varies the training subsample as well as model initialisation: without that,
the deterministic models (persistence, RC) would report exactly zero variance for
uninteresting reasons, and the spread for the others would reflect only initialisation
rather than sensitivity to the data they were given.

## A note on scale

The targets are small in absolute terms -- a few tenths of a kelvin at 300 s -- because
the hall's thermal time constant is 60 s and rack-level workload is smooth. Two
consequences worth stating in the paper rather than leaving for a reviewer to find:

1. **Persistence is a strong baseline at short horizons** and should be. A model that
   fails to beat it at 30 s has learned nothing.
2. **Absolute RMSE numbers look small.** The hot-spot RMSE, restricted to the hottest
   10% of samples, is reported alongside because mean error over a whole hall is
   dominated by racks that never approach a limit, and it is the hot racks a controller
   actually needs right.
