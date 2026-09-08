# Phase 2: baselines, and what they say about the graph model's prospects

The brief puts the baselines before the graph model deliberately: *"If gradient
boosting is already close to the ceiling, I need to know now, so I can reshape the
paper around transfer."* This is that answer.

All numbers: hall_a, test fold (episodes 85-99), 5 seeds, mean ± sd. Test RMSE of the
temperature delta, in kelvin. Built from `results/runs.csv`.

## Accuracy

| model | 30 s | 60 s | 300 s |
|---|---|---|---|
| persistence | 0.1057 ± 0.0000 | 0.1908 ± 0.0000 | 0.6367 ± 0.0000 |
| RC network | 0.1682 ± 0.1393 | 0.1978 ± 0.0499 | 0.4833 ± 0.0411 |
| per-rack LSTM | **0.0272** ± 0.0017 | **0.0598** ± 0.0057 | 0.2245 ± 0.0079 |
| LightGBM k=2 | 0.0432 ± 0.0012 | 0.0609 ± 0.0023 | **0.1884** ± 0.0017 |
| LightGBM k=4 | 0.0433 ± 0.0012 | 0.0606 ± 0.0025 | **0.1884** ± 0.0024 |
| LightGBM k=6 | 0.0432 ± 0.0012 | 0.0610 ± 0.0023 | 0.1886 ± 0.0017 |

Hot-spot RMSE, hottest 10% of test samples:

| model | 30 s | 60 s | 300 s |
|---|---|---|---|
| persistence | 0.0900 | 0.2183 | 0.8034 |
| RC network | 0.1325 ± 0.0589 | 0.2243 ± 0.0131 | 0.7068 ± 0.0070 |
| per-rack LSTM | **0.0242** ± 0.0018 | 0.0577 ± 0.0047 | 0.2589 ± 0.0177 |
| LightGBM k=4 | 0.0252 ± 0.0007 | **0.0549** ± 0.0022 | **0.2616** ± 0.0089 |

## The finding that matters

**k makes no difference.** LightGBM with the 2, 4 and 6 nearest racks' load and
temperature scores identically to three decimal places at every horizon, well inside
the seed-to-seed spread. Adding four more neighbours' worth of context buys nothing.

That is not a bug. Measured on the same features:

| | linear R² on the 300 s target |
|---|---|
| own-rack features only | 0.8251 |
| own-rack + all neighbour features | 0.8265 |
| neighbour features only | 0.0280 |

and the reason:

> **corr(own inlet temperature, mean inlet temperature of the 6 nearest racks) = 0.9953**

Neighbouring racks share almost exactly the same recirculation environment. The
coupling matrix `D` is dense and diffuse -- each rack's heat is spread broadly across
the hall rather than dumped on its immediate neighbours -- so a rack's own temperature
already encodes what its neighbours are experiencing. The neighbour features are
nearly collinear with it and carry 2.8% of the target variance on their own.

### What this means for the paper

The honest reading is that **on this twin, spatial context adds very little to
single-rack accuracy**, and a graph model has correspondingly little room to win the
accuracy experiment on hall_a. Three consequences:

1. The accuracy table should not be the paper's headline. If the graph model beats
   LightGBM it will be by a margin comparable to the seed spread.
2. **Transfer is the experiment with room to move.** LightGBM's k-nearest features are
   hall-specific by construction and the RC network's parameters are per-rack; neither
   transfers to a hall with different racks without refitting. A graph model carrying
   geometry in its edges is the only one of these that can be applied zero-shot. Phase
   3 tests exactly that.
3. The Phase 6 shuffle ablation is now *expected* to come back close to null, and the
   brief already says to publish that whatever it says. Given the 0.995 correlation, a
   shuffled graph destroying almost no accuracy would be consistent with everything
   measured here rather than a surprise.

This is the outcome the brief's ordering was designed to catch, and it was caught in
Phase 2 rather than after six weeks on the model.

## Other observations

**The LSTM wins at short horizons, LightGBM at long ones.** The LSTM has no spatial
input at all and still beats every neighbour-aware model at 30 s (0.0272 against
0.0432). At 30 s the target is dominated by the first-order thermal lag, which is a
smooth function of the rack's own recent trajectory -- exactly what a sequence model
is for. By 300 s the response has largely settled and what matters is the plant's
steady state, where gradient boosting's nonlinearity in the setpoint features wins
(0.1884 against 0.2245).

**Persistence is beaten by a wide margin at every horizon**, 3.9× at 30 s and 3.4× at
300 s. This is the sanity check that the task is learnable at all -- and it is only
true because the control plan is an input. Before that fix, LightGBM scored *worse*
than persistence at 300 s. See [prediction-task.md](prediction-task.md).

**The RC network is the weakest fitted model and the least stable.** Its seed spread at
30 s was ±0.139 K against a mean of 0.168 -- larger than its own mean. Cause: the same
0.995 correlation between neighbouring racks makes each per-rack design matrix severely
collinear, and the ridge term was far too small to control it. Fixed by normalising the
gram by the sample count so the ridge is scale-free, and raising it to 1e-3. The numbers
above are from before that fix and are superseded by the rerun recorded in
`results/runs.csv`.

## Threats to these numbers

- **One hall.** Everything here is hall_a. hall_c is uncontained and has a 9.7 K inlet
  spread against hall_a's 2.5 K; the neighbour redundancy may well be weaker there, and
  that is worth measuring before generalising the claim.
- **The 0.995 correlation is a property of our recirculation kernel**, specifically of
  how broadly `decay_length_m` spreads each rack's exhaust. A hall with tighter, more
  local recirculation would show more spatial signal. This is a real limitation of a
  synthetic twin and belongs in the paper's limitations section, not buried.
- **LightGBM ran single-threaded** for the OpenMP reason in `src/models/lgbm.py`. That
  affects wall-clock only, not the fits.
