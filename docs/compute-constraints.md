# Compute constraints on this machine

The numbers in this repository were produced on an Apple M2 with **8.6 GB of unified
memory**, several gigabytes of which were already swapped during the runs. That is not
a footnote: it changed the Phase 3 configuration, and anyone rerunning this elsewhere
should raise the budget rather than inherit these settings.

## MPS is 70x slower than CPU here

Benchmarked with hall_a's trajectories resident in memory, one training step of the
graph model:

| device | ms/step | s/epoch (2,100 snapshots, batch 16) |
|---|---|---|
| MPS | 7697 | 1010 |
| CPU | 110 | 14.5 |

An earlier micro-benchmark with **no data loaded** put MPS at 186 ms/step, which is why
the first Phase 3 run was launched on MPS and appeared to need ten minutes per epoch.
Once a few hundred megabytes of trajectories are resident, MPS buffer allocation
thrashes against the same unified memory. `pick_device` in
`scripts/phase3_train_gnn.py` therefore defaults to CPU, with `--device mps` available
for a machine that has headroom.

## Activation memory has a cliff, and it is steep

One epoch, hall_a, 60 s horizon, measured directly:

| stride | hidden | heads | batch | train snapshots | s/epoch |
|---|---|---|---|---|---|
| 20 | 64 | 4 | 16 | 2,100 | 279.6 |
| 60 | 48 | 4 | 16 | 700 | 155.2 |
| 60 | 32 | 2 | 16 | 700 | **4.3** |
| 100 | 48 | 4 | 16 | 420 | **5.4** |
| 40 | 48 | 4 | 8 | 1,050 | **8.2** (in isolation) |
| 40 | 48 | 4 | 8 | 1,050 | 110 (in a full run, minutes later) |
| 40 | 32 | 2 | 8 | 1,050 | the configuration actually used |

Thirty-fold differences between neighbouring settings are not compute. The same model
(hidden 48, 4 heads) takes 155 s/epoch at 700 snapshots and 5.4 s/epoch at 420 -- 1.7x
less data for 29x the speed. That is a swap cliff: message passing over 3,496 edges
retains its intermediates for the backward pass, and at batch 16 with hidden 64 the
retained activations are large enough to tip the machine over.

Halving the batch to 8 halves activation memory. That was enough in an isolated
benchmark -- 8.2 s/epoch -- but the *same configuration* took 110 s/epoch in a full run
started minutes later, because the machine's swap state had changed in between. The
cliff is not at a fixed configuration; it moves with whatever else the machine is
doing.

Rather than keep chasing it, Phase 3 runs at the most frugal point measured: hidden 32,
2 attention heads, batch 8. That is a compute decision, not a modelling one, and it is
recorded in `configs/experiment/phase3.yaml` next to the numbers that forced it.

## What this cost the experiment

| | intended | run |
|---|---|---|
| training snapshots | 2,100 (420,000 rack predictions) | 1,050 (210,000) |
| hidden width | 64 | 32 |
| attention heads | 4 | 2 |
| batch size | 16 | 8 |
| epochs | 30 | 15 (patience 4) |
| seeds x horizons | 5 x 3 | 5 x 3 (unchanged) |

The science is unchanged: three message-passing layers, attention over incoming edges,
directed edges, delta targets, hall_a-only normalisation, zero-shot transfer. What
shrank is the training budget, and LightGBM's 400,000-row budget is therefore about
twice the graph model's 210,000. **If the graph model loses to LightGBM, that gap is a
confound and has to be stated; if it wins, it wins from less data and from a model
roughly a quarter the intended width.** Either way the paper should say so rather than
present the comparison as budget-matched.

Reproducing this on a machine with more memory: raise `sample_stride` back to 20,
`batch_size` to 16, `hidden` to 64 and `heads` to 4 in
`configs/experiment/phase3.yaml`, and the comparison becomes budget-matched. **This is
the single highest-value thing to redo before submission** -- the graph model's
headline number is currently produced by a deliberately undersized network.

## Other environment notes

- **LightGBM must run single-threaded and torch must be imported first.** Both ship an
  OpenMP runtime and on macOS arm64 they cannot both hold a thread pool in one process.
  The failure is a segfault with no traceback, which a shell pipeline reports as
  success. See `src/models/lgbm.py`.
- **torch reports 4 threads** on this 8-core machine by default; raising it was not
  investigated, since the binding constraint is memory rather than cores.
