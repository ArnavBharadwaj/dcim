"""Tests for the train/test splitter.

The brief names the splitter as one of three places where a silent bug destroys the
results, and says why: consecutive 30 s timesteps are near-identical, so a random split
puts a sample's own near-duplicate on the other side of the boundary and every model
looks excellent. These tests hold the split strictly chronological and hold every input
window inside a single episode.
"""

import numpy as np
import pytest

from src.data.split import Split, fold_mask, sample_indices, time_split


@pytest.fixture
def episode_id():
    """20 episodes of 100 steps each."""
    return np.repeat(np.arange(20), 100)


def test_split_is_chronological(episode_id):
    s = time_split(20, train_frac=0.7, val_frac=0.15)
    assert s.train[1] <= s.val[0] <= s.val[1] <= s.test[0]
    assert s.train == (0, 14) and s.val == (14, 17) and s.test == (17, 20)


def test_folds_do_not_overlap(episode_id):
    s = time_split(20)
    masks = {f: fold_mask(episode_id, s, f) for f in ("train", "val", "test")}
    assert not (masks["train"] & masks["val"]).any()
    assert not (masks["val"] & masks["test"]).any()
    assert not (masks["train"] & masks["test"]).any()


def test_every_train_timestep_precedes_every_test_timestep(episode_id):
    """The property a random split would violate."""
    s = time_split(20)
    t = np.flatnonzero(fold_mask(episode_id, s, "train"))
    te = np.flatnonzero(fold_mask(episode_id, s, "test"))
    assert t.max() < te.min()


def test_samples_never_cross_an_episode_boundary(episode_id):
    """A window's history and its target must come from the same episode, so they
    share a placement policy and a cooling schedule."""
    s = time_split(20)
    lags, horizon = 6, 10
    idx = sample_indices(episode_id, s, "train", lags, horizon)
    assert np.all(episode_id[idx - (lags - 1)] == episode_id[idx])
    assert np.all(episode_id[idx + horizon] == episode_id[idx])


def test_samples_stay_inside_their_fold(episode_id):
    s = time_split(20)
    idx = sample_indices(episode_id, s, "test", lags=6, horizon_steps=10)
    assert set(np.unique(episode_id[idx]).tolist()) <= set(range(*s.test))


def test_horizon_and_lag_windows_are_in_range(episode_id):
    s = time_split(20)
    lags, horizon = 8, 20
    idx = sample_indices(episode_id, s, "train", lags, horizon)
    assert idx.min() - (lags - 1) >= 0
    assert idx.max() + horizon < episode_id.size


def test_stride_subsamples_without_reordering(episode_id):
    s = time_split(20)
    full = sample_indices(episode_id, s, "train", 6, 10, stride=1)
    strided = sample_indices(episode_id, s, "train", 6, 10, stride=5)
    assert strided.size < full.size
    assert np.all(np.diff(strided) > 0)
    assert set(strided.tolist()) <= set(full.tolist())


def test_overlapping_splits_are_rejected():
    with pytest.raises(ValueError, match="ordered in time"):
        Split(train=(0, 10), val=(8, 12), test=(12, 15))


def test_empty_split_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        Split(train=(0, 0), val=(1, 2), test=(2, 3))


def test_split_needs_enough_episodes():
    with pytest.raises(ValueError, match="at least 3 episodes"):
        time_split(2)


def test_episodes_shorter_than_the_window_are_skipped():
    """Short episodes must be dropped, not silently produce out-of-range windows."""
    ep = np.repeat(np.arange(6), 5)          # 5 steps per episode
    s = time_split(6, train_frac=0.5, val_frac=0.2)
    with pytest.raises(ValueError, match="no usable samples"):
        sample_indices(ep, s, "train", lags=6, horizon_steps=10)
