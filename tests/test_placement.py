"""Tests for job placement and rack occupancy.

This is the machinery SustainDC did not have: `dc_gym.step` hardcoded one scalar
utilisation for every rack, so every placement policy produced identical trajectories.
If occupancy accounting is wrong, the utilisation field is wrong and every downstream
temperature is wrong with it.
"""

import numpy as np
import pytest

from src.data.placement import POLICIES, RackOccupancy


@pytest.fixture
def occ():
    return RackOccupancy(n_racks=10, gpus_per_rack=8.0)


def test_starts_empty(occ):
    assert occ.allocated.sum() == 0
    np.testing.assert_allclose(occ.free, 8.0)
    np.testing.assert_allclose(occ.utilisation_pct(), 0.0)


def test_place_and_release_conserve_capacity(occ):
    occ.place(3, 5.0, now_s=0.0, duration_s=100.0)
    assert occ.free[3] == pytest.approx(3.0)
    assert occ.release_until(50.0) == 0        # not expired yet
    assert occ.free[3] == pytest.approx(3.0)
    assert occ.release_until(100.0) == 1
    assert occ.free[3] == pytest.approx(8.0)


def test_utilisation_tracks_allocation(occ):
    occ.place(0, 4.0, 0.0, 100.0)
    assert occ.utilisation_pct()[0] == pytest.approx(50.0)
    occ.place(0, 4.0, 0.0, 100.0)
    assert occ.utilisation_pct()[0] == pytest.approx(100.0)


def test_overcommitting_a_rack_is_rejected(occ):
    occ.place(1, 7.0, 0.0, 100.0)
    with pytest.raises(ValueError, match="cannot take"):
        occ.place(1, 2.0, 0.0, 100.0)


def test_out_of_range_rack_is_rejected(occ):
    with pytest.raises(IndexError):
        occ.place(99, 1.0, 0.0, 10.0)


def test_capacity_is_never_exceeded_under_load(occ):
    """The invariant that matters: no policy may ever oversubscribe a rack."""
    rng = np.random.default_rng(0)
    for name, policy in POLICIES.items():
        o = RackOccupancy(10, 8.0)
        for step in range(300):
            now = step * 30.0
            o.release_until(now)
            for _ in range(4):
                g = float(rng.choice([0.25, 0.5, 1.0, 2.0, 4.0]))
                r = policy(o, g, rng)
                if r >= 0:
                    o.place(r, g, now, float(rng.uniform(60, 600)))
            assert o.allocated.max() <= 8.0 + 1e-9, name
            assert o.allocated.min() >= -1e-9, name


def test_release_frees_everything_eventually(occ):
    for r in range(10):
        occ.place(r, 2.0, 0.0, 100.0)
    occ.release_until(1e9)
    np.testing.assert_allclose(occ.allocated, 0.0, atol=1e-12)
    assert occ.n_live == 0


def test_policies_reject_when_nothing_fits(occ):
    rng = np.random.default_rng(0)
    for r in range(10):
        occ.place(r, 8.0, 0.0, 1000.0)
    for name, policy in POLICIES.items():
        assert policy(occ, 1.0, rng) == -1, name


def test_round_robin_cycles(occ):
    rng = np.random.default_rng(0)
    picks = [POLICIES["round_robin"](occ, 1.0, rng) for _ in range(5)]
    for p in picks:
        occ.place(p, 1.0, 0.0, 1000.0)
    assert picks == [0, 1, 2, 3, 4]


def test_corner_stack_fills_the_lowest_indices(occ):
    """The deliberately bad policy. Racks are indexed row-major, so filling the
    lowest indices piles load into one corner of the hall -- which is how the dataset
    gets the hot states it needs."""
    rng = np.random.default_rng(0)
    for _ in range(8):
        r = POLICIES["corner_stack"](occ, 1.0, rng)
        occ.place(r, 1.0, 0.0, 1000.0)
    assert occ.allocated[0] == pytest.approx(8.0)
    assert occ.allocated[1:].sum() == 0


def test_best_fit_picks_the_tightest_rack(occ):
    rng = np.random.default_rng(0)
    occ.place(0, 2.0, 0.0, 1000.0)      # 6 free
    occ.place(1, 5.0, 0.0, 1000.0)      # 3 free  <- tightest that fits a 3
    occ.place(2, 1.0, 0.0, 1000.0)      # 7 free
    assert POLICIES["best_fit"](occ, 3.0, rng) == 1


def test_policies_spread_differently(occ):
    """random and round_robin should spread; best_fit and corner_stack concentrate.
    If they did not differ, the placement experiment would be measuring nothing."""
    rng = np.random.default_rng(3)
    used = {}
    for name, policy in POLICIES.items():
        o = RackOccupancy(50, 16.0)
        r = np.random.default_rng(3)
        for step in range(200):
            now = step * 30.0
            o.release_until(now)
            for _ in range(3):
                k = policy(o, 1.0, r)
                if k >= 0:
                    o.place(k, 1.0, now, 600.0)
        used[name] = int((o.utilisation_pct() > 0).sum())
    assert used["corner_stack"] < used["random"]
    assert used["best_fit"] < used["round_robin"]


def test_bad_construction_is_rejected():
    with pytest.raises(ValueError, match="at least one rack"):
        RackOccupancy(0, 8.0)
    with pytest.raises(ValueError, match="at least one rack"):
        RackOccupancy(4, 0.0)
