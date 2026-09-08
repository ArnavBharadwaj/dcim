"""Tests for workload trace loading and the bootstrap replay."""

import numpy as np
import pandas as pd
import pytest

from src.data.trace import (arrival_rate_for_utilisation, bootstrap_stream,
                            hourly_arrival_profile, load_trace,
                            mean_gpu_hours_per_job)


@pytest.fixture
def fake_trace():
    rng = np.random.default_rng(0)
    n = 5000
    return pd.DataFrame({
        "num_gpu": rng.choice([0.0, 0.25, 0.5, 1.0, 2.0], n),
        "num_cpu": rng.integers(1, 64, n).astype(float),
        "duration": rng.integers(60, 20000, n).astype(float),
        "submit_time": np.sort(rng.integers(0, 3600 * 240, n)).astype(float),
    })


def test_arrival_profile_is_normalised(fake_trace):
    p = hourly_arrival_profile(fake_trace)
    assert p.size == 24
    assert p.mean() == pytest.approx(1.0)


def test_bootstrap_covers_the_requested_span(fake_trace):
    s = bootstrap_stream(fake_trace, hours=10, jobs_per_hour=200, seed=0)
    assert len(s) > 0
    assert s.arrival_s.min() >= 0
    assert s.arrival_s.max() <= 10 * 3600
    assert np.all(np.diff(s.arrival_s) >= 0), "arrivals must be sorted"


def test_bootstrap_is_seeded(fake_trace):
    a = bootstrap_stream(fake_trace, 5, 200, seed=3)
    b = bootstrap_stream(fake_trace, 5, 200, seed=3)
    np.testing.assert_array_equal(a.arrival_s, b.arrival_s)
    np.testing.assert_array_equal(a.duration_s, b.duration_s)
    c = bootstrap_stream(fake_trace, 5, 200, seed=4)
    assert not np.array_equal(a.arrival_s, c.arrival_s)


def test_bootstrap_preserves_the_job_distribution(fake_trace):
    """Attributes are drawn from real rows, so the joint distribution of GPU
    request and duration is the trace's own, not a fitted approximation."""
    s = bootstrap_stream(fake_trace, 400, 500, seed=0)
    assert set(np.unique(s.gpus)) <= set(np.unique(fake_trace.num_gpu))
    assert s.duration_s.mean() == pytest.approx(fake_trace.duration.mean(), rel=0.1)


def test_duration_truncation_is_applied(fake_trace):
    cap = 3600.0
    s = bootstrap_stream(fake_trace, 20, 300, seed=0, max_duration_s=cap)
    assert s.duration_s.max() <= cap


def test_arrival_rate_hits_the_target_utilisation(fake_trace):
    """Little's law inversion: the resulting stream must actually occupy about the
    requested fraction of the slots. This is the knob the Phase 1 thermal gate turns."""
    slots = 1000.0
    for target in (0.3, 0.6):
        rate = arrival_rate_for_utilisation(fake_trace, target, slots, seed=0,
                                            max_duration_s=3 * 3600)
        s = bootstrap_stream(fake_trace, 200, rate, seed=1, max_duration_s=3 * 3600)
        busy = len(s) / 200.0 * mean_gpu_hours_per_job(s)     # expected slots in use
        assert busy / slots == pytest.approx(target, rel=0.15)


def test_higher_target_needs_a_higher_rate(fake_trace):
    lo = arrival_rate_for_utilisation(fake_trace, 0.3, 1000.0, seed=0)
    hi = arrival_rate_for_utilisation(fake_trace, 0.8, 1000.0, seed=0)
    assert hi > lo


def test_window_reindexes_to_zero(fake_trace):
    s = bootstrap_stream(fake_trace, 20, 300, seed=0)
    w = s.window(3600.0, 7200.0)
    assert w.arrival_s.min() >= 0
    assert w.arrival_s.max() < 3600.0


def test_invalid_arguments_are_rejected(fake_trace):
    with pytest.raises(ValueError, match="must be positive"):
        bootstrap_stream(fake_trace, 0, 100, seed=0)
    with pytest.raises(ValueError, match="target_util"):
        arrival_rate_for_utilisation(fake_trace, 0.0, 100.0, seed=0)


def test_missing_trace_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_trace"):
        load_trace(tmp_path / "nope.csv")
