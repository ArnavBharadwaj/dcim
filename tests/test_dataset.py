"""Tests for feature construction and the normaliser.

Two rules from the brief are enforced here rather than by convention:

*   no node identity and no absolute coordinates in the features, or transfer to
    hall_b and hall_c is meaningless;
*   normalisation statistics come from hall_a only, because recomputing them on
    another hall is leakage that inflates the transfer numbers.
"""

import numpy as np
import pytest

from src.data.dataset import (GLOBAL_FEATURES, Normalizer, WindowSpec,
                              feature_names, flatten, neighbour_index,
                              node_features, targets)
from src.twin.geometry import build_hall


@pytest.fixture
def fake_data():
    rng = np.random.default_rng(0)
    t, n = 200, 12
    return {
        "util": rng.uniform(0, 100, (t, n)).astype(np.float32),
        "inlet": rng.uniform(18, 30, (t, n)).astype(np.float32),
        "power": rng.uniform(5e3, 35e3, (t, n)).astype(np.float32),
        "supply_temp": rng.uniform(16, 21, t).astype(np.float32),
        "fan_frac": rng.uniform(0.4, 1.0, t).astype(np.float32),
        "provisioning": rng.uniform(0.6, 2.0, t).astype(np.float32),
    }


def test_feature_count_matches_names(fake_data):
    spec = WindowSpec(lags=4)
    idx = np.arange(10, 100)
    f = node_features(fake_data, idx, spec)
    assert f.shape == (idx.size, 12, len(feature_names(spec.lags)))


def test_feature_count_matches_names_with_neighbours(fake_data, hall_cfg):
    spec = WindowSpec(lags=4)
    geom = build_hall(dict(hall_cfg, rows=4, racks_per_row=3))
    nb = neighbour_index(geom, 4)
    f = node_features(fake_data, np.arange(10, 100), spec, nb)
    assert f.shape[-1] == len(feature_names(spec.lags, 4))


def test_features_carry_no_rack_identity(fake_data):
    """Relabelling the racks must permute the features and change nothing else.

    If a rack index or coordinate leaked in, a model could memorise per-rack offsets
    and transfer to another hall would measure nothing.
    """
    spec = WindowSpec(lags=4)
    idx = np.arange(10, 60)
    base = node_features(fake_data, idx, spec)

    perm = np.random.default_rng(1).permutation(12)
    shuffled = {k: (v[:, perm] if v.ndim == 2 else v) for k, v in fake_data.items()}
    after = node_features(shuffled, idx, spec)
    np.testing.assert_allclose(after, base[:, perm], rtol=1e-6)


def test_features_are_invariant_to_hall_size(fake_data):
    """The per-node feature block must not encode how many racks there are."""
    spec = WindowSpec(lags=3)
    idx = np.arange(5, 40)
    full = node_features(fake_data, idx, spec)
    subset = {k: (v[:, :6] if v.ndim == 2 else v) for k, v in fake_data.items()}
    np.testing.assert_allclose(node_features(subset, idx, spec), full[:, :6], rtol=1e-6)


def test_global_features_are_broadcast_identically(fake_data):
    spec = WindowSpec(lags=3)
    idx = np.arange(5, 40)
    f = node_features(fake_data, idx, spec)
    names = feature_names(spec.lags)
    for g in GLOBAL_FEATURES:
        col = f[:, :, names.index(g)]
        assert np.allclose(col, col[:, :1]), f"{g} must be the same at every rack"


def test_lag_zero_is_the_current_step(fake_data):
    spec = WindowSpec(lags=4)
    idx = np.arange(10, 50)
    f = node_features(fake_data, idx, spec)
    names = feature_names(spec.lags)
    np.testing.assert_allclose(f[:, :, names.index("util_lag0")],
                               fake_data["util"][idx], rtol=1e-6)
    np.testing.assert_allclose(f[:, :, names.index("inlet_lag1")],
                               fake_data["inlet"][idx - 1], rtol=1e-6)


def test_target_is_a_delta_by_default(fake_data):
    idx = np.arange(10, 50)
    d = targets(fake_data, idx, horizon_steps=10, as_delta=True)
    a = targets(fake_data, idx, horizon_steps=10, as_delta=False)
    np.testing.assert_allclose(d, a - fake_data["inlet"][idx], rtol=1e-6)


def test_normalizer_fits_on_one_hall_and_does_not_move(fake_data):
    """The leakage test the brief asks for: statistics come from hall_a, and using
    the normaliser on another hall must not update them."""
    spec = WindowSpec(lags=3)
    idx = np.arange(5, 100)
    X, y = flatten(node_features(fake_data, idx, spec),
                   targets(fake_data, idx, 10))
    nz = Normalizer.fit(X, y, "hall_a", feature_names(spec.lags))
    before = (nz.mean.copy(), nz.scale.copy(), nz.target_mean, nz.target_scale)

    other = {k: (v * 3.0 + 40.0 if v.ndim == 2 else v * 1.5)
             for k, v in fake_data.items()}
    nz.transform(flatten(node_features(other, idx, spec), targets(other, idx, 10))[0])

    np.testing.assert_array_equal(nz.mean, before[0])
    np.testing.assert_array_equal(nz.scale, before[1])
    assert nz.target_mean == before[2] and nz.target_scale == before[3]
    assert nz.fitted_on == "hall_a"


def test_normalizer_round_trips(fake_data):
    spec = WindowSpec(lags=3)
    idx = np.arange(5, 100)
    X, y = flatten(node_features(fake_data, idx, spec), targets(fake_data, idx, 10))
    nz = Normalizer.fit(X, y, "hall_a", feature_names(spec.lags))
    np.testing.assert_allclose(nz.inverse_target(nz.transform_target(y)), y, rtol=1e-5)
    z = nz.transform(X)
    assert abs(z.mean()) < 1e-5 and abs(z.std() - 1.0) < 0.1


def test_normalizer_survives_a_constant_feature():
    X = np.column_stack([np.ones(50), np.arange(50, dtype=float)])
    nz = Normalizer.fit(X, np.arange(50, dtype=float), "hall_a", ("const", "ramp"))
    assert np.all(np.isfinite(nz.transform(X)))
    assert nz.scale[0] == 1.0


def test_normalizer_rejects_wrong_width(fake_data):
    nz = Normalizer.fit(np.random.rand(20, 4), np.random.rand(20), "hall_a",
                        ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="expected 4 features"):
        nz.transform(np.random.rand(5, 3))


def test_normalizer_saves_and_loads(tmp_path, fake_data):
    nz = Normalizer.fit(np.random.rand(30, 5), np.random.rand(30), "hall_a",
                        tuple("abcde"))
    p = tmp_path / "norm.json"
    nz.save(p)
    back = Normalizer.load(p)
    np.testing.assert_allclose(back.mean, nz.mean)
    assert back.fitted_on == "hall_a" and back.feature_names == nz.feature_names


def test_neighbour_index_excludes_self(hall_cfg):
    geom = build_hall(hall_cfg)
    nb = neighbour_index(geom, 4)
    assert nb.shape == (geom.n_racks, 4)
    for i in range(geom.n_racks):
        assert i not in nb[i]


def test_neighbour_index_is_nearest_first(hall_cfg):
    geom = build_hall(hall_cfg)
    nb = neighbour_index(geom, 3)
    d = np.linalg.norm(geom.centre[0] - geom.centre[nb[0]], axis=1)
    assert np.all(np.diff(d) >= -1e-9)


def test_window_spec_rejects_a_non_integer_horizon():
    with pytest.raises(ValueError, match="whole number"):
        WindowSpec(dt_s=30.0).horizon_steps(45)
