"""Sanity tests for the baseline models.

The brief says to test the data pipeline rather than the model, because the model is
stochastic. These are not accuracy tests: they check contracts that would otherwise
fail silently and produce a plausible-looking but wrong results table.
"""

import numpy as np
import pytest

from src.graph.build import build_graph
from src.models.persistence import Persistence
from src.models.rc import RCNetwork
from src.twin.geometry import build_hall


def test_persistence_predicts_zero_delta():
    p = Persistence()
    np.testing.assert_array_equal(p.predict_delta(np.random.rand(17, 4)), np.zeros(17))


@pytest.fixture
def rc_setup(hall_cfg):
    geom = build_hall(hall_cfg)
    graph = build_graph(geom)
    rng = np.random.default_rng(0)
    s, n = 600, geom.n_racks
    inlet = 20.0 + rng.normal(0, 2.0, (s, n))
    power = rng.uniform(8.0, 34.0, (s, n))
    supply = rng.uniform(16.0, 21.0, s)
    return graph, inlet, power, supply, n


def test_rc_beats_the_zero_predictor_on_its_own_training_data(rc_setup):
    """A fitted linear model must not be worse than predicting zero on data it was
    fitted to. It once was: without an intercept, the large driving terms had to
    cancel exactly and small coefficient errors swamped the target.
    """
    graph, inlet, power, supply, n = rc_setup
    rng = np.random.default_rng(1)
    # A target genuinely explained by the RC form: own power, coupling to the supply,
    # and coupling to neighbours.
    true = 0.004 * power + 0.02 * (supply[:, None] - inlet) + rng.normal(0, 0.01, inlet.shape)
    rc = RCNetwork(graph).fit(inlet, power, supply, true)
    pred = rc.predict_delta(inlet, power, supply)
    rmse_fit = np.sqrt(np.mean((pred - true) ** 2))
    rmse_zero = np.sqrt(np.mean(true ** 2))
    assert rmse_fit < rmse_zero, f"fit {rmse_fit:.4f} vs zero {rmse_zero:.4f}"


def test_rc_recovers_a_known_linear_response(rc_setup):
    """With a noiseless target of exactly the RC form, the fit should be near-exact."""
    graph, inlet, power, supply, n = rc_setup
    true = 0.003 * power + 0.015 * (supply[:, None] - inlet)
    rc = RCNetwork(graph).fit(inlet, power, supply, true)
    pred = rc.predict_delta(inlet, power, supply)
    assert np.sqrt(np.mean((pred - true) ** 2)) < 1e-3


def test_rc_has_one_capacitance_per_rack_and_one_conductance_per_edge(rc_setup):
    graph, inlet, power, supply, n = rc_setup
    rc = RCNetwork(graph).fit(inlet, power, supply, np.zeros_like(inlet))
    assert len(rc.coef_) == n
    # intercept + power + supply coupling + one per incoming rack-to-rack neighbour
    for rack in range(n):
        assert rc.coef_[rack].size == 3 + rc._neighbours[rack].size


def test_rc_rejects_shape_mismatch(rc_setup):
    graph, inlet, power, supply, n = rc_setup
    rc = RCNetwork(graph)
    with pytest.raises(ValueError, match="share shape"):
        rc.fit(inlet, power[:, :-1], supply, inlet)


def test_rc_predict_before_fit_is_an_error(rc_setup):
    graph, inlet, power, supply, n = rc_setup
    with pytest.raises(RuntimeError, match="fit"):
        RCNetwork(graph).predict_delta(inlet, power, supply)


def test_gnn_output_shape_is_hall_independent(hall_cfg):
    """The weights must apply unchanged to a hall with a different rack count, or
    zero-shot transfer is impossible by construction."""
    import torch

    from src.models.gnn import ThermalGNN
    small = build_graph(build_hall(hall_cfg))
    large = build_graph(build_hall(dict(hall_cfg, rows=8, racks_per_row=10)))
    model = ThermalGNN(rack_features=9, crac_features=3, edge_dim=small.edge_dim,
                       hidden=32, layers=2, heads=2)
    for g in (small, large):
        out = model(torch.randn(2, g.n_racks, 9), torch.randn(2, g.n_cracs, 3),
                    torch.from_numpy(g.edge_index), torch.from_numpy(g.edge_attr))
        assert out.shape == (2, g.n_racks)


def test_gnn_mean_aggregation_is_a_distinct_model(hall_cfg):
    """The aggregation ablation must actually change the computation."""
    import torch

    from src.models.gnn import ThermalGNN
    g = build_graph(build_hall(hall_cfg))
    torch.manual_seed(0)
    attn = ThermalGNN(9, 3, g.edge_dim, hidden=32, layers=2, heads=2,
                      aggregation="attention", dropout=0.0).eval()
    torch.manual_seed(0)
    mean = ThermalGNN(9, 3, g.edge_dim, hidden=32, layers=2, heads=2,
                      aggregation="mean", dropout=0.0).eval()
    rx, cx = torch.randn(1, g.n_racks, 9), torch.randn(1, g.n_cracs, 3)
    ei, ea = torch.from_numpy(g.edge_index), torch.from_numpy(g.edge_attr)
    with torch.no_grad():
        assert not torch.allclose(attn(rx, cx, ei, ea), mean(rx, cx, ei, ea))


def test_gnn_without_edge_features_ignores_them(hall_cfg):
    """The remove-edge-features ablation must genuinely drop them, not merely
    down-weight them."""
    import torch

    from src.models.gnn import ThermalGNN
    g = build_graph(build_hall(hall_cfg))
    torch.manual_seed(0)
    m = ThermalGNN(9, 3, g.edge_dim, hidden=32, layers=2, heads=2,
                   use_edge_features=False, dropout=0.0).eval()
    rx, cx = torch.randn(1, g.n_racks, 9), torch.randn(1, g.n_cracs, 3)
    ei = torch.from_numpy(g.edge_index)
    ea = torch.from_numpy(g.edge_attr)
    with torch.no_grad():
        a = m(rx, cx, ei, ea)
        b = m(rx, cx, ei, torch.randn_like(ea) * 10.0)
    torch.testing.assert_close(a, b)
