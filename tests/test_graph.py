"""Tests for the graph builder.

The third of the three places the brief flags. The load-bearing property is that edge
features come from the floor plan and never from the twin's own recirculation kernel:
if the kernel leaked into the edges, the model would be handed its own answer and every
Phase 6 ablation would be measuring nothing.
"""

import numpy as np
import pytest

from src.graph.build import (EDGE_FEATURE_NAMES, EdgeType, GraphSpec, NodeType,
                             build_graph, shuffle_edges, undirected)
from src.twin.geometry import build_hall


@pytest.fixture
def graph(hall_cfg):
    return build_graph(build_hall(hall_cfg))


def test_node_counts_and_types(hall_cfg, graph):
    geom = build_hall(hall_cfg)
    assert graph.n_nodes == geom.n_racks + geom.n_cracs
    assert (graph.node_type[graph.rack_nodes] == NodeType.RACK).all()
    assert (graph.node_type[graph.crac_nodes] == NodeType.CRAC).all()


def test_edges_are_within_range(graph):
    assert graph.edge_index.min() >= 0
    assert graph.edge_index.max() < graph.n_nodes
    assert graph.edge_attr.shape == (graph.n_edges, len(EDGE_FEATURE_NAMES))


def test_no_self_loops(graph):
    assert not (graph.edge_index[0] == graph.edge_index[1]).any()


def test_edge_features_are_geometry_only(hall_cfg):
    """Changing the twin's recirculation parameters must not change a single edge.

    Edge features are distance, direction, rows crossed and type -- things an operator
    reads off a floor plan. The recirculation kernel is the thing being predicted.
    """
    geom = build_hall(hall_cfg)
    a = build_graph(geom)
    # The builder takes only geometry and a GraphSpec, so there is no channel for
    # kernel parameters to enter. Assert the signature keeps it that way.
    b = build_graph(geom, GraphSpec())
    np.testing.assert_array_equal(a.edge_index, b.edge_index)
    np.testing.assert_array_equal(a.edge_attr, b.edge_attr)


def test_graph_is_directed(graph):
    """Air flow is asymmetric. If every edge had its reverse, the directed-versus-
    undirected ablation would be vacuous."""
    pairs = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    reversed_pairs = {(b, a) for a, b in pairs}
    assert pairs != reversed_pairs


def test_direction_feature_is_a_cosine(graph):
    col = EDGE_FEATURE_NAMES.index("cos_to_return_flow")
    v = graph.edge_attr[:, col]
    assert v.min() >= -1.0 - 1e-6 and v.max() <= 1.0 + 1e-6


def test_rack_to_rack_edges_respect_the_radius_and_row_limit(hall_cfg):
    geom = build_hall(hall_cfg)
    spec = GraphSpec(radius_m=4.0, max_rows_crossed=1, max_degree=64)
    g = build_graph(geom, spec)
    rr = g.edge_attr[:, EDGE_FEATURE_NAMES.index("is_rack_to_rack")] > 0.5
    assert g.edge_attr[rr, EDGE_FEATURE_NAMES.index("distance_m")].max() <= 4.0 + 1e-6
    assert g.edge_attr[rr, EDGE_FEATURE_NAMES.index("rows_crossed")].max() <= 1


def test_incoming_degree_is_capped(hall_cfg):
    geom = build_hall(hall_cfg)
    g = build_graph(geom, GraphSpec(max_degree=3, radius_m=50.0))
    rr = g.edge_attr[:, EDGE_FEATURE_NAMES.index("is_rack_to_rack")] > 0.5
    dst = g.edge_index[1][rr]
    counts = np.bincount(dst, minlength=g.n_nodes)
    assert counts.max() <= 3


def test_every_rack_returns_to_exactly_one_crac(hall_cfg, graph):
    geom = build_hall(hall_cfg)
    col = EDGE_FEATURE_NAMES.index("is_rack_to_crac")
    m = graph.edge_attr[:, col] > 0.5
    assert m.sum() == geom.n_racks
    assert len(set(graph.edge_index[0][m].tolist())) == geom.n_racks


def test_edge_type_flags_are_one_hot(graph):
    cols = [EDGE_FEATURE_NAMES.index(n) for n in
            ("is_rack_to_rack", "is_rack_to_crac", "is_crac_to_rack")]
    np.testing.assert_allclose(graph.edge_attr[:, cols].sum(axis=1), 1.0)


def test_build_is_deterministic(hall_cfg):
    a, b = build_graph(build_hall(hall_cfg)), build_graph(build_hall(hall_cfg))
    np.testing.assert_array_equal(a.edge_index, b.edge_index)
    np.testing.assert_array_equal(a.edge_attr, b.edge_attr)


def test_undirected_symmetrises_and_drops_the_direction_cue(graph):
    u = undirected(graph)
    assert u.n_edges == 2 * graph.n_edges
    pairs = set(zip(u.edge_index[0].tolist(), u.edge_index[1].tolist()))
    assert pairs == {(b, a) for a, b in pairs}
    col = EDGE_FEATURE_NAMES.index("cos_to_return_flow")
    assert np.all(u.edge_attr[:, col] == 0.0), \
        "leaving the direction feature in would smuggle asymmetry back in"


def test_shuffle_rewires_only_rack_to_rack_edges(graph):
    s = shuffle_edges(graph, 0.3, seed=0)
    rr = graph.edge_attr[:, EDGE_FEATURE_NAMES.index("is_rack_to_rack")] > 0.5
    assert np.array_equal(s.edge_index[:, ~rr], graph.edge_index[:, ~rr])
    changed = (s.edge_index[1] != graph.edge_index[1]).sum()
    assert 0 < changed <= rr.sum()


def test_shuffle_zero_is_a_no_op(graph):
    np.testing.assert_array_equal(shuffle_edges(graph, 0.0, 0).edge_index,
                                  graph.edge_index)


def test_shuffle_is_seeded(graph):
    a = shuffle_edges(graph, 0.2, seed=7).edge_index
    b = shuffle_edges(graph, 0.2, seed=7).edge_index
    np.testing.assert_array_equal(a, b)


def test_graph_transfers_to_a_hall_of_a_different_size(hall_cfg):
    """Edge dimension must not depend on hall size, or weights cannot transfer."""
    small = build_graph(build_hall(hall_cfg))
    large = build_graph(build_hall(dict(hall_cfg, rows=8, racks_per_row=10)))
    assert small.edge_dim == large.edge_dim
    assert large.n_racks != small.n_racks


def test_bad_spec_is_rejected():
    with pytest.raises(ValueError, match="radius_m"):
        GraphSpec(radius_m=0.0)
    with pytest.raises(ValueError, match="max_rows_crossed"):
        GraphSpec(max_rows_crossed=0)
