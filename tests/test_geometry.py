"""Tests for the hall geometry builder.

The brief calls out the graph builder as one of the places where a silent bug would
destroy the results. Geometry is upstream of the graph, fully deterministic, and
therefore worth pinning hard.
"""

import numpy as np
import pytest

from src.twin.geometry import build_hall, rows_crossed


def test_rack_count_and_row_major_indexing(hall_cfg):
    g = build_hall(hall_cfg)
    assert g.n_racks == hall_cfg["rows"] * hall_cfg["racks_per_row"]
    # Flat id i must decompose as (i // C, i % C).
    expected_row = np.arange(g.n_racks) // g.racks_per_row
    expected_col = np.arange(g.n_racks) % g.racks_per_row
    np.testing.assert_array_equal(g.row, expected_row)
    np.testing.assert_array_equal(g.col, expected_col)


def test_paired_rows_face_each_other_across_a_cold_aisle(hall_cfg):
    g = build_hall(hall_cfg)
    for p in range(g.n_rows // 2):
        a, b = 2 * p, 2 * p + 1
        inlet_a = g.inlet_aisle[g.row == a]
        inlet_b = g.inlet_aisle[g.row == b]
        assert len(set(inlet_a.tolist())) == 1
        # Rows 2p and 2p+1 draw from the same aisle, and it is a cold one.
        assert inlet_a[0] == inlet_b[0]
        idx = list(g.aisle_index).index(inlet_a[0])
        assert g.aisle_type[idx] == "cold"


def test_paired_adjacent_rows_share_a_hot_exhaust_aisle(hall_cfg):
    g = build_hall(hall_cfg)
    for p in range((g.n_rows - 1) // 2):
        a, b = 2 * p + 1, 2 * p + 2
        ex_a = g.exhaust_aisle[g.row == a][0]
        ex_b = g.exhaust_aisle[g.row == b][0]
        assert ex_a == ex_b
        idx = list(g.aisle_index).index(ex_a)
        assert g.aisle_type[idx] == "hot"


def test_inlet_and_exhaust_are_opposite_faces(hall_cfg):
    g = build_hall(hall_cfg)
    sep = np.linalg.norm(g.inlet_xy - g.exhaust_xy, axis=1)
    np.testing.assert_allclose(sep, hall_cfg["rack_depth_m"])
    # Same x: air enters and leaves across the depth of the rack, not its width.
    np.testing.assert_allclose(g.inlet_xy[:, 0], g.exhaust_xy[:, 0])


def test_end_aisles_are_hot_in_a_paired_layout(hall_cfg):
    g = build_hall(hall_cfg)
    assert g.aisle_type[0] == "hot"          # aisle -1, before row 0
    assert g.aisle_type[-1] == "hot"         # aisle R-1, after the last row
    assert g.aisle_type == ("hot", "cold", "hot", "cold", "hot", "cold", "hot")


def test_uniform_layout_has_only_mixed_aisles(hall_cfg):
    g = build_hall(dict(hall_cfg, orientation_pattern="uniform",
                        crac_units=[{"aisle": 2, "end": "left"}]))
    assert set(g.aisle_type) == {"mixed"}
    assert np.all(g.facing == 1)


def test_rows_crossed_is_at_least_one_and_odd_when_paired(hall_cfg):
    g = build_hall(hall_cfg)
    rc = rows_crossed(g)
    assert rc.min() >= 1, "air always crosses at least one row to recirculate"
    assert np.all(rc % 2 == 1), "exhaust aisles are odd and inlet aisles even"


def test_crac_placement_rejects_a_cold_aisle(hall_cfg):
    with pytest.raises(ValueError, match="cold aisle"):
        build_hall(dict(hall_cfg, crac_units=[{"aisle": 2, "end": "left"}]))


def test_crac_placement_rejects_a_nonexistent_aisle(hall_cfg):
    with pytest.raises(ValueError, match="does not exist"):
        build_hall(dict(hall_cfg, crac_units=[{"aisle": 99, "end": "left"}]))


def test_crac_shares_normalise(hall_cfg):
    g = build_hall(dict(hall_cfg, crac_units=[
        {"aisle": 1, "end": "left", "flow_share": 3.0},
        {"aisle": 3, "end": "right", "flow_share": 1.0}]))
    np.testing.assert_allclose(g.crac_flow_share.sum(), 1.0)
    np.testing.assert_allclose(g.crac_flow_share, [0.75, 0.25])


def test_return_directions_are_unit_vectors(hall_cfg):
    g = build_hall(hall_cfg)
    np.testing.assert_allclose(np.linalg.norm(g.return_dir, axis=1), 1.0)


def test_nearest_crac_is_actually_nearest(hall_cfg):
    g = build_hall(hall_cfg)
    d = np.linalg.norm(g.exhaust_xy[:, None, :] - g.crac_xy[None, :, :], axis=2)
    np.testing.assert_array_equal(g.nearest_crac, d.argmin(axis=1))
    np.testing.assert_allclose(g.dist_to_crac, d.min(axis=1))


def test_build_is_deterministic(hall_cfg):
    a, b = build_hall(hall_cfg), build_hall(hall_cfg)
    for field in ("row", "col", "facing", "centre", "inlet_xy", "exhaust_xy",
                  "inlet_aisle", "exhaust_aisle", "crac_xy", "dist_to_crac",
                  "return_dir"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def test_degenerate_halls_are_rejected(hall_cfg):
    with pytest.raises(ValueError, match="at least 2 rows"):
        build_hall(dict(hall_cfg, rows=1))
    with pytest.raises(ValueError, match="unknown orientation_pattern"):
        build_hall(dict(hall_cfg, orientation_pattern="spiral"))
