"""Tests for the liquid-cooled twin.

The load-bearing one is `test_racks_on_different_cdus_are_thermally_independent`: it
pins the property the whole liquid-cooling argument rests on, that coupling follows the
hydraulics and not the floor plan.
"""

import numpy as np
import pytest

from src.graph.build import (LIQUID_EDGE_FEATURE_NAMES, LiquidNodeType,
                             build_liquid_graph)
from src.twin.geometry import build_hall
from src.twin.liquid import (CP_COOLANT, LiquidParams, build_topology,
                             coupling_blocks, solve_coolant)
from src.twin.liquid_thermal import LiquidCooledTwin
from src.twin.power import PowerParams


@pytest.fixture
def liquid_hall():
    cfg = {
        "name": "test_liquid", "rows": 4, "racks_per_row": 8,
        "orientation_pattern": "paired", "rack_width_m": 0.6,
        "rack_depth_m": 1.2, "aisle_width_m": 1.2,
        "crac_units": [{"aisle": 1, "end": "left"}],
    }
    geom = build_hall(cfg)
    cdus = [
        {"capacity_kw": 700, "branches": [{"row": 0, "feed": "left"},
                                          {"row": 1, "feed": "left"}]},
        {"capacity_kw": 700, "branches": [{"row": 2, "feed": "right"},
                                          {"row": 3, "feed": "right"}]},
    ]
    return geom, build_topology(geom, cdus, 700.0)


# ------------------------------------------------------------------ topology

def test_every_rack_is_served_exactly_once(liquid_hall):
    geom, topo = liquid_hall
    assert topo.n_racks == geom.n_racks
    assert np.all(topo.cdu_of_rack >= 0)
    assert np.all(topo.branch_of_rack >= 0)
    assert topo.n_branches == 4
    counts = np.bincount(topo.branch_of_rack)
    assert np.all(counts == geom.racks_per_row)


def test_unassigned_rows_are_rejected(liquid_hall):
    geom, _ = liquid_hall
    with pytest.raises(ValueError, match="no coolant branch"):
        build_topology(geom, [{"branches": [{"row": 0, "feed": "left"}]}], 700.0)


def test_double_assigned_rows_are_rejected(liquid_hall):
    geom, _ = liquid_hall
    cdus = [{"branches": [{"row": r, "feed": "left"} for r in range(4)]},
            {"branches": [{"row": 0, "feed": "left"}]}]
    with pytest.raises(ValueError, match="more than one CDU"):
        build_topology(geom, cdus, 700.0)


def test_feed_end_sets_the_manifold_direction(liquid_hall):
    geom, topo = liquid_hall
    left = topo.racks_on_branch(0)      # row 0, fed from the left
    right = topo.racks_on_branch(2)     # row 2, fed from the right
    assert geom.col[left[0]] == 0, "left-fed branch starts at column 0"
    assert geom.col[right[0]] == geom.racks_per_row - 1, "right-fed starts at the far end"
    assert topo.manifold_dist_m[left[0]] == pytest.approx(0.0)
    assert topo.manifold_dist_m[right[0]] == pytest.approx(0.0)


def test_manifold_distance_increases_along_a_branch(liquid_hall):
    _, topo = liquid_hall
    for b in range(topo.n_branches):
        d = topo.manifold_dist_m[topo.racks_on_branch(b)]
        assert np.all(np.diff(d) > 0)


# ------------------------------------------------------------------- coolant

def test_heat_split_conserves_rack_power(liquid_hall):
    _, topo = liquid_hall
    p = np.full(topo.n_racks, 70e3)
    s = solve_coolant(topo, LiquidParams(), p, facility_water_c=30.0)
    np.testing.assert_allclose(s.liquid_power_w + s.air_power_w, p, rtol=1e-12)
    np.testing.assert_allclose(s.liquid_power_w, 0.80 * p, rtol=1e-12)


def test_rack_temperature_rise_matches_its_own_heat_and_flow(liquid_hall):
    _, topo = liquid_hall
    p = np.full(topo.n_racks, 70e3)
    s = solve_coolant(topo, LiquidParams(), p, facility_water_c=30.0)
    np.testing.assert_allclose(s.return_temp_c - s.supply_temp_c,
                               s.liquid_power_w / (s.flow_kgs * CP_COOLANT),
                               rtol=1e-12)


def test_cdu_load_is_the_sum_of_its_racks(liquid_hall):
    _, topo = liquid_hall
    rng = np.random.default_rng(0)
    p = rng.uniform(20e3, 80e3, topo.n_racks)
    s = solve_coolant(topo, LiquidParams(), p, facility_water_c=30.0)
    for c in range(topo.n_cdus):
        expected = s.liquid_power_w[topo.racks_on_cdu(c)].sum() / 1e3
        assert s.cdu_load_kw[c] == pytest.approx(expected)


def test_approach_temperature_grows_with_duty(liquid_hall):
    _, topo = liquid_hall
    light = solve_coolant(topo, LiquidParams(), np.full(topo.n_racks, 20e3), 30.0)
    heavy = solve_coolant(topo, LiquidParams(), np.full(topo.n_racks, 80e3), 30.0)
    assert np.all(heavy.cdu_approach_k > light.cdu_approach_k)
    assert np.all(heavy.supply_temp_c > light.supply_temp_c)


def test_slower_pumps_raise_case_temperature(liquid_hall):
    """Pump speed must move what a chip cares about, not only the return temperature.

    A cold plate's convective coefficient falls with flow, so the case-to-coolant rise
    grows. Without that, pump speed would be a pure energy knob and the cooling
    controller would have one real lever instead of two.
    """
    _, topo = liquid_hall
    p = np.full(topo.n_racks, 70e3)
    fast = solve_coolant(topo, LiquidParams(), p, 30.0, pump_frac=1.0)
    slow = solve_coolant(topo, LiquidParams(), p, 30.0, pump_frac=0.6)
    assert np.all(slow.case_temp_c > fast.case_temp_c)

    # Slowing the pumps also warms downstream supply, because less flow carries the
    # same upstream heat and the cross-talk term grows. The first rack on a branch has
    # nothing upstream of it, so its supply is the one that must not move.
    firsts = [int(topo.racks_on_branch(b)[0]) for b in range(topo.n_branches)]
    np.testing.assert_allclose(slow.supply_temp_c[firsts],
                               fast.supply_temp_c[firsts], rtol=1e-9)
    # The cold plate dominates: case rises by more than supply does everywhere.
    assert np.all((slow.case_temp_c - fast.case_temp_c)
                  > (slow.supply_temp_c - fast.supply_temp_c))


def test_derating_a_cdu_only_warms_its_own_racks(liquid_hall):
    """The Phase 4 drift hook. A failed unit must not warm the other loop."""
    _, topo = liquid_hall
    p = np.full(topo.n_racks, 70e3)
    base = solve_coolant(topo, LiquidParams(), p, 30.0)
    hurt = solve_coolant(topo, LiquidParams(), p, 30.0,
                         cdu_derate=np.array([0.6, 1.0]))
    on0 = topo.racks_on_cdu(0)
    on1 = topo.racks_on_cdu(1)
    assert np.all(hurt.supply_temp_c[on0] > base.supply_temp_c[on0])
    np.testing.assert_allclose(hurt.supply_temp_c[on1], base.supply_temp_c[on1])


def test_upstream_racks_warm_downstream_ones_but_not_the_reverse(liquid_hall):
    """Manifold cross-talk is directed: coolant reaching a rack has passed the racks
    before it on the branch, never the ones after."""
    _, topo = liquid_hall
    params = LiquidParams()
    order = topo.racks_on_branch(0)
    first, last = int(order[0]), int(order[-1])
    base_p = np.full(topo.n_racks, 40e3)

    hot_first = base_p.copy(); hot_first[first] = 80e3
    hot_last = base_p.copy(); hot_last[last] = 80e3
    base = solve_coolant(topo, params, base_p, 30.0)
    a = solve_coolant(topo, params, hot_first, 30.0)
    b = solve_coolant(topo, params, hot_last, 30.0)

    # Loading the first rack warms the last one's supply beyond the shared CDU effect.
    gain_down = a.supply_temp_c[last] - base.supply_temp_c[last]
    gain_up = b.supply_temp_c[first] - base.supply_temp_c[first]
    assert gain_down > gain_up, (
        f"downstream gain {gain_down:.5f} K should exceed upstream {gain_up:.5f} K")


def test_racks_on_different_cdus_are_thermally_independent(liquid_hall):
    """The property the whole liquid-cooling argument rests on.

    Two racks on different coolant loops do not influence each other's coolant supply,
    however close they stand on the floor. This is what a Euclidean k-nearest-neighbour
    feature set cannot represent: it would select physically adjacent racks that carry
    no information at all.
    """
    _, topo = liquid_hall
    params = LiquidParams()
    p = np.full(topo.n_racks, 50e3)
    base = solve_coolant(topo, params, p, 30.0).supply_temp_c

    src = int(topo.racks_on_cdu(0)[0])
    bumped = p.copy(); bumped[src] += 20e3
    after = solve_coolant(topo, params, bumped, 30.0).supply_temp_c

    other = topo.racks_on_cdu(1)
    np.testing.assert_allclose(after[other], base[other], atol=1e-12)
    assert np.any(after[topo.racks_on_cdu(0)] > base[topo.racks_on_cdu(0)])


def test_coupling_blocks_match_the_branch_assignment(liquid_hall):
    _, topo = liquid_hall
    blocks = coupling_blocks(topo)
    assert blocks.shape == (topo.n_racks, topo.n_racks)
    assert np.all(np.diag(blocks))
    i, j = int(topo.racks_on_branch(0)[0]), int(topo.racks_on_branch(1)[0])
    assert not blocks[i, j]


def test_params_validate():
    with pytest.raises(ValueError, match="liquid_fraction"):
        LiquidParams(liquid_fraction=0.0)
    with pytest.raises(ValueError, match="manifold_crosstalk"):
        LiquidParams(manifold_crosstalk=1.0)
    with pytest.raises(ValueError, match="cdu_capacity_kw"):
        LiquidParams(cdu_capacity_kw=0.0)


# --------------------------------------------------------------------- twin

def test_twin_reaches_a_steady_state_and_stays(liquid_hall):
    geom, topo = liquid_hall
    tw = LiquidCooledTwin(geom, topo, power=PowerParams(
        servers_per_rack=8, server_full_load_w=9500.0, server_idle_w=1900.0,
        fan_ref_power_w=120.0))
    u = np.full(geom.n_racks, 80.0)
    tw.reset(u, facility_water_c=32.0, air_supply_c=22.0)
    first = tw.step(u, 32.0, 22.0, dt_s=1e-6).case_temp_c.copy()
    later = tw.step(u, 32.0, 22.0, dt_s=3600.0).case_temp_c
    np.testing.assert_allclose(later, first, atol=2e-3)


def test_twin_rejects_a_wrong_shaped_utilisation(liquid_hall):
    geom, topo = liquid_hall
    tw = LiquidCooledTwin(geom, topo)
    tw.reset(np.full(geom.n_racks, 50.0), 32.0)
    with pytest.raises(ValueError, match="must have shape"):
        tw.step(np.full(3, 50.0), 32.0)


def test_twin_step_before_reset_is_an_error(liquid_hall):
    geom, topo = liquid_hall
    with pytest.raises(RuntimeError, match="reset"):
        LiquidCooledTwin(geom, topo).step(np.full(geom.n_racks, 50.0), 32.0)


def test_warmer_facility_water_warms_the_chips(liquid_hall):
    geom, topo = liquid_hall
    tw = LiquidCooledTwin(geom, topo)
    u = np.full(geom.n_racks, 80.0)
    cool = tw.reset(u, facility_water_c=26.0).case_temp_c.max()
    warm = tw.reset(u, facility_water_c=42.0).case_temp_c.max()
    assert warm > cool + 10.0


# -------------------------------------------------------------------- graph

def test_liquid_graph_has_both_coolant_and_air_edges(liquid_hall):
    geom, topo = liquid_hall
    g = build_liquid_graph(geom, topo)
    names = LIQUID_EDGE_FEATURE_NAMES
    for kind in ("is_manifold_downstream", "is_manifold_upstream",
                 "is_rack_to_cdu", "is_cdu_to_rack", "is_air_recirc"):
        assert (g.edge_attr[:, names.index(kind)] > 0.5).sum() > 0, kind
    assert g.n_nodes == geom.n_racks + topo.n_cdus
    assert (g.node_type[g.crac_nodes] == LiquidNodeType.CDU).all()


def test_manifold_edges_only_join_racks_on_the_same_branch(liquid_hall):
    geom, topo = liquid_hall
    g = build_liquid_graph(geom, topo)
    names = LIQUID_EDGE_FEATURE_NAMES
    man = ((g.edge_attr[:, names.index("is_manifold_downstream")] > 0.5)
           | (g.edge_attr[:, names.index("is_manifold_upstream")] > 0.5))
    s, d = g.edge_index[0][man], g.edge_index[1][man]
    assert np.all(topo.branch_of_rack[s] == topo.branch_of_rack[d])


def test_manifold_direction_is_encoded_and_signed(liquid_hall):
    geom, topo = liquid_hall
    g = build_liquid_graph(geom, topo)
    names = LIQUID_EDGE_FEATURE_NAMES
    off = g.edge_attr[:, names.index("manifold_offset_m")]
    down = g.edge_attr[:, names.index("is_manifold_downstream")] > 0.5
    up = g.edge_attr[:, names.index("is_manifold_upstream")] > 0.5
    assert np.all(off[down] > 0)
    assert np.all(off[up] < 0)


def test_liquid_graph_is_deterministic(liquid_hall):
    geom, topo = liquid_hall
    a, b = build_liquid_graph(geom, topo), build_liquid_graph(geom, topo)
    np.testing.assert_array_equal(a.edge_index, b.edge_index)
    np.testing.assert_array_equal(a.edge_attr, b.edge_attr)
