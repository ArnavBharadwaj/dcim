"""Tests for the rack power and airflow model."""

import numpy as np
import pytest

from src.twin.power import (C_AIR, RHO_AIR, PowerParams, crac_supply_flow_m3s,
                            fan_ratio, rack_power)


def test_power_is_monotone_in_utilisation():
    p = PowerParams()
    t = np.full(5, 22.0)
    lo = rack_power(np.full(5, 10.0), t, p).total_power_w
    hi = rack_power(np.full(5, 90.0), t, p).total_power_w
    assert np.all(hi > lo)


def test_power_is_monotone_in_inlet_temperature():
    """The feedback that closes the loop: warmer intake air costs more power."""
    p = PowerParams()
    u = np.full(5, 50.0)
    cool = rack_power(u, np.full(5, 18.0), p).total_power_w
    warm = rack_power(u, np.full(5, 30.0), p).total_power_w
    assert np.all(warm > cool)


def test_idle_floor_holds_however_cold_the_air():
    p = PowerParams()
    s = rack_power(np.zeros(4), np.full(4, -40.0), p)
    np.testing.assert_allclose(s.it_power_w, p.servers_per_rack * p.server_idle_w)


def test_fan_ratio_stays_within_bounds():
    p = PowerParams()
    for temp in (-20.0, 20.0, 80.0):
        for util in (0.0, 50.0, 100.0):
            r = fan_ratio(np.array([util]), np.array([temp]), p)
            assert p.fan_min_ratio - 1e-12 <= r[0] <= 1.0 + 1e-12


def test_fan_reaches_full_flow_at_full_load_and_reference_temp():
    p = PowerParams()
    r = fan_ratio(np.array([100.0]), np.array([p.reference_temp_c]), p)
    np.testing.assert_allclose(r, 1.0)


def test_airflow_is_sized_from_the_design_temperature_rise():
    """At full load and full flow the rise across a rack must equal design_delta_t_k."""
    p = PowerParams()
    s = rack_power(np.array([100.0]), np.array([p.reference_temp_c]), p)
    delta_t = s.total_power_w / s.k_w_per_k
    np.testing.assert_allclose(delta_t, p.design_delta_t_k, rtol=1e-9)


def test_k_matches_rho_cp_flow():
    p = PowerParams()
    s = rack_power(np.full(3, 60.0), np.full(3, 24.0), p)
    np.testing.assert_allclose(s.k_w_per_k, RHO_AIR * C_AIR * s.airflow_m3s)
    assert np.all(s.k_w_per_k > 0)


def test_fan_power_follows_a_cube_law():
    p = PowerParams()
    s = rack_power(np.array([100.0, 0.0]), np.full(2, p.reference_temp_c), p)
    expected = p.servers_per_rack * p.fan_ref_power_w * s.fan_ratio ** 3
    np.testing.assert_allclose(s.fan_power_w, expected)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="must match"):
        rack_power(np.zeros(4), np.zeros(5), PowerParams())


def test_provisioning_scales_with_fan_setting():
    p = PowerParams()
    full = crac_supply_flow_m3s(100, p, 1.15, 1.0)
    half = crac_supply_flow_m3s(100, p, 1.15, 0.5)
    np.testing.assert_allclose(half, full / 2)
    with pytest.raises(ValueError, match="fan_frac"):
        crac_supply_flow_m3s(100, p, 1.15, 0.0)


def test_params_validate():
    with pytest.raises(ValueError, match="server_idle_w"):
        PowerParams(server_idle_w=5000.0, server_full_load_w=4000.0)
    with pytest.raises(ValueError, match="fan_min_ratio"):
        PowerParams(fan_min_ratio=0.0)
    with pytest.raises(ValueError, match="design_delta_t_k"):
        PowerParams(design_delta_t_k=0.0)
