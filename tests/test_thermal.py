"""Tests for the twin's thermal dynamics."""

import numpy as np
import pytest

from src.twin.geometry import build_hall
from src.twin.thermal import ThermalParams, ThermalTwin


@pytest.fixture
def twin(hall_cfg):
    return ThermalTwin(build_hall(hall_cfg))


def test_steady_state_is_a_fixed_point(twin):
    """Stepping from equilibrium must not move, whatever the step size."""
    n = twin.geom.n_racks
    u = np.full(n, 60.0)
    twin.reset(u, supply_temp_c=20.0)
    start = twin.step(u, 20.0, dt_s=1e-6).inlet_temp_c.copy()
    after = twin.step(u, 20.0, dt_s=3600.0).inlet_temp_c
    np.testing.assert_allclose(after, start, atol=1e-6)


def test_dynamics_converge_to_the_steady_state(twin):
    n = twin.geom.n_racks
    u = np.full(n, 75.0)
    target = twin.steady_state(u, 21.0, 1.0)
    twin.reset(u, 21.0, inlet_temp_c=np.full(n, 21.0))
    for _ in range(400):
        s = twin.step(u, 21.0, dt_s=30.0)
    np.testing.assert_allclose(s.inlet_temp_c, target, atol=1e-3)


def test_time_constant_governs_the_approach(twin):
    """After exactly one time constant the gap must have closed by 1 - 1/e."""
    n = twin.geom.n_racks
    u = np.full(n, 80.0)
    tau = twin.thermal.time_constant_s
    twin.reset(u, 20.0, inlet_temp_c=np.full(n, 20.0))
    start = np.full(n, 20.0)
    target = twin.step(u, 20.0, dt_s=1e-9).steady_target_c.copy()
    twin.reset(u, 20.0, inlet_temp_c=start.copy())
    after = twin.step(u, 20.0, dt_s=tau).inlet_temp_c
    closed = (after - start) / (target - start)
    np.testing.assert_allclose(closed, 1.0 - np.exp(-1.0), rtol=1e-3)


def test_supply_setpoint_gain_is_below_one_and_varies_by_rack(twin):
    """Raising the supply temperature warms every rack, but by less than the full
    amount and by a different amount at each rack.

    Two competing feedbacks: warmer intake costs more power (pushing temperature
    up) but also makes server fans ramp, which raises airflow, enlarges K and
    shrinks the recirculation term. The fan response wins, so the gain is
    sub-unity.

    This is the direct contrast with SustainDC, where d(inlet)/d(setpoint) is
    exactly 1.000000 at every rack, so the setpoint rigidly translates the whole
    temperature field and can never reshape it.
    """
    n = twin.geom.n_racks
    u = np.full(n, 50.0)
    a = twin.steady_state(u, 18.0, 1.0)
    b = twin.steady_state(u, 23.0, 1.0)
    gain = (b - a) / 5.0
    assert np.all(gain > 0.0), "warmer supply air must warm every rack"
    assert np.all(gain < 1.0), "fan ramp-up must partly absorb the increase"
    assert gain.max() - gain.min() > 1e-3, \
        "the setpoint must reshape the field, not merely translate it"


def test_lower_crac_fan_speed_makes_the_hall_hotter(twin):
    """The mechanism that gives the cooling controller a lever on the coupling."""
    n = twin.geom.n_racks
    u = np.full(n, 100.0)
    fast = twin.steady_state(u, 20.0, 1.0)
    slow = twin.steady_state(u, 20.0, 0.6)
    assert slow.mean() > fast.mean()


def test_concentrating_load_costs_peak_temperature(twin):
    """Placement must have leverage: the same total load stacked in one place must
    be hotter at the peak than the same load spread out. Without this the placement
    controller has nothing to optimise."""
    n = twin.geom.n_racks
    stacked = np.zeros(n)
    stacked[: n // 4] = 100.0
    spread = np.full(n, 25.0)
    assert np.isclose(stacked.sum(), spread.sum())
    assert twin.steady_state(stacked, 20.0, 1.0).max() > \
        twin.steady_state(spread, 20.0, 1.0).max() + 0.5


def test_outlet_is_hotter_than_inlet_by_the_rack_rise(twin):
    n = twin.geom.n_racks
    s = twin.reset(np.full(n, 70.0), 20.0)
    np.testing.assert_allclose(s.outlet_temp_c - s.inlet_temp_c,
                               s.rack_power_w / (1.19 * 1006.0 * s.airflow_m3s),
                               rtol=1e-9)


def test_crac_return_lies_between_inlet_and_outlet(twin):
    n = twin.geom.n_racks
    s = twin.reset(np.full(n, 70.0), 20.0)
    assert s.outlet_temp_c.min() <= s.crac_return_temp_c <= s.outlet_temp_c.max()


def test_step_before_reset_is_an_error(twin):
    with pytest.raises(RuntimeError, match="reset"):
        twin.step(np.full(twin.geom.n_racks, 50.0), 20.0)


def test_wrong_shaped_utilisation_is_rejected(twin):
    twin.reset(np.full(twin.geom.n_racks, 50.0), 20.0)
    with pytest.raises(ValueError, match="must have shape"):
        twin.step(np.full(3, 50.0), 20.0)


def test_stepping_is_deterministic(hall_cfg):
    n = build_hall(hall_cfg).n_racks
    rng = np.random.default_rng(7)
    u = rng.uniform(0, 100, size=n)
    runs = []
    for _ in range(2):
        t = ThermalTwin(build_hall(hall_cfg))
        t.reset(u, 20.0)
        for _ in range(20):
            s = t.step(u, 20.0, dt_s=30.0)
        runs.append(s.inlet_temp_c)
    np.testing.assert_array_equal(runs[0], runs[1])


def test_params_validate():
    with pytest.raises(ValueError, match="time_constant_s"):
        ThermalParams(time_constant_s=0.0)


def test_loading_a_rack_never_cools_it(twin):
    """Raising one rack's utilisation must raise its own inlet temperature.

    Regression test for the dilution artefact described in
    tests/test_recirculation.py::test_entrainment_weight_makes_rise_independent_of_own_airflow.
    A placement controller trained against a model that violates this would learn to
    stack load onto exactly the racks it should be avoiding.
    """
    n = twin.geom.n_racks
    base = np.full(n, 50.0)
    cold = twin.steady_state(base, 20.0, 1.0)
    for src in range(0, n, max(1, n // 8)):
        load = base.copy()
        load[src] = 100.0
        hot = twin.steady_state(load, 20.0, 1.0)
        assert hot[src] > cold[src], (
            f"loading rack {src} from 50% to 100% cooled it by "
            f"{cold[src] - hot[src]:.4f} K")


def test_added_heat_warms_the_hall_on_balance(twin):
    """Whatever the local redistribution, adding power must raise the mean."""
    n = twin.geom.n_racks
    base = np.full(n, 50.0)
    load = base.copy()
    load[n // 2] = 100.0
    assert twin.steady_state(load, 20.0, 1.0).mean() > \
        twin.steady_state(base, 20.0, 1.0).mean()
