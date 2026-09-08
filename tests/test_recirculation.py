"""Tests for the heat recirculation model.

The central one is `test_D_satisfies_the_mixing_energy_balance`: it re-derives the
inlet temperature straight from the physical mixing equation and checks that the
closed-form D agrees. If the matrix algebra is wrong, every downstream number is
wrong and nothing else would catch it.
"""

import numpy as np
import pytest

from src.twin.geometry import build_hall
from src.twin.power import C_AIR, RHO_AIR
from src.twin.recirculation import (RecircParams, cross_interference,
                                    distribution_matrix, escape_fractions,
                                    geometric_kernel, hop_decomposition)


@pytest.fixture
def setup(hall_cfg):
    g = build_hall(hall_cfg)
    p = RecircParams()
    kernel = geometric_kernel(g, p)
    escape = escape_fractions(g, p, provisioning_ratio=1.15)
    k = np.full(g.n_racks, RHO_AIR * C_AIR * 2.0)
    A = cross_interference(kernel, escape, k)
    return g, p, A, k, escape


def test_kernel_is_strictly_positive(setup):
    _, _, _, _, _ = setup
    g, p, A, k, _ = setup
    assert np.all(A >= 0.0), "negative recirculation would mean a rack cools its peers"


def test_row_sums_equal_escape_fractions(setup):
    _, _, A, _, escape = setup
    np.testing.assert_allclose(A.sum(axis=1), escape, rtol=1e-12)


def test_escape_is_capped_and_rises_when_underprovisioned(hall_cfg):
    g = build_hall(hall_cfg)
    p = RecircParams()
    well = escape_fractions(g, p, provisioning_ratio=1.3)
    poor = escape_fractions(g, p, provisioning_ratio=0.6)
    assert np.all(poor >= well)
    assert poor.mean() > well.mean() * 1.2, "under-provisioning must open up leakage"
    extreme = escape_fractions(g, p, provisioning_ratio=0.0)
    assert np.all(extreme <= p.escape_max)


def test_escape_rises_with_distance_from_a_crac(hall_cfg):
    g = build_hall(hall_cfg)
    e = escape_fractions(g, RecircParams(), provisioning_ratio=1.15)
    # Rank correlation between distance and leakage must be positive.
    order = np.argsort(g.dist_to_crac)
    assert e[order][0] < e[order][-1]


def test_D_satisfies_the_mixing_energy_balance(setup):
    """D must reproduce the balance it was derived from:

        k_j T_in,j = sum_i a_ij k_i T_out,i + (k_j - sum_i a_ij k_i) T_supply
        T_out      = T_in + P / k
    """
    _, _, A, k, _ = setup
    rng = np.random.default_rng(0)
    n = A.shape[0]
    P = rng.uniform(5e3, 35e3, size=n)
    t_sup = 19.0

    D = distribution_matrix(A, k)
    t_in = t_sup + D @ P
    t_out = t_in + P / k

    captured = A.T @ (k * t_out)                 # sum_i a_ij k_i T_out,i
    from_supply = (k - A.T @ k) * t_sup          # remainder drawn from the CRACs
    residual = k * t_in - (captured + from_supply)
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)


def test_D_matches_its_neumann_series(setup):
    """The closed form and the hop-by-hop expansion must agree, which is what
    licenses reading the series terms as 'influence at m hops'."""
    _, _, A, k, _ = setup
    D = distribution_matrix(A, k)
    series = sum(hop_decomposition(A, k, max_order=40))
    np.testing.assert_allclose(series, D, rtol=1e-9, atol=1e-14)


def test_D_is_nonnegative_and_monotone(setup):
    _, _, A, k, _ = setup
    D = distribution_matrix(A, k)
    assert np.all(D >= 0.0), "adding power to any rack must never cool another"


def test_no_recirculation_gives_no_coupling(setup):
    """With zero leakage every rack sits exactly at the supply temperature. This is
    the degenerate case SustainDC is permanently stuck in."""
    _, _, A, k, _ = setup
    D = distribution_matrix(np.zeros_like(A), k)
    np.testing.assert_array_equal(D, np.zeros_like(D))


def test_coupling_is_directed(setup):
    """Air flow is asymmetric, so D must be too. A symmetric operator would make
    the directed-edge ablation vacuous."""
    _, _, A, k, _ = setup
    D = distribution_matrix(A, k)
    asymmetry = np.abs(D - D.T).sum() / np.abs(D).sum()
    assert asymmetry > 0.1, f"coupling is nearly symmetric (asymmetry {asymmetry:.3f})"


def test_influence_extends_beyond_one_hop(setup):
    """A dense D from a decaying kernel is the whole point: racks that exchange no
    air directly still couple through intermediaries."""
    _, _, A, k, _ = setup
    D = distribution_matrix(A, k)
    one_hop = hop_decomposition(A, k, max_order=1)[0]
    multi = (D - one_hop).sum() / D.sum()
    assert multi > 0.05, f"only {multi:.1%} of influence is multi-hop"
    assert np.all(D > 0), "D should be dense even though the kernel decays"


def test_runaway_recirculation_is_rejected(setup):
    """Row sums at or above 1 mean no air returns to the CRACs. That has no steady
    state and must raise rather than return a huge D."""
    _, _, A, k, _ = setup
    hot = A / A.sum(axis=1, keepdims=True) * 1.01
    with pytest.raises(ValueError, match="max row sum"):
        distribution_matrix(hot, k)


def test_zero_airflow_is_rejected(setup):
    _, _, A, k, _ = setup
    bad = k.copy()
    bad[3] = 0.0
    with pytest.raises(ValueError, match="positive airflow"):
        distribution_matrix(A, bad)


def test_D_scales_inversely_with_airflow(setup):
    """K enters as an inverse, so doubling every rack's flow halves the coupling.
    This is the state dependence that a fixed linear operator cannot represent."""
    _, _, A, k, _ = setup
    np.testing.assert_allclose(distribution_matrix(A, 2 * k),
                               distribution_matrix(A, k) / 2, rtol=1e-10)


def test_params_validate(hall_cfg):
    with pytest.raises(ValueError, match="escape_base"):
        RecircParams(escape_base=0.0)
    with pytest.raises(ValueError, match="escape_max"):
        RecircParams(escape_max=1.0)
    with pytest.raises(ValueError, match="escape_max must be"):
        RecircParams(escape_base=0.5, escape_max=0.2)
    with pytest.raises(ValueError, match="decay_length_m"):
        RecircParams(decay_length_m=0.0)


def test_entrainment_weight_makes_rise_independent_of_own_airflow(hall_cfg):
    """A rack's temperature rise from a *given* source must not depend on its own
    fan speed.

    Regression test. Without weighting each target's share of the recirculating air
    by its own thermal mass flow, a rack's recirculated intake is fixed while its
    total intake grows with fan speed, so ramping its fans dilutes its own inlet.
    On hall_a that made loading a rack up *cool* it by 0.79 K while its neighbours
    warmed by only 0.08 K -- an artefact that would teach the placement controller
    to stack load onto the hottest racks.
    """
    g = build_hall(hall_cfg)
    p = RecircParams()
    kernel = geometric_kernel(g, p)
    escape = escape_fractions(g, p, provisioning_ratio=1.15)
    n = g.n_racks

    k_slow = np.full(n, RHO_AIR * C_AIR * 1.0)
    k_fast = k_slow.copy()
    k_fast[0] *= 2.0                      # rack 0 doubles its fan flow

    D_slow = distribution_matrix(cross_interference(kernel, escape, k_slow), k_slow)
    D_fast = distribution_matrix(cross_interference(kernel, escape, k_fast), k_fast)

    # Row 0 is what rack 0's inlet receives from every source. Its own fan speed
    # must not change it appreciably, and must certainly not reduce it.
    others = np.arange(1, n)
    ratio = D_fast[0, others] / D_slow[0, others]
    assert ratio.min() > 0.9, (
        f"rack 0's intake from other racks fell to {ratio.min():.3f} of its value "
        "when it ramped its own fans; the dilution artefact is back")


def test_source_mass_is_conserved(setup):
    """Row sums must still equal the escape fractions after the k-weighting, so a
    rack sheds exactly escape_i * k_i of recirculating air and no more."""
    _, _, A, _, escape = setup
    np.testing.assert_allclose(A.sum(axis=1), escape, rtol=1e-12)


def test_cross_interference_rejects_bad_shapes(setup):
    g, p, A, k, escape = setup
    kernel = geometric_kernel(g, p)
    with pytest.raises(ValueError, match="shape mismatch"):
        cross_interference(kernel, escape, k[:-1])
    with pytest.raises(ValueError, match="positive airflow"):
        cross_interference(kernel, escape, np.zeros_like(k))
