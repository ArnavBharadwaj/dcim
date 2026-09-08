"""Heat recirculation: the cross-interference matrix A and the distribution matrix D.

This replaces SustainDC's inlet model, which Phase 0 found to be
`T_inlet[i] = const[i] + T_setpoint` -- no rack power term of any kind, zero hops of
spatial coupling (see docs/simulator-thermal-model.md).

Formulation
-----------
Following the standard heat-recirculation account of a raised-floor hall (Tang et al.,
"Thermal-aware task scheduling for data centers through minimizing heat recirculation"),
let

    a_ij  = fraction of rack i's exhaust flow that is drawn into rack j's inlet
    k_i   = rho * c_p * f_i,  the thermal mass flow of rack i's fans   [W/K]

Rack i raises the air it draws by P_i / k_i, so `T_out = T_in + K^-1 P`. The air
entering rack j is a mixture of CRAC supply at T_supply and other racks' exhaust:

    k_j T_in,j = sum_i a_ij k_i T_out,i + (k_j - sum_i a_ij k_i) T_supply

Substituting and rearranging gives `T_in = T_supply + D P` with

    D = (K - A^T K)^-1 - K^-1
      = K^-1 [ (I - A^T)^-1 - I ]                     (the form used here)
      = K^-1 [ A^T + (A^T)^2 + (A^T)^3 + ... ]        (Neumann series)

The second form is what we compute: it is better conditioned, and it makes the
multi-hop structure explicit. Term m of the series is air that has passed through m
racks before arriving, so `hop_decomposition` can report exactly how many hops of
spatial influence the model carries -- the question Phase 0 asked of SustainDC and
got the answer "zero" to.

Two properties worth stating, because they are what make the model non-trivial:

*   D is dense even when A is sparse. Racks that exchange no air directly still
    couple through chains of intermediaries. The coupling is not a fixed two-hop
    stencil.
*   D is state-dependent. K moves with rack fan speed, and A's leakage moves with
    the CRAC-to-rack flow ratio, so a cooling action reshapes the coupling rather
    than merely shifting it. A single fixed linear operator cannot represent this.

Everything here is deterministic given (geometry, params, state).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .geometry import HallGeometry, rows_crossed


@dataclasses.dataclass(frozen=True)
class RecircParams:
    """Parameters of the recirculation kernel.

    Defaults sit inside the ranges reported for air-cooled raised-floor halls with
    partial containment. They are our modelling choice, not measurements; see
    docs/thermal-model-design.md for the provenance of each one.
    """

    decay_length_m: float = 3.0      # e-folding length of recirculation along an aisle
    row_attenuation: float = 0.35    # factor per extra rack row the air must cross
    weight_downstream: float = 1.0   # air carried toward the CRAC return
    weight_upstream: float = 0.25    # air working against the return path
    escape_base: float = 0.16        # hall-average fraction of exhaust that recirculates
    escape_max: float = 0.55         # cap; above this the hall is in thermal runaway
    crac_distance_gain: float = 0.45     # extra leakage for racks far from a return
    underprovision_gain: float = 1.20    # extra leakage when rack fans outpace the CRACs

    def __post_init__(self) -> None:
        if not 0.0 < self.escape_base < 1.0:
            raise ValueError("escape_base must lie in (0, 1)")
        if not 0.0 < self.escape_max < 1.0:
            raise ValueError("escape_max must lie in (0, 1); at 1.0 no air ever "
                             "returns to the CRACs and D is singular")
        if self.escape_max < self.escape_base:
            raise ValueError("escape_max must be >= escape_base")
        if self.decay_length_m <= 0:
            raise ValueError("decay_length_m must be positive")
        if not 0.0 <= self.row_attenuation <= 1.0:
            raise ValueError("row_attenuation must lie in [0, 1]")


def geometric_kernel(geom: HallGeometry, params: RecircParams) -> np.ndarray:
    """(N, N) unnormalised recirculation weights. Depends only on geometry.

    Entry (i, j) is the relative propensity for rack i's exhaust to reach rack j's
    inlet, before any leakage budget is applied. Computed once per hall and reused
    at every timestep; only the leakage scaling is state-dependent.
    """
    delta = geom.inlet_xy[None, :, :] - geom.exhaust_xy[:, None, :]   # (N, N, 2)
    dist = np.linalg.norm(delta, axis=2)

    # Distance decay along the recirculation path.
    weight = np.exp(-dist / params.decay_length_m)

    # Crossing a row of racks costs a further factor. rows_crossed == 1 is the
    # over-the-top short circuit into an adjacent aisle and is left unattenuated.
    crossings = rows_crossed(geom)
    weight = weight * params.row_attenuation ** np.maximum(crossings - 1, 0)

    # Asymmetry: exhaust is carried toward the CRAC return, so racks downstream of i
    # receive more of i's air than racks upstream. This is what makes the induced
    # graph directed; a symmetric aggregation would be the wrong operator for it.
    norm = np.linalg.norm(delta, axis=2, keepdims=True)
    unit = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 0)
    proj = np.einsum("ijk,ik->ij", unit, geom.return_dir)   # cos angle to return path
    w_up, w_dn = params.weight_upstream, params.weight_downstream
    weight = weight * (w_up + (w_dn - w_up) * (proj + 1.0) / 2.0)

    return weight


def escape_fractions(geom: HallGeometry, params: RecircParams,
                     provisioning_ratio: float,
                     scale: np.ndarray | None = None) -> np.ndarray:
    """(N,) fraction of each rack's exhaust that recirculates instead of returning.

    Two effects, one geometric and one state-dependent:

    *   Racks far from a CRAC return leak more, because their exhaust has further to
        travel before it is captured. Normalised so the hall mean multiplier is 1,
        which keeps `escape_base` interpretable as the hall-average leakage.
    *   When the racks' own fans move more air than the CRACs supply
        (provisioning_ratio < 1), the deficit can only be made up by ingesting
        recirculated air, and leakage climbs steeply. This is the physical mechanism
        that puts CRAC fan speed in control of the coupling structure.
    """
    dist = geom.dist_to_crac
    mean_dist = float(dist.mean())
    if mean_dist <= 0:
        prox = np.ones_like(dist)
    else:
        prox = 1.0 + params.crac_distance_gain * (dist - mean_dist) / mean_dist
    prox = np.maximum(prox, 0.0)

    deficit = max(0.0, 1.0 - float(provisioning_ratio))
    escape = params.escape_base * prox * (1.0 + params.underprovision_gain * deficit)
    if scale is not None:
        # Drift hook: a pulled blanking panel or a failed containment door raises
        # leakage locally. Phase 4 uses this; it is the identity by default.
        escape = escape * np.asarray(scale, dtype=float)
    return np.clip(escape, 0.0, params.escape_max)


def cross_interference(kernel: np.ndarray, escape: np.ndarray,
                       k: np.ndarray) -> np.ndarray:
    """Normalise the geometric kernel into a cross-interference matrix A.

    Row i sums to `escape[i]`, so rack i sheds exactly `escape_i * k_i` of
    recirculating air: mass is conserved at the source.

    The share each target receives is weighted by the target's own thermal mass
    flow `k_j`, not by geometry alone. This says a rack entrains recirculating air
    in proportion to how hard it is pulling, and it is load-bearing rather than
    cosmetic. Without the `k_j` weight, a rack's share of recirculated air is fixed
    while its intake grows with its fan speed, so ramping its fans *dilutes* its own
    inlet and loading a rack up makes it colder. Measured on hall_a, that artefact
    was worth -0.79 K at the perturbed rack while its neighbours rose only +0.08 K,
    which would have taught the placement controller to stack load onto hot racks.

    Real halls run the other way: a rack whose airflow demand outruns the cold air
    delivered to its aisle has to make up the deficit by ingesting exhaust. With the
    weight in place, the temperature contribution a rack receives from any source is
    independent of its own fan speed, and the dilution artefact is gone.
    """
    if kernel.shape[0] != k.shape[0] or kernel.shape[0] != escape.shape[0]:
        raise ValueError(f"shape mismatch: kernel {kernel.shape}, escape "
                         f"{escape.shape}, k {k.shape}")
    if np.any(k <= 0):
        raise ValueError("every rack must have positive airflow")
    weighted = kernel * k[None, :]
    total = weighted.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("a rack has zero recirculation weight to every inlet; "
                         "check decay_length_m against the hall dimensions")
    return weighted * (escape[:, None] / total)


def distribution_matrix(A: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Heat distribution matrix D, so that `T_inlet = T_supply + D @ P`.

    D = K^-1 [ (I - A^T)^-1 - I ], with K = diag(k).

    Raises if A's recirculation is strong enough that the linear system is unusable,
    rather than silently returning a huge or negative-entried D.
    """
    n = A.shape[0]
    if A.shape != (n, n) or k.shape != (n,):
        raise ValueError(f"shape mismatch: A {A.shape}, k {k.shape}")
    if np.any(k <= 0):
        raise ValueError("every rack must have positive airflow; k_i <= 0 for "
                         f"{int((k <= 0).sum())} rack(s)")

    row_sums = A.sum(axis=1)
    if row_sums.max() >= 1.0:
        raise ValueError(f"max row sum of A is {row_sums.max():.4f}; at >= 1 no air "
                         "returns to the CRACs and the hall has no steady state")

    M = np.eye(n) - A.T
    cond = np.linalg.cond(M)
    if not np.isfinite(cond) or cond > 1e10:
        raise ValueError(f"(I - A^T) is badly conditioned (cond={cond:.3e})")

    # (I - A^T)^-1 - I, solved rather than inverted-then-subtracted.
    inv_minus_I = np.linalg.solve(M, A.T)
    return inv_minus_I / k[:, None]


def hop_decomposition(A: np.ndarray, k: np.ndarray, max_order: int = 8) -> list[np.ndarray]:
    """[D_1, ..., D_max_order] where D_m is the m-hop term of the Neumann series.

    D_m = K^-1 (A^T)^m is the contribution of air that passed through exactly m racks
    before reaching the inlet. sum_m D_m converges to D. Used to report how many hops
    of spatial influence the model actually carries.
    """
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    terms, power = [], np.eye(A.shape[0])
    for _ in range(max_order):
        power = power @ A.T
        terms.append(power / k[:, None])
    return terms


def build_D(geom: HallGeometry, params: RecircParams, k: np.ndarray,
            provisioning_ratio: float,
            kernel: np.ndarray | None = None,
            escape_scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: geometry + state -> (D, A).

    `kernel` may be passed in to avoid recomputing the geometry-only part, which is
    fixed for the life of a hall.
    """
    if kernel is None:
        kernel = geometric_kernel(geom, params)
    escape = escape_fractions(geom, params, provisioning_ratio, escape_scale)
    A = cross_interference(kernel, escape, k)
    return distribution_matrix(A, k), A
