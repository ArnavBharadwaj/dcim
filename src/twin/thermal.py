"""The twin's thermal core: one timestep of the data hall.

Composition of the pieces:

    power.rack_power          util, T_inlet          ->  P, airflow, K
    recirculation.build_D     geometry, K, phi       ->  D, A
                              T_steady = T_supply + D @ P
    first-order lag           T_inlet -> T_steady with time constant tau

The lag is what makes prediction horizons mean anything. With a memoryless model,
predicting temperature at t+H collapses into predicting workload at t+H and every
horizon measures the same thing. With a rack-air time constant near a minute, a 30 s
horizon is transient and a 300 s horizon is nearly steady-state.

Power is evaluated at the *current* inlet temperature rather than the steady-state
one, so the feedback loop (hotter air -> more power and more fan flow -> hotter air)
resolves through time rather than through an implicit solve. That is both more
physical and cheaper. `steady_state` does run the implicit solve, and is used to
initialise an episode without a long spin-up.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .geometry import HallGeometry
from .power import C_AIR, RHO_AIR, PowerParams, crac_supply_flow_m3s, rack_power
from .recirculation import RecircParams, build_D, geometric_kernel


@dataclasses.dataclass(frozen=True)
class ThermalParams:
    """Dynamics and operating envelope."""

    time_constant_s: float = 60.0    # rack-air thermal time constant
    design_provisioning: float = 1.15   # CRAC flow / rack flow at full load, fans at 100%
    runaway_threshold_c: float = 60.0   # above this the state is flagged, never clipped

    def __post_init__(self) -> None:
        if self.time_constant_s <= 0:
            raise ValueError("time_constant_s must be positive")


@dataclasses.dataclass(frozen=True)
class TwinState:
    """Everything observable about the hall at one instant."""

    time_s: float
    inlet_temp_c: np.ndarray      # (N,) the quantity the GNN predicts
    outlet_temp_c: np.ndarray     # (N,)
    steady_target_c: np.ndarray   # (N,) where inlet temperature is heading
    util_pct: np.ndarray          # (N,)
    rack_power_w: np.ndarray      # (N,)
    it_power_w: np.ndarray        # (N,)
    fan_power_w: np.ndarray       # (N,)
    airflow_m3s: np.ndarray       # (N,)
    supply_temp_c: float
    crac_fan_frac: float
    provisioning_ratio: float
    crac_return_temp_c: float
    runaway: bool

    @property
    def total_it_power_w(self) -> float:
        return float(self.it_power_w.sum() + self.fan_power_w.sum())


class ThermalTwin:
    """Dynamic thermal model of one hall.

    The geometric part of the recirculation kernel is computed once at construction;
    only the leakage scaling and K change per step, so a step is one dense solve.
    """

    def __init__(self, geom: HallGeometry,
                 recirc: RecircParams | None = None,
                 power: PowerParams | None = None,
                 thermal: ThermalParams | None = None):
        self.geom = geom
        self.recirc = recirc or RecircParams()
        self.power = power or PowerParams()
        self.thermal = thermal or ThermalParams()
        self._kernel = geometric_kernel(self.geom, self.recirc)
        self._inlet: np.ndarray | None = None
        self._time_s = 0.0

    # ---------------------------------------------------------------- helpers

    def _provisioning(self, fan_frac: float, airflow: np.ndarray,
                      crac_flow_scale: np.ndarray | None) -> float:
        supply = crac_supply_flow_m3s(self.geom.n_racks, self.power,
                                      self.thermal.design_provisioning, fan_frac)
        if crac_flow_scale is not None:
            # Per-unit derate, weighted by each unit's share of total flow. A failed
            # CRAC removes its share; Phase 4's drift injection uses this.
            share = self.geom.crac_flow_share
            supply *= float(np.dot(share, np.asarray(crac_flow_scale, dtype=float)))
        demand = float(airflow.sum())
        if demand <= 0:
            raise ValueError("total rack airflow is zero")
        return supply / demand

    def _crac_return_temp(self, outlet: np.ndarray, k: np.ndarray,
                          escape: np.ndarray) -> float:
        """Flow-weighted mixed temperature of the air that reaches the CRAC returns.

        Only the fraction of each rack's exhaust that is *not* recirculated arrives
        at a return, so the mixture is weighted by (1 - escape_i) * k_i. This is an
        energy balance rather than SustainDC's unweighted mean over racks.
        """
        w = (1.0 - escape) * k
        total = float(w.sum())
        if total <= 0:
            raise ValueError("no exhaust reaches the CRAC returns")
        return float(np.dot(w, outlet) / total)

    # ------------------------------------------------------------------- API

    def steady_state(self, util_pct: np.ndarray, supply_temp_c: float,
                     crac_fan_frac: float = 1.0,
                     escape_scale: np.ndarray | None = None,
                     crac_flow_scale: np.ndarray | None = None,
                     tol: float = 1e-9, max_iter: int = 200,
                     damping: float = 0.5) -> np.ndarray:
        """Solve the implicit fixed point T = T_supply + D(T) @ P(T).

        Damped Picard iteration. Used to start an episode in equilibrium rather than
        from an arbitrary temperature; the dynamics themselves never need it.
        """
        util = np.asarray(util_pct, dtype=float)
        temp = np.full(self.geom.n_racks, float(supply_temp_c))
        for _ in range(max_iter):
            state = rack_power(util, temp, self.power)
            phi = self._provisioning(crac_fan_frac, state.airflow_m3s, crac_flow_scale)
            D, _ = build_D(self.geom, self.recirc, state.k_w_per_k, phi,
                           kernel=self._kernel, escape_scale=escape_scale)
            target = supply_temp_c + D @ state.total_power_w
            new = temp + damping * (target - temp)
            if np.max(np.abs(new - temp)) < tol:
                return new
            temp = new
        raise RuntimeError(
            f"steady state did not converge in {max_iter} iterations; the hall may be "
            "in thermal runaway at this operating point")

    def reset(self, util_pct: np.ndarray, supply_temp_c: float,
              crac_fan_frac: float = 1.0, time_s: float = 0.0,
              inlet_temp_c: np.ndarray | None = None) -> TwinState:
        """Initialise. Starts at the steady state for the given operating point
        unless an explicit inlet temperature field is supplied."""
        if inlet_temp_c is None:
            self._inlet = self.steady_state(util_pct, supply_temp_c, crac_fan_frac)
        else:
            self._inlet = np.array(inlet_temp_c, dtype=float)
            if self._inlet.shape != (self.geom.n_racks,):
                raise ValueError(f"inlet_temp_c must have shape ({self.geom.n_racks},)")
        self._time_s = float(time_s)
        return self._observe(np.asarray(util_pct, dtype=float), supply_temp_c,
                             crac_fan_frac, self._inlet, None, None)

    def step(self, util_pct: np.ndarray, supply_temp_c: float,
             crac_fan_frac: float = 1.0, dt_s: float = 30.0,
             escape_scale: np.ndarray | None = None,
             crac_flow_scale: np.ndarray | None = None) -> TwinState:
        """Advance the hall by dt_s seconds and return the new state."""
        if self._inlet is None:
            raise RuntimeError("call reset() before step()")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")

        util = np.asarray(util_pct, dtype=float)
        if util.shape != (self.geom.n_racks,):
            raise ValueError(f"util_pct must have shape ({self.geom.n_racks},), "
                             f"got {util.shape}")

        state = self._observe(util, supply_temp_c, crac_fan_frac, self._inlet,
                              escape_scale, crac_flow_scale)

        # Exact discretisation of dT/dt = (T_ss - T) / tau over one step.
        alpha = 1.0 - np.exp(-dt_s / self.thermal.time_constant_s)
        self._inlet = self._inlet + alpha * (state.steady_target_c - self._inlet)
        self._time_s += dt_s

        return self._observe(util, supply_temp_c, crac_fan_frac, self._inlet,
                             escape_scale, crac_flow_scale)

    def _observe(self, util: np.ndarray, supply_temp_c: float, crac_fan_frac: float,
                 inlet: np.ndarray,
                 escape_scale: np.ndarray | None,
                 crac_flow_scale: np.ndarray | None) -> TwinState:
        rs = rack_power(util, inlet, self.power)
        phi = self._provisioning(crac_fan_frac, rs.airflow_m3s, crac_flow_scale)
        D, A = build_D(self.geom, self.recirc, rs.k_w_per_k, phi,
                       kernel=self._kernel, escape_scale=escape_scale)
        target = supply_temp_c + D @ rs.total_power_w
        outlet = inlet + rs.total_power_w / rs.k_w_per_k
        escape = A.sum(axis=1)
        return TwinState(
            time_s=self._time_s,
            inlet_temp_c=inlet.copy(),
            outlet_temp_c=outlet,
            steady_target_c=target,
            util_pct=util.copy(),
            rack_power_w=rs.total_power_w,
            it_power_w=rs.it_power_w,
            fan_power_w=rs.fan_power_w,
            airflow_m3s=rs.airflow_m3s,
            supply_temp_c=float(supply_temp_c),
            crac_fan_frac=float(crac_fan_frac),
            provisioning_ratio=phi,
            crac_return_temp_c=self._crac_return_temp(outlet, rs.k_w_per_k, escape),
            runaway=bool(np.max(inlet) > self.thermal.runaway_threshold_c),
        )

    def influence_matrix(self, util_pct: np.ndarray, inlet_temp_c: np.ndarray,
                         crac_fan_frac: float = 1.0) -> np.ndarray:
        """D at a given operating point, for analysis and for probing the model."""
        rs = rack_power(np.asarray(util_pct, dtype=float),
                        np.asarray(inlet_temp_c, dtype=float), self.power)
        phi = self._provisioning(crac_fan_frac, rs.airflow_m3s, None)
        D, _ = build_D(self.geom, self.recirc, rs.k_w_per_k, phi, kernel=self._kernel)
        return D


def air_thermal_capacity(airflow_m3s: float) -> float:
    """rho * c_p * flow, in W/K. Exposed for tests and for the RC baseline."""
    return RHO_AIR * C_AIR * airflow_m3s
