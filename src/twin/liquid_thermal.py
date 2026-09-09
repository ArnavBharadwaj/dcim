"""The liquid-cooled twin: a coolant loop and a residual air loop, stepped together.

A direct-to-chip hall is two cooling systems at once. Cold plates take most of the heat
into coolant; the rest -- power supplies, memory, network, drives -- still leaves the
rack as warm air and still recirculates. Both are modelled, and the second reuses the
air twin's recirculation kernel unchanged.

The two sides couple in one direction only, which keeps the step explicit:

    rack power  ->  liquid share  ->  coolant loop  ->  coolant supply, case temperature
                ->  air share     ->  recirculation ->  air inlet temperature

Chip power then depends on case temperature at the *next* step rather than this one, so
the leakage feedback resolves through time exactly as it does in the air twin, with no
implicit solve.

What the controllers get here that they did not get in the air hall: two independent
cooling levers with different costs. Facility water temperature is slow and plant-wide;
pump speed is fast, per-loop, and moves chip temperature through the cold plate's
convective coefficient without moving the coolant supply at all.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .geometry import HallGeometry
from .liquid import CoolantState, LiquidParams, LoopTopology, solve_coolant
from .power import C_AIR, RHO_AIR, PowerParams, crac_supply_flow_m3s, rack_power
from .recirculation import RecircParams, build_D, geometric_kernel
from .thermal import ThermalParams


@dataclasses.dataclass(frozen=True)
class LiquidLimits:
    """Operating envelope for a liquid-cooled AI hall.

    Facility water bands follow the ASHRAE liquid cooling classes; W32 covers a hall
    running on a dry cooler in most climates without mechanical chilling, which is the
    reason to build liquid in the first place.
    """

    case_temp_max_c: float = 85.0        # chip case limit; throttling above this
    case_temp_warn_c: float = 78.0
    coolant_return_max_c: float = 65.0   # secondary loop material limit
    air_inlet_max_c: float = 27.0        # residual air side, ASHRAE A1 recommended
    # ASHRAE liquid classes W32 through W45. The upper end is where warm-water halls
    # run when they have no mechanical chilling at all, and it is where the case-
    # temperature constraint actually binds; at the W32 end the hall has easy margin.
    facility_water_min_c: float = 26.0
    facility_water_max_c: float = 42.0
    # The residual-air side runs on its own chilled loop, not on the warm facility
    # water. A W32 hall cannot deliver 27 C air from 32 C water, and real deployments
    # that mix liquid and air keep a separate CRAH loop for exactly that reason.
    air_supply_min_c: float = 18.0
    air_supply_max_c: float = 26.0
    pump_frac_min: float = 0.55
    pump_frac_max: float = 1.0


@dataclasses.dataclass(frozen=True)
class LiquidTwinState:
    """Everything observable about a liquid-cooled hall at one instant."""

    time_s: float
    # Coolant side
    coolant_supply_c: np.ndarray     # (N,) the prediction target
    coolant_return_c: np.ndarray
    case_temp_c: np.ndarray          # (N,) what actually throttles
    flow_kgs: np.ndarray
    # Air side
    air_inlet_c: np.ndarray          # (N,)
    air_outlet_c: np.ndarray
    # Power split
    rack_power_w: np.ndarray
    liquid_power_w: np.ndarray
    air_power_w: np.ndarray
    util_pct: np.ndarray
    # Plant
    facility_water_c: float
    air_supply_c: float
    pump_frac: float
    crac_fan_frac: float
    cdu_load_kw: np.ndarray          # (M,)
    cdu_utilisation: np.ndarray      # (M,)
    cdu_approach_k: np.ndarray       # (M,)
    provisioning_ratio: float
    throttling: bool

    @property
    def total_it_power_w(self) -> float:
        return float(self.rack_power_w.sum())


class LiquidCooledTwin:
    """Dynamic model of a direct-to-chip liquid-cooled hall."""

    def __init__(self, geom: HallGeometry, topology: LoopTopology,
                 liquid: LiquidParams | None = None,
                 recirc: RecircParams | None = None,
                 power: PowerParams | None = None,
                 thermal: ThermalParams | None = None,
                 limits: LiquidLimits | None = None):
        self.geom = geom
        self.topo = topology
        self.liquid = liquid or LiquidParams()
        self.recirc = recirc or RecircParams()
        self.power = power or PowerParams()
        self.thermal = thermal or ThermalParams()
        self.limits = limits or LiquidLimits()
        self._kernel = geometric_kernel(self.geom, self.recirc)
        self._case: np.ndarray | None = None
        self._air: np.ndarray | None = None
        self._time_s = 0.0

    # -------------------------------------------------------------- internals

    def _air_side(self, util, case_temp, air_inlet, air_power_w, crac_fan_frac,
                  escape_scale=None):
        """Residual-air recirculation, using the air twin's kernel unchanged.

        Rack airflow is scaled by the air share of the load. A liquid-cooled rack ships
        with far smaller fans than an air-cooled one of the same power, and sizing its
        airflow on total rack power would understate the air-side temperature rise by
        the same factor the cold plates removed.
        """
        rs = rack_power(util, air_inlet, self.power)
        air_share = 1.0 - self.liquid.liquid_fraction
        airflow = rs.airflow_m3s * air_share
        k = RHO_AIR * C_AIR * airflow

        supply = crac_supply_flow_m3s(self.geom.n_racks, self.power,
                                      self.thermal.design_provisioning, crac_fan_frac)
        supply *= air_share
        phi = supply / float(airflow.sum())

        D, _ = build_D(self.geom, self.recirc, k, phi, kernel=self._kernel,
                       escape_scale=escape_scale)
        return D, k, phi, airflow

    def _observe(self, util, facility_water_c, air_supply_c, pump_frac, crac_fan_frac,
                 case_temp, air_inlet, cdu_derate, escape_scale):
        # Chip power responds to case temperature; fan power to air inlet temperature.
        rs = rack_power(util, case_temp, self.power)
        p = rs.total_power_w

        coolant: CoolantState = solve_coolant(
            self.topo, self.liquid, p, facility_water_c, pump_frac, cdu_derate)

        D, k, phi, airflow = self._air_side(
            util, case_temp, air_inlet, coolant.air_power_w, crac_fan_frac,
            escape_scale)
        # The air side carries only the residual load, off its own chilled loop.
        air_target = air_supply_c + D @ coolant.air_power_w
        air_out = air_inlet + coolant.air_power_w / k

        return coolant, air_target, air_out, p, phi, rs

    # ------------------------------------------------------------------- API

    def reset(self, util_pct, facility_water_c, air_supply_c=22.0, pump_frac=1.0,
              crac_fan_frac=1.0, time_s=0.0, iterations=80) -> LiquidTwinState:
        """Start from the equilibrium of the given operating point."""
        util = np.asarray(util_pct, dtype=float)
        if util.shape != (self.geom.n_racks,):
            raise ValueError(f"util_pct must have shape ({self.geom.n_racks},)")
        case = np.full(self.geom.n_racks, float(facility_water_c) + 15.0)
        air = np.full(self.geom.n_racks, float(air_supply_c) + 2.0)
        for _ in range(iterations):
            coolant, air_target, _, _, _, _ = self._observe(
                util, facility_water_c, air_supply_c, pump_frac, crac_fan_frac,
                case, air, None, None)
            new_case = case + 0.5 * (coolant.case_temp_c - case)
            new_air = air + 0.5 * (air_target - air)
            if (np.max(np.abs(new_case - case)) < 1e-9
                    and np.max(np.abs(new_air - air)) < 1e-9):
                case, air = new_case, new_air
                break
            case, air = new_case, new_air
        self._case, self._air, self._time_s = case, air, float(time_s)
        return self._state(util, facility_water_c, air_supply_c, pump_frac,
                           crac_fan_frac, None, None)

    def step(self, util_pct, facility_water_c, air_supply_c=22.0, pump_frac=1.0,
             crac_fan_frac=1.0, dt_s=30.0, cdu_derate=None,
             escape_scale=None) -> LiquidTwinState:
        """Advance the hall by dt_s seconds.

        The coolant loop's own transport delay is seconds, far below the 30 s step, so
        the coolant side is treated as quasi-static and only the thermal masses lag:
        the chip-and-cold-plate assembly, and the rack air.
        """
        if self._case is None:
            raise RuntimeError("call reset() before step()")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        util = np.asarray(util_pct, dtype=float)
        if util.shape != (self.geom.n_racks,):
            raise ValueError(f"util_pct must have shape ({self.geom.n_racks},), "
                             f"got {util.shape}")

        coolant, air_target, _, _, _, _ = self._observe(
            util, facility_water_c, air_supply_c, pump_frac, crac_fan_frac,
            self._case, self._air, cdu_derate, escape_scale)

        # Silicon and cold plate settle much faster than a rack of air.
        a_case = 1.0 - np.exp(-dt_s / max(self.thermal.time_constant_s * 0.25, 1e-6))
        a_air = 1.0 - np.exp(-dt_s / self.thermal.time_constant_s)
        self._case = self._case + a_case * (coolant.case_temp_c - self._case)
        self._air = self._air + a_air * (air_target - self._air)
        self._time_s += dt_s
        return self._state(util, facility_water_c, air_supply_c, pump_frac,
                           crac_fan_frac, cdu_derate, escape_scale)

    def _state(self, util, facility_water_c, air_supply_c, pump_frac, crac_fan_frac,
               cdu_derate, escape_scale) -> LiquidTwinState:
        coolant, air_target, air_out, p, phi, rs = self._observe(
            util, facility_water_c, air_supply_c, pump_frac, crac_fan_frac,
            self._case, self._air, cdu_derate, escape_scale)
        return LiquidTwinState(
            time_s=self._time_s,
            coolant_supply_c=coolant.supply_temp_c,
            coolant_return_c=coolant.return_temp_c,
            case_temp_c=self._case.copy(),
            flow_kgs=coolant.flow_kgs,
            air_inlet_c=self._air.copy(),
            air_outlet_c=air_out,
            rack_power_w=p,
            liquid_power_w=coolant.liquid_power_w,
            air_power_w=coolant.air_power_w,
            util_pct=util.copy(),
            facility_water_c=float(facility_water_c),
            air_supply_c=float(air_supply_c),
            pump_frac=float(pump_frac),
            crac_fan_frac=float(crac_fan_frac),
            cdu_load_kw=coolant.cdu_load_kw,
            cdu_utilisation=coolant.cdu_utilisation,
            cdu_approach_k=coolant.cdu_approach_k,
            provisioning_ratio=phi,
            throttling=bool(np.max(self._case) > self.limits.case_temp_max_c),
        )

    def influence_matrix(self, util_pct, facility_water_c, pump_frac=1.0,
                         delta_w: float = 5e3) -> np.ndarray:
        """(N, N) finite-difference sensitivity of coolant supply to rack power.

        Row j, column i is how much rack j's coolant supply moves when rack i draws
        `delta_w` more. Used to show that the coupling follows the hydraulic topology
        rather than the floor plan.
        """
        util = np.asarray(util_pct, dtype=float)
        rs = rack_power(util, np.full(self.geom.n_racks, facility_water_c + 20.0),
                        self.power)
        base = solve_coolant(self.topo, self.liquid, rs.total_power_w,
                             facility_water_c, pump_frac).supply_temp_c
        out = np.zeros((self.geom.n_racks, self.geom.n_racks))
        for i in range(self.geom.n_racks):
            p = rs.total_power_w.copy()
            p[i] += delta_w
            out[:, i] = solve_coolant(self.topo, self.liquid, p, facility_water_c,
                                      pump_frac).supply_temp_c - base
        return out
