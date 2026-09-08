"""Rack power and airflow as functions of utilisation and inlet temperature.

Why this is ours rather than SustainDC's
----------------------------------------
SustainDC's power curves (`Rack.compute_instantaneous_pwr_vecd`) are not usable as a
physical model. Two problems, both visible in the source:

*   The CPU power ratio carries a hardcoded `+0.05` temperature slope
    (`(self.m_cpu+0.05)*inlet_temp + self.c_cpu`), i.e. about 5% of full-load power
    per kelvin of inlet temperature. Measured server behaviour is roughly an order of
    magnitude below that. Adopting it would make the twin's dynamics dominated by an
    artefact.
*   The IT-fan curve is scaled by three constants introduced with the comments
    `#1 -> 10`, `#5 -> 5` and `#100 -> 20`, and evaluates to an airflow "ratio" above
    3 at ordinary operating points. It is a fitted fudge, not a fan law.

So we keep the *structure* SustainDC uses -- power rising with both utilisation and
inlet temperature, an idle floor, fan flow rising with both -- and re-parameterise it
with defensible values, exposed in config. Fan power follows a cube law in flow,
which is the actual fan affinity law and is what SustainDC itself uses for its CRAC
fans. See docs/thermal-model-design.md for the provenance of each default.

The temperature dependence matters beyond realism: it closes a feedback loop through
the recirculation model. Hotter inlet air means more rack power and more fan flow,
which changes both P and K, which changes the inlet temperature again.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Dry air at typical hall conditions.
RHO_AIR = 1.19      # kg/m^3
C_AIR = 1006.0      # J/(kg.K)


@dataclasses.dataclass(frozen=True)
class PowerParams:
    """Per-rack power and airflow parameters.

    Defaults describe a GPU-dense rack, since the paper is about AI-heavy halls:
    8 accelerated nodes at 4 kW each, so roughly 32 kW at full load.
    """

    servers_per_rack: int = 8
    server_full_load_w: float = 4000.0
    server_idle_w: float = 800.0

    # Fraction of a server's full-load power added per kelvin of inlet temperature,
    # covering leakage current and internal fan response. SustainDC ships ~0.0517
    # here; that is far outside the measured range and we do not adopt it.
    cpu_temp_coeff_per_k: float = 0.004
    reference_temp_c: float = 20.0

    # Server fan: flow ratio in [fan_min_ratio, 1], power as ratio cubed.
    fan_ref_power_w: float = 250.0
    fan_min_ratio: float = 0.30
    fan_temp_gain_per_k: float = 0.030

    # Rack airflow is sized from the design temperature rise across the rack at full
    # load: f = P / (rho * c_p * dT). 12 K is a standard air-cooled design point.
    design_delta_t_k: float = 12.0

    def __post_init__(self) -> None:
        if self.servers_per_rack < 1:
            raise ValueError("servers_per_rack must be >= 1")
        if self.server_idle_w < 0 or self.server_full_load_w <= self.server_idle_w:
            raise ValueError("require 0 <= server_idle_w < server_full_load_w")
        if not 0.0 < self.fan_min_ratio < 1.0:
            raise ValueError("fan_min_ratio must lie in (0, 1)")
        if self.design_delta_t_k <= 0:
            raise ValueError("design_delta_t_k must be positive")

    @property
    def rack_full_load_w(self) -> float:
        """Rack power at 100% utilisation and reference inlet temperature."""
        return self.servers_per_rack * (self.server_full_load_w + self.fan_ref_power_w)

    @property
    def nominal_airflow_m3s(self) -> float:
        """Rack airflow at full fan speed, sized from the design temperature rise."""
        return self.rack_full_load_w / (RHO_AIR * C_AIR * self.design_delta_t_k)


@dataclasses.dataclass(frozen=True)
class RackState:
    """Per-rack power and airflow at one instant. All arrays are (N,)."""

    it_power_w: np.ndarray      # compute power
    fan_power_w: np.ndarray     # server fan power
    total_power_w: np.ndarray   # it + fan, the P that drives recirculation
    airflow_m3s: np.ndarray     # volumetric flow through the rack
    fan_ratio: np.ndarray       # flow as a fraction of nominal
    k_w_per_k: np.ndarray       # rho * c_p * airflow, the K of the D formulation


def fan_ratio(util_pct: np.ndarray, inlet_temp_c: np.ndarray,
              params: PowerParams) -> np.ndarray:
    """Rack fan flow as a fraction of nominal, in [fan_min_ratio, 1].

    Rises with utilisation (more heat to move) and with inlet temperature (server
    fan controllers ramp when intake air is warm). Saturates at 1.
    """
    u = np.clip(np.asarray(util_pct, dtype=float), 0.0, 100.0) / 100.0
    dt = np.asarray(inlet_temp_c, dtype=float) - params.reference_temp_c
    r = params.fan_min_ratio + (1.0 - params.fan_min_ratio) * u \
        + params.fan_temp_gain_per_k * dt
    return np.clip(r, params.fan_min_ratio, 1.0)


def rack_power(util_pct: np.ndarray, inlet_temp_c: np.ndarray,
               params: PowerParams) -> RackState:
    """Power and airflow for every rack, given utilisation and inlet temperature."""
    util = np.clip(np.asarray(util_pct, dtype=float), 0.0, 100.0)
    temp = np.asarray(inlet_temp_c, dtype=float)
    if util.shape != temp.shape:
        raise ValueError(f"util {util.shape} and inlet temp {temp.shape} must match")

    span = params.server_full_load_w - params.server_idle_w
    temp_term = params.server_full_load_w * params.cpu_temp_coeff_per_k \
        * (temp - params.reference_temp_c)
    per_server = params.server_idle_w + span * (util / 100.0) + temp_term
    # An idle server cannot draw less than its idle power however cold the air is.
    per_server = np.maximum(per_server, params.server_idle_w)
    it_power = params.servers_per_rack * per_server

    ratio = fan_ratio(util, temp, params)
    fan_power = params.servers_per_rack * params.fan_ref_power_w * ratio ** 3

    airflow = params.nominal_airflow_m3s * ratio
    k = RHO_AIR * C_AIR * airflow

    return RackState(
        it_power_w=it_power,
        fan_power_w=fan_power,
        total_power_w=it_power + fan_power,
        airflow_m3s=airflow,
        fan_ratio=ratio,
        k_w_per_k=k,
    )


def crac_supply_flow_m3s(n_racks: int, params: PowerParams,
                         design_provisioning: float, fan_frac: float) -> float:
    """Total CRAC supply flow at a given fan setting.

    Sized so that at full rack load and full CRAC fan speed the hall runs at
    `design_provisioning` times the racks' own airflow demand. Turning the CRAC fans
    down below that margin is what drives the provisioning ratio under 1 and opens up
    recirculation -- the mechanism that gives the cooling controller a real lever.
    """
    if not 0.0 < fan_frac <= 1.0:
        raise ValueError("fan_frac must lie in (0, 1]")
    if design_provisioning <= 0:
        raise ValueError("design_provisioning must be positive")
    return design_provisioning * n_racks * params.nominal_airflow_m3s * fan_frac
