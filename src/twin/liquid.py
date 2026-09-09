"""Direct-to-chip liquid cooling: coolant loops, CDUs, and the coupling they induce.

Why this exists
---------------
The air model's Phase 2 result was that neighbouring racks' inlet temperatures
correlate at 0.995, so spatial context adds little to single-rack accuracy. That is a
property of air recirculation: it is diffuse, so every rack in a region shares the same
thermal environment and a rack's own temperature already encodes its neighbours'.

Liquid cooling does not work that way, and the difference is the point.

In a direct-to-chip hall, coolant leaves a Coolant Distribution Unit, runs along a
branch manifold, and is drawn off in parallel by the racks on that branch. Two facts
follow, and neither has an analogue in the air model:

*   **Coupling is topological, not spatial.** Racks on one branch share a supply
    manifold, a flow budget and a CDU. Racks on a different branch, however physically
    close, share almost nothing. Two racks standing side by side in adjacent rows can
    be more thermally independent than two racks at opposite ends of the same row.

*   **Physical adjacency is therefore actively misleading.** A k-nearest-neighbour
    feature set built on Euclidean distance will pull in racks that are hydraulically
    unrelated and miss the ones that matter. A model that carries the loop topology in
    its edges will not. This is the strongest available case for a graph model, and it
    is the case the air twin could not make.

What is modelled
----------------
Heat splits: a fraction `liquid_fraction` of rack power goes to the coolant through
cold plates, the remainder to air, where the existing recirculation model still
applies. Real direct-to-chip deployments capture roughly 75-85%.

Along a branch:

    T_supply,i  = T_loop + manifold_gain * d_i          warming along the manifold
    m_dot,i     = m_design * pump_frac * (1 - droop * d_i)   flow falls with distance
    T_return,i  = T_supply,i + Q_liquid,i / (m_dot,i * c_p)
    T_case,i    = T_supply,i + case_rise * Q_liquid,i

At the CDU, the secondary supply temperature sits above the facility water by an
approach that grows with how hard the unit is working:

    T_loop = T_facility + approach_min * (1 + approach_gain * Q_loop / capacity)

That term is what couples every rack on a loop to every other rack on it, and it is why
the induced graph has block structure by loop rather than by geography.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .geometry import HallGeometry

#: Specific heat and density of 25% propylene glycol / water, the usual secondary fluid.
CP_COOLANT = 3600.0     # J/(kg.K)
RHO_COOLANT = 1030.0    # kg/m^3


@dataclasses.dataclass(frozen=True)
class LiquidParams:
    """Coolant-side parameters. Defaults describe a modern direct-to-chip AI hall."""

    # Share of rack power carried away by the coolant rather than by air. Direct-to-chip
    # cold plates cover the GPUs and CPUs but not the PSUs, DIMMs or switches.
    liquid_fraction: float = 0.80

    # Per-rack design flow and the coolant temperature rise it is sized for.
    design_rack_flow_lpm: float = 90.0     # litres per minute at full pump speed
    design_delta_t_k: float = 12.0

    # Supply manifold: coolant warms and flow droops with distance from the branch feed.
    manifold_gain_k_per_m: float = 0.014
    flow_droop_per_m: float = 0.006
    # Supply-return cross-talk along the branch. The supply and return manifolds run
    # alongside each other, so coolant reaching a rack has been warmed by the return of
    # every rack upstream of it. This is the term that makes coupling *directed* and
    # *ordered* within a branch rather than merely block-constant per CDU: without it,
    # perturbing any rack moves every rack on its loop by exactly the same amount, and
    # a single categorical "which CDU" feature would capture the whole structure.
    manifold_crosstalk: float = 0.16

    # CDU heat exchanger. The approach temperature is what a plate exchanger cannot do
    # better than, and it degrades as the unit approaches its rated duty.
    cdu_approach_min_k: float = 2.0
    cdu_approach_gain: float = 3.0
    cdu_capacity_kw: float = 2500.0

    # Case-to-coolant rise at full rack load and design flow. Expressed this way, not
    # as K per kW of rack power: every accelerator has its own cold plate fed in
    # parallel off the rack manifold, so a rack with more GPUs does not run each GPU
    # hotter. Scaling the rise with total rack power got that wrong and made a dense
    # rack look thermally worse than a sparse one at the same per-device load.
    case_rise_full_load_k: float = 26.0
    design_rack_liquid_kw: float = 65.0
    # A cold plate's convective coefficient falls with flow, so slowing the pumps
    # raises chip temperature even when the coolant supply is unchanged. Without this
    # exponent, pump speed would move the return temperature and nothing a chip cares
    # about, and the cooling controller would have only one real lever instead of two.
    case_flow_exponent: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.liquid_fraction <= 1.0:
            raise ValueError("liquid_fraction must lie in (0, 1]")
        if self.design_rack_flow_lpm <= 0 or self.design_delta_t_k <= 0:
            raise ValueError("design flow and delta-T must be positive")
        if self.cdu_capacity_kw <= 0:
            raise ValueError("cdu_capacity_kw must be positive")
        if self.design_rack_liquid_kw <= 0:
            raise ValueError("design_rack_liquid_kw must be positive")
        if self.cdu_approach_min_k < 0:
            raise ValueError("cdu_approach_min_k must be non-negative")
        if not 0.0 <= self.manifold_crosstalk < 1.0:
            raise ValueError("manifold_crosstalk must lie in [0, 1)")

    @property
    def design_flow_kgs(self) -> float:
        return self.design_rack_flow_lpm / 60.0 * RHO_COOLANT / 1000.0


@dataclasses.dataclass(frozen=True)
class LoopTopology:
    """Which CDU and branch each rack hangs off, and how far along the manifold it sits.

    Built once from the hall config. Deterministic, and the thing the graph's coolant
    edges are derived from.
    """

    cdu_of_rack: np.ndarray        # (N,) int   index of the serving CDU
    branch_of_rack: np.ndarray     # (N,) int   global branch index
    manifold_dist_m: np.ndarray    # (N,) float distance from the branch feed point
    branch_order: np.ndarray       # (N,) int   position along the branch, 0 = first
    n_cdus: int
    n_branches: int
    cdu_capacity_kw: np.ndarray    # (M,) float per-unit rated duty

    @property
    def n_racks(self) -> int:
        return int(self.cdu_of_rack.size)

    def racks_on_branch(self, branch: int) -> np.ndarray:
        """Rack ids on one branch, ordered from the feed point outward."""
        idx = np.flatnonzero(self.branch_of_rack == branch)
        return idx[np.argsort(self.branch_order[idx])]

    def racks_on_cdu(self, cdu: int) -> np.ndarray:
        return np.flatnonzero(self.cdu_of_rack == cdu)


def build_topology(geom: HallGeometry, cdu_units: list[dict],
                   default_capacity_kw: float) -> LoopTopology:
    """Assign racks to CDUs and branches from a hall config.

    Each CDU lists branches; each branch names the rack row it serves and the end of
    that row its manifold is fed from. Racks on a row hang off that manifold in
    parallel, ordered by column from the feed.
    """
    if not cdu_units:
        raise ValueError("a liquid-cooled hall must define at least one CDU")

    n = geom.n_racks
    cdu_of = np.full(n, -1, dtype=int)
    branch_of = np.full(n, -1, dtype=int)
    dist = np.zeros(n, dtype=float)
    order = np.full(n, -1, dtype=int)
    caps = []
    b = 0

    for c, unit in enumerate(cdu_units):
        caps.append(float(unit.get("capacity_kw", default_capacity_kw)))
        branches = unit.get("branches")
        if not branches:
            raise ValueError(f"cdu_units[{c}] defines no branches")
        for spec in branches:
            row = int(spec["row"])
            if not 0 <= row < geom.n_rows:
                raise ValueError(f"cdu_units[{c}]: row {row} is outside the hall "
                                 f"(0..{geom.n_rows - 1})")
            feed = str(spec.get("feed", "left"))
            if feed not in ("left", "right"):
                raise ValueError(f"cdu_units[{c}]: feed must be 'left' or 'right'")
            racks = np.flatnonzero(geom.row == row)
            cols = geom.col[racks]
            # Distance along the manifold from whichever end feeds it.
            pos = cols if feed == "left" else (geom.racks_per_row - 1 - cols)
            seq = np.argsort(pos)
            if np.any(cdu_of[racks] >= 0):
                raise ValueError(f"row {row} is served by more than one CDU branch")
            cdu_of[racks] = c
            branch_of[racks] = b
            dist[racks] = pos * geom.rack_width_m
            order[racks[seq]] = np.arange(racks.size)
            b += 1

    unserved = np.flatnonzero(cdu_of < 0)
    if unserved.size:
        rows = sorted(set(geom.row[unserved].tolist()))
        raise ValueError(f"{unserved.size} rack(s) have no coolant branch; "
                         f"rows {rows} are unassigned")

    return LoopTopology(cdu_of_rack=cdu_of, branch_of_rack=branch_of,
                        manifold_dist_m=dist, branch_order=order,
                        n_cdus=len(cdu_units), n_branches=b,
                        cdu_capacity_kw=np.asarray(caps, dtype=float))


@dataclasses.dataclass(frozen=True)
class CoolantState:
    """Coolant-side state of the hall at one instant. Arrays are (N,) unless noted."""

    supply_temp_c: np.ndarray       # coolant entering each rack -- the prediction target
    return_temp_c: np.ndarray       # coolant leaving each rack
    case_temp_c: np.ndarray         # chip case temperature, what actually throttles
    flow_kgs: np.ndarray            # coolant mass flow through each rack
    liquid_power_w: np.ndarray      # heat carried away by coolant
    air_power_w: np.ndarray         # heat left for the air side
    cdu_load_kw: np.ndarray         # (M,) duty on each CDU
    cdu_approach_k: np.ndarray      # (M,) secondary supply above facility water
    cdu_loop_temp_c: np.ndarray     # (M,) secondary supply temperature
    cdu_utilisation: np.ndarray     # (M,) duty over rated capacity


def solve_coolant(topo: LoopTopology, params: LiquidParams, rack_power_w: np.ndarray,
                  facility_water_c: float, pump_frac: float = 1.0,
                  cdu_derate: np.ndarray | None = None) -> CoolantState:
    """Coolant temperatures for every rack.

    Explicit, no iteration: rack power is given, so loop duty follows, and the CDU
    approach follows from that. `cdu_derate` scales a unit's rated capacity and is the
    hook Phase 4's drift injection uses to fail a CDU.
    """
    p = np.asarray(rack_power_w, dtype=float)
    if p.shape != (topo.n_racks,):
        raise ValueError(f"rack_power_w must have shape ({topo.n_racks},), got {p.shape}")
    if not 0.0 < pump_frac <= 1.0:
        raise ValueError("pump_frac must lie in (0, 1]")

    q_liquid = params.liquid_fraction * p
    q_air = p - q_liquid

    capacity = topo.cdu_capacity_kw.astype(float).copy()
    if cdu_derate is not None:
        capacity = capacity * np.asarray(cdu_derate, dtype=float)
    if np.any(capacity <= 0):
        raise ValueError("every CDU must retain positive capacity")

    # Duty per CDU, and the approach temperature that duty costs.
    load_kw = np.zeros(topo.n_cdus)
    np.add.at(load_kw, topo.cdu_of_rack, q_liquid / 1e3)
    utilisation = load_kw / capacity
    approach = params.cdu_approach_min_k * (1.0 + params.cdu_approach_gain * utilisation)
    loop_temp = facility_water_c + approach

    # Along each branch manifold: flow droops with distance from the feed.
    droop = 1.0 - params.flow_droop_per_m * topo.manifold_dist_m
    flow = params.design_flow_kgs * pump_frac * np.clip(droop, 0.15, 1.0)
    if np.any(flow <= 0):
        raise ValueError("coolant flow fell to zero; check flow_droop_per_m against "
                         "the manifold length")

    # Supply temperature at each rack: the loop temperature, a static gain along the
    # manifold, and cross-talk from the return manifold carrying the heat of every rack
    # upstream on the same branch. The last term is directed -- only upstream racks
    # appear in it -- and it is what gives the coupling structure within a branch.
    upstream_rise = np.zeros(topo.n_racks)
    for b in range(topo.n_branches):
        members = topo.racks_on_branch(b)
        q_cum = np.cumsum(q_liquid[members])
        m_cum = np.cumsum(flow[members])
        # Exclude the rack's own contribution: coolant reaching it has passed the racks
        # before it, not itself.
        q_before = q_cum - q_liquid[members]
        m_before = m_cum - flow[members]
        with np.errstate(divide="ignore", invalid="ignore"):
            rise = np.where(m_before > 0, q_before / (m_before * CP_COOLANT), 0.0)
        upstream_rise[members] = rise

    supply = (loop_temp[topo.cdu_of_rack]
              + params.manifold_gain_k_per_m * topo.manifold_dist_m
              + params.manifold_crosstalk * upstream_rise)

    ret = supply + q_liquid / (flow * CP_COOLANT)
    flow_penalty = (params.design_flow_kgs / flow) ** params.case_flow_exponent
    device_load = (q_liquid / 1e3) / params.design_rack_liquid_kw
    case = supply + params.case_rise_full_load_k * device_load * flow_penalty

    return CoolantState(
        supply_temp_c=supply, return_temp_c=ret, case_temp_c=case, flow_kgs=flow,
        liquid_power_w=q_liquid, air_power_w=q_air,
        cdu_load_kw=load_kw, cdu_approach_k=approach, cdu_loop_temp_c=loop_temp,
        cdu_utilisation=utilisation)


def coupling_blocks(topo: LoopTopology) -> np.ndarray:
    """(N, N) bool: whether two racks share a coolant branch.

    Exposed for analysis rather than used by the model. It is the structure a graph can
    represent and a Euclidean k-nearest-neighbour feature set cannot.
    """
    return topo.branch_of_rack[:, None] == topo.branch_of_rack[None, :]
