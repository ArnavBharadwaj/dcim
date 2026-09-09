"""The twin's internal state as a directed graph.

Node types are racks and CRAC units. The brief also lists sensors; the twin exposes no
separate sensor measurements, and inventing readings for nodes that measure nothing
would be fabricating data, so sensor nodes are not instantiated. `NodeType` leaves room
for them if the simulator ever grows real ones.

**Edges are built from the floor plan, never from the twin's own kernel.** An operator
knows rack positions, aisle assignments and where the CRAC returns are; they do not know
the recirculation matrix -- that is the thing being predicted. So edge features are
distance, direction relative to the return-air path, rows of racks crossed and edge
type. Using the kernel weight as an edge feature would hand the model its own answer
and every ablation would be meaningless.

Direction matters and the graph says so. An edge i -> j exists when air leaving i can
reach j, and the reverse edge carries a different direction feature because the return
path is asymmetric. This is what the directed-versus-undirected ablation tests.
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np

from ..twin.geometry import HallGeometry, rows_crossed


class NodeType(enum.IntEnum):
    RACK = 0
    CRAC = 1
    SENSOR = 2      # reserved; not instantiated, see the module docstring


class EdgeType(enum.IntEnum):
    RACK_TO_RACK = 0     # recirculation path through the air
    RACK_TO_CRAC = 1     # exhaust returning to a unit
    CRAC_TO_RACK = 2     # supply air reaching a rack


EDGE_FEATURE_NAMES = (
    "distance_m",
    "cos_to_return_flow",
    "rows_crossed",
    "same_aisle",
    "is_rack_to_rack",
    "is_rack_to_crac",
    "is_crac_to_rack",
)


@dataclasses.dataclass(frozen=True)
class GraphSpec:
    """How much of the floor plan to wire up."""

    radius_m: float = 8.0        # rack-to-rack connection radius
    max_rows_crossed: int = 3    # beyond this the air path is not credible
    max_degree: int = 16         # cap on incoming rack-to-rack edges per node
    crac_supply_racks: int = 24  # racks each CRAC unit feeds directly

    def __post_init__(self) -> None:
        if self.radius_m <= 0 or self.max_degree < 1:
            raise ValueError("radius_m must be positive and max_degree >= 1")
        if self.max_rows_crossed < 1:
            raise ValueError("max_rows_crossed must be >= 1")


@dataclasses.dataclass(frozen=True)
class HallGraph:
    """Static graph for one hall. Built once; node features change per timestep."""

    edge_index: np.ndarray      # (2, E) int64, [source; target]
    edge_attr: np.ndarray       # (E, F) float32
    node_type: np.ndarray       # (V,) int64
    n_racks: int
    n_cracs: int
    rack_nodes: np.ndarray      # (n_racks,) node ids of racks, in rack order
    crac_nodes: np.ndarray      # (n_cracs,) node ids of CRAC units

    @property
    def n_nodes(self) -> int:
        return int(self.node_type.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def edge_dim(self) -> int:
        return int(self.edge_attr.shape[1])


def _edge_features(dist: float, cos_flow: float, crossed: int, same_aisle: bool,
                   etype: EdgeType) -> list[float]:
    return [
        dist,
        cos_flow,
        float(crossed),
        1.0 if same_aisle else 0.0,
        1.0 if etype == EdgeType.RACK_TO_RACK else 0.0,
        1.0 if etype == EdgeType.RACK_TO_CRAC else 0.0,
        1.0 if etype == EdgeType.CRAC_TO_RACK else 0.0,
    ]


def build_graph(geom: HallGeometry, spec: GraphSpec | None = None) -> HallGraph:
    """Wire a hall into a directed graph. Deterministic given geometry and spec."""
    spec = spec or GraphSpec()
    n, m = geom.n_racks, geom.n_cracs
    rack_nodes = np.arange(n)
    crac_nodes = np.arange(n, n + m)
    node_type = np.concatenate([
        np.full(n, int(NodeType.RACK)), np.full(m, int(NodeType.CRAC))]).astype(np.int64)

    # --- rack -> rack, along the air path -------------------------------------
    delta = geom.inlet_xy[None, :, :] - geom.exhaust_xy[:, None, :]   # (i, j, 2)
    dist = np.linalg.norm(delta, axis=2)
    norm = np.linalg.norm(delta, axis=2, keepdims=True)
    unit = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 0)
    cos_flow = np.einsum("ijk,ik->ij", unit, geom.return_dir)
    crossed = rows_crossed(geom)
    same_aisle = geom.exhaust_aisle[:, None] == geom.inlet_aisle[None, :]

    eligible = (dist <= spec.radius_m) & (crossed <= spec.max_rows_crossed)
    np.fill_diagonal(eligible, False)

    src, dst, attr = [], [], []
    for j in range(n):                       # cap by *incoming* degree
        cand = np.flatnonzero(eligible[:, j])
        if cand.size > spec.max_degree:
            cand = cand[np.argsort(dist[cand, j])[: spec.max_degree]]
        for i in cand:
            src.append(int(i))
            dst.append(j)
            attr.append(_edge_features(float(dist[i, j]), float(cos_flow[i, j]),
                                       int(crossed[i, j]), bool(same_aisle[i, j]),
                                       EdgeType.RACK_TO_RACK))

    # --- rack -> CRAC: exhaust returns to the nearest unit ---------------------
    for i in range(n):
        c = int(geom.nearest_crac[i])
        src.append(i)
        dst.append(int(crac_nodes[c]))
        attr.append(_edge_features(float(geom.dist_to_crac[i]), 1.0, 0, False,
                                   EdgeType.RACK_TO_CRAC))

    # --- CRAC -> rack: supply reaches the racks nearest each unit --------------
    d_crac = np.linalg.norm(geom.centre[:, None, :] - geom.crac_xy[None, :, :], axis=2)
    take = min(spec.crac_supply_racks, n)
    for c in range(m):
        for i in np.argsort(d_crac[:, c])[:take]:
            src.append(int(crac_nodes[c]))
            dst.append(int(i))
            attr.append(_edge_features(float(d_crac[i, c]), -1.0, 0, False,
                                       EdgeType.CRAC_TO_RACK))

    if not src:
        raise ValueError("graph has no edges; check radius_m against hall dimensions")

    return HallGraph(
        edge_index=np.stack([np.asarray(src), np.asarray(dst)]).astype(np.int64),
        edge_attr=np.asarray(attr, dtype=np.float32),
        node_type=node_type, n_racks=n, n_cracs=m,
        rack_nodes=rack_nodes, crac_nodes=crac_nodes,
    )


def undirected(graph: HallGraph) -> HallGraph:
    """Symmetrised copy, for the directed-versus-undirected ablation.

    Every edge gets its reverse with the same features, so the model can no longer
    tell which way the air was moving. The direction feature is zeroed rather than
    left in place, since keeping it would smuggle the asymmetry back in through the
    edge attributes.
    """
    ei, ea = graph.edge_index, graph.edge_attr.copy()
    ea[:, EDGE_FEATURE_NAMES.index("cos_to_return_flow")] = 0.0
    rev = np.stack([ei[1], ei[0]])
    return dataclasses.replace(
        graph,
        edge_index=np.concatenate([ei, rev], axis=1),
        edge_attr=np.concatenate([ea, ea], axis=0),
    )


def shuffle_edges(graph: HallGraph, frac: float, seed: int) -> HallGraph:
    """Rewire a fraction of rack-to-rack edges to random targets.

    The brief's fifth ablation: if accuracy barely drops, the graph is decorative.
    Only rack-to-rack edges are touched -- the plant connections are part of the hall's
    definition, not of the learned spatial structure -- and edge features travel with
    the original edge, so the model receives a distance and direction that no longer
    describe the pair they are attached to.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    ei = graph.edge_index.copy()
    is_rr = graph.edge_attr[:, EDGE_FEATURE_NAMES.index("is_rack_to_rack")] > 0.5
    cand = np.flatnonzero(is_rr)
    n_shuf = int(round(frac * cand.size))
    if n_shuf:
        chosen = rng.choice(cand, size=n_shuf, replace=False)
        ei[1, chosen] = rng.integers(0, graph.n_racks, size=n_shuf)
    return dataclasses.replace(graph, edge_index=ei)


# ---------------------------------------------------------------------------
# Liquid cooling
# ---------------------------------------------------------------------------

class LiquidNodeType(enum.IntEnum):
    RACK = 0
    CDU = 1
    CRAC = 2


class LiquidEdgeType(enum.IntEnum):
    MANIFOLD_DOWNSTREAM = 0   # coolant path along a shared branch manifold
    MANIFOLD_UPSTREAM = 1     # the reverse, which is hydraulically not the same thing
    RACK_TO_CDU = 2           # coolant return
    CDU_TO_RACK = 3           # coolant supply
    AIR_RECIRC = 4            # residual air, the share the cold plates did not take


LIQUID_EDGE_FEATURE_NAMES = (
    "distance_m",
    "manifold_offset_m",       # signed position difference along the branch
    "same_branch",
    "same_cdu",
    "is_manifold_downstream",
    "is_manifold_upstream",
    "is_rack_to_cdu",
    "is_cdu_to_rack",
    "is_air_recirc",
)


def build_liquid_graph(geom: HallGeometry, topo, spec: GraphSpec | None = None,
                       air_radius_m: float = 6.0,
                       air_max_degree: int = 8) -> HallGraph:
    """Wire a liquid-cooled hall: coolant topology first, residual air second.

    The coolant edges are the point. In a direct-to-chip hall the racks that share a
    branch manifold share a supply temperature, a flow budget and a CDU; racks on a
    different branch share almost nothing however close they stand. So the edges follow
    the hydraulics, not the floor plan, and two racks side by side in adjacent rows may
    have no coolant edge between them at all.

    That is precisely the structure a Euclidean k-nearest-neighbour feature set cannot
    represent: its neighbours are chosen by distance, which here is close to
    uninformative about coupling. `scripts/liquid_probe.py` measures how far apart the
    two orderings actually are.

    Residual air edges are still included, with a shorter radius and lower degree than
    the air-cooled hall uses, because only about a fifth of the heat leaves by air.
    """
    spec = spec or GraphSpec()
    n = geom.n_racks
    m = int(topo.n_cdus)
    rack_nodes = np.arange(n)
    cdu_nodes = np.arange(n, n + m)
    node_type = np.concatenate([
        np.full(n, int(LiquidNodeType.RACK)),
        np.full(m, int(LiquidNodeType.CDU))]).astype(np.int64)

    def feat(dist, offset, same_branch, same_cdu, etype):
        return [
            dist, offset,
            1.0 if same_branch else 0.0,
            1.0 if same_cdu else 0.0,
            1.0 if etype == LiquidEdgeType.MANIFOLD_DOWNSTREAM else 0.0,
            1.0 if etype == LiquidEdgeType.MANIFOLD_UPSTREAM else 0.0,
            1.0 if etype == LiquidEdgeType.RACK_TO_CDU else 0.0,
            1.0 if etype == LiquidEdgeType.CDU_TO_RACK else 0.0,
            1.0 if etype == LiquidEdgeType.AIR_RECIRC else 0.0,
        ]

    src, dst, attr = [], [], []
    centre = geom.centre
    dist_xy = np.linalg.norm(centre[:, None, :] - centre[None, :, :], axis=2)

    # --- coolant: every ordered pair on a shared branch, both directions -------
    # Direction matters hydraulically: a rack nearer the feed sees coolant that has not
    # yet passed its downstream neighbours, so the two directions carry different
    # features and the ablation on edge direction is a real question here.
    for b in range(int(topo.n_branches)):
        members = topo.racks_on_branch(b)
        for a_i in members:
            for b_j in members:
                if a_i == b_j:
                    continue
                offset = float(topo.manifold_dist_m[b_j] - topo.manifold_dist_m[a_i])
                etype = (LiquidEdgeType.MANIFOLD_DOWNSTREAM if offset > 0
                         else LiquidEdgeType.MANIFOLD_UPSTREAM)
                src.append(int(a_i)); dst.append(int(b_j))
                attr.append(feat(float(dist_xy[a_i, b_j]), offset, True, True, etype))

    # --- coolant: return to and supply from the serving CDU ---------------------
    for i in range(n):
        c = int(topo.cdu_of_rack[i])
        d = float(topo.manifold_dist_m[i])
        src.append(i); dst.append(int(cdu_nodes[c]))
        attr.append(feat(d, d, False, True, LiquidEdgeType.RACK_TO_CDU))
        src.append(int(cdu_nodes[c])); dst.append(i)
        attr.append(feat(d, -d, False, True, LiquidEdgeType.CDU_TO_RACK))

    # --- residual air recirculation --------------------------------------------
    exhaust, inlet = geom.exhaust_xy, geom.inlet_xy
    air_d = np.linalg.norm(inlet[None, :, :] - exhaust[:, None, :], axis=2)
    crossed = rows_crossed(geom)
    eligible = (air_d <= air_radius_m) & (crossed <= spec.max_rows_crossed)
    np.fill_diagonal(eligible, False)
    for j in range(n):
        cand = np.flatnonzero(eligible[:, j])
        if cand.size > air_max_degree:
            cand = cand[np.argsort(air_d[cand, j])[:air_max_degree]]
        for i in cand:
            src.append(int(i)); dst.append(j)
            attr.append(feat(float(air_d[i, j]), 0.0,
                             bool(topo.branch_of_rack[i] == topo.branch_of_rack[j]),
                             bool(topo.cdu_of_rack[i] == topo.cdu_of_rack[j]),
                             LiquidEdgeType.AIR_RECIRC))

    if not src:
        raise ValueError("liquid graph has no edges; check the CDU topology")

    return HallGraph(
        edge_index=np.stack([np.asarray(src), np.asarray(dst)]).astype(np.int64),
        edge_attr=np.asarray(attr, dtype=np.float32),
        node_type=node_type, n_racks=n, n_cracs=m,
        rack_nodes=rack_nodes, crac_nodes=cdu_nodes)
