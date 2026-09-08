"""Hall geometry: rack and CRAC placement, aisle structure, return-air paths.

Deterministic by construction. A hall config goes in, fixed arrays come out. No
state, no randomness, no thermal parameters. This is one of the places the brief
flags as where a silent bug would quietly destroy every downstream result, so it is
kept free of anything stochastic and is covered by tests.

Coordinates (metres):
    x   along a rack row
    y   across rows, increasing with row index
    Rack r,c sits at x = (c + 0.5) * rack_width, y = r * row_pitch.

Aisles are indexed by the gap they occupy: aisle k lies between rack rows k and
k + 1, so a hall of R rows has aisles k = -1 .. R-1 (k = -1 is the gap before row 0,
k = R-1 the gap after the last row). Aisle k sits at y = (k + 0.5) * row_pitch.

Two orientation patterns are supported.

`paired` -- the conventional hot-aisle/cold-aisle layout. Rows are grouped into
facing pairs, so aisle parity alternates and every aisle is purely hot or purely
cold:

    aisle -1  HOT    backs of row 0 against the wall
    row 0            inlets face +y
    aisle  0  COLD   rows 0 and 1 face each other
    row 1            inlets face -y
    aisle  1  HOT    rows 1 and 2 exhaust into each other
    row 2            inlets face +y
    ...

`uniform` -- every row faces the same way, the legacy layout with no containment.
Each aisle then takes one row's inlets and the next row's exhaust, so every aisle is
MIXED and separation is poor. Used for hall_c, to make layout transfer a real test
rather than a relabelling.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

AisleType = Literal["cold", "hot", "mixed"]


@dataclasses.dataclass(frozen=True)
class HallGeometry:
    """Fixed spatial description of one data hall.

    Arrays are indexed by flat rack id, assigned row-major: rack id = r * C + c.
    """

    name: str
    n_rows: int
    racks_per_row: int
    orientation_pattern: str

    row: np.ndarray          # (N,) int   rack row index
    col: np.ndarray          # (N,) int   rack column index
    facing: np.ndarray       # (N,) int   +1 if the inlet faces +y, -1 otherwise
    centre: np.ndarray       # (N, 2) float  rack centre
    inlet_xy: np.ndarray     # (N, 2) float  inlet face midpoint
    exhaust_xy: np.ndarray   # (N, 2) float  exhaust face midpoint
    inlet_aisle: np.ndarray  # (N,) int   aisle index the inlet draws from
    exhaust_aisle: np.ndarray  # (N,) int aisle index the exhaust discharges into

    aisle_index: np.ndarray  # (A,) int   aisle indices present, ascending
    aisle_y: np.ndarray      # (A,) float aisle centreline y
    aisle_type: tuple[AisleType, ...]

    crac_xy: np.ndarray      # (M, 2) float CRAC return positions
    crac_flow_share: np.ndarray  # (M,) float  fraction of total supply flow

    # Return-air path, derived from CRAC placement.
    nearest_crac: np.ndarray       # (N,) int   index of the CRAC an exhaust returns to
    dist_to_crac: np.ndarray       # (N,) float metres, exhaust face to that CRAC
    return_dir: np.ndarray         # (N, 2) float unit vector along the return path

    rack_width_m: float
    rack_depth_m: float
    aisle_width_m: float

    @property
    def n_racks(self) -> int:
        return int(self.row.size)

    @property
    def n_cracs(self) -> int:
        return int(self.crac_xy.shape[0])

    @property
    def row_pitch_m(self) -> float:
        return self.rack_depth_m + self.aisle_width_m

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the rack field, excluding CRACs."""
        return (0.0,
                self.racks_per_row * self.rack_width_m,
                -0.5 * self.row_pitch_m,
                (self.n_rows - 0.5) * self.row_pitch_m)


def _facing(n_rows: int, pattern: str) -> np.ndarray:
    """Inlet-facing direction per row: +1 toward +y, -1 toward -y."""
    r = np.arange(n_rows)
    if pattern == "paired":
        # Even rows face +y, odd rows face -y, so rows 2p and 2p+1 face each other.
        return np.where(r % 2 == 0, 1, -1).astype(int)
    if pattern == "uniform":
        return np.ones(n_rows, dtype=int)
    raise ValueError(f"unknown orientation_pattern {pattern!r}; "
                     "expected 'paired' or 'uniform'")


def _aisle_types(aisle_index: np.ndarray, pattern: str) -> tuple[AisleType, ...]:
    if pattern == "paired":
        # Aisle k is bounded by the inlets of rows k, k+1 when k is even.
        return tuple("cold" if k % 2 == 0 else "hot" for k in aisle_index)
    if pattern == "uniform":
        # Every interior aisle takes row k+1's inlets and row k's exhaust.
        return tuple("mixed" for _ in aisle_index)
    raise ValueError(pattern)


def build_hall(cfg: dict) -> HallGeometry:
    """Construct a HallGeometry from a hall config dict (see configs/hall/*.yaml)."""
    n_rows = int(cfg["rows"])
    racks_per_row = int(cfg["racks_per_row"])
    pattern = str(cfg.get("orientation_pattern", "paired"))
    w = float(cfg["rack_width_m"])
    d = float(cfg["rack_depth_m"])
    aw = float(cfg["aisle_width_m"])
    pitch = d + aw

    if n_rows < 2 or racks_per_row < 2:
        raise ValueError("a hall needs at least 2 rows and 2 racks per row")

    row_of_rack = np.repeat(np.arange(n_rows), racks_per_row)
    col_of_rack = np.tile(np.arange(racks_per_row), n_rows)

    facing_by_row = _facing(n_rows, pattern)
    facing = facing_by_row[row_of_rack]

    cx = (col_of_rack + 0.5) * w
    cy = row_of_rack.astype(float) * pitch
    centre = np.stack([cx, cy], axis=1)

    inlet_xy = np.stack([cx, cy + facing * d / 2.0], axis=1)
    exhaust_xy = np.stack([cx, cy - facing * d / 2.0], axis=1)

    # Aisle k lies between rows k and k+1. A row's inlet draws from the aisle on the
    # side it faces: row r facing +1 draws from aisle r, facing -1 from aisle r-1.
    inlet_aisle = np.where(facing > 0, row_of_rack, row_of_rack - 1)
    exhaust_aisle = np.where(facing > 0, row_of_rack - 1, row_of_rack)

    aisle_index = np.arange(-1, n_rows)
    aisle_y = (aisle_index + 0.5) * pitch
    aisle_type = _aisle_types(aisle_index, pattern)

    crac_xy, crac_flow_share = _place_cracs(cfg, racks_per_row, n_rows, pattern,
                                            w, pitch)

    # Return-air path: exhaust travels to the nearest CRAC return. This defines the
    # downstream direction that makes the recirculation kernel asymmetric, and it
    # works for any CRAC layout without hand-assigning CRACs to aisles.
    delta = crac_xy[None, :, :] - exhaust_xy[:, None, :]      # (N, M, 2)
    dist = np.linalg.norm(delta, axis=2)                      # (N, M)
    nearest = np.argmin(dist, axis=1)
    dist_to_crac = dist[np.arange(dist.shape[0]), nearest]
    vec = delta[np.arange(delta.shape[0]), nearest]           # (N, 2)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return_dir = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 0)

    return HallGeometry(
        name=str(cfg.get("name", "hall")),
        n_rows=n_rows,
        racks_per_row=racks_per_row,
        orientation_pattern=pattern,
        row=row_of_rack,
        col=col_of_rack,
        facing=facing,
        centre=centre,
        inlet_xy=inlet_xy,
        exhaust_xy=exhaust_xy,
        inlet_aisle=inlet_aisle,
        exhaust_aisle=exhaust_aisle,
        aisle_index=aisle_index,
        aisle_y=aisle_y,
        aisle_type=aisle_type,
        crac_xy=crac_xy,
        crac_flow_share=crac_flow_share,
        nearest_crac=nearest,
        dist_to_crac=dist_to_crac,
        return_dir=return_dir,
        rack_width_m=w,
        rack_depth_m=d,
        aisle_width_m=aw,
    )


def _place_cracs(cfg: dict, racks_per_row: int, n_rows: int, pattern: str,
                 w: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """CRAC return positions from a hall config.

    Two spellings are accepted per unit.

    `{aisle: k, end: "left"|"right"}` -- the preferred one. Puts the return at the
    end of aisle k, which is where a CRAC actually stands. For the `paired` layout
    the aisle must be a hot one; a cold aisle is rejected rather than quietly
    modelled, because a return drawing from a cold aisle is a configuration error
    and it produces a periodic artefact in the recirculation pattern that is easy
    to mistake for physics.

    `{x_frac: .., y_frac: ..}` -- raw fractional coordinates against the hall
    bounding box, padded by half a row pitch so a unit can sit outside the rack
    field. Kept so hall_c can place returns somewhere deliberately unusual.
    """
    units = cfg.get("crac_units")
    if not units:
        raise ValueError("hall config must list at least one entry under 'crac_units'")

    x_lo, x_hi = 0.0, racks_per_row * w
    y_lo, y_hi = -0.5 * pitch, (n_rows - 0.5) * pitch
    pad_x = 0.5 * pitch

    aisle_index = np.arange(-1, n_rows)
    aisle_type = _aisle_types(aisle_index, pattern)

    xs, ys, shares = [], [], []
    for n, u in enumerate(units):
        if "aisle" in u:
            k = int(u["aisle"])
            if k not in set(int(a) for a in aisle_index):
                raise ValueError(
                    f"crac_units[{n}]: aisle {k} does not exist; this hall has "
                    f"aisles {aisle_index.min()}..{aisle_index.max()}")
            kind = aisle_type[list(aisle_index).index(k)]
            if kind == "cold":
                raise ValueError(
                    f"crac_units[{n}]: aisle {k} is a cold aisle. A CRAC return "
                    "must draw from a hot aisle; hot aisles here are "
                    f"{[int(a) for a, t in zip(aisle_index, aisle_type) if t == 'hot']}")
            end = str(u.get("end", "left"))
            if end not in ("left", "right"):
                raise ValueError(f"crac_units[{n}]: end must be 'left' or 'right'")
            xs.append(x_lo - pad_x if end == "left" else x_hi + pad_x)
            ys.append((k + 0.5) * pitch)
        elif "x_frac" in u and "y_frac" in u:
            xf, yf = float(u["x_frac"]), float(u["y_frac"])
            xs.append(x_lo - pad_x + xf * (x_hi - x_lo + 2 * pad_x))
            ys.append(y_lo + yf * (y_hi - y_lo))
        else:
            raise ValueError(
                f"crac_units[{n}]: give either 'aisle' (+ optional 'end') or both "
                "'x_frac' and 'y_frac'")
        shares.append(float(u.get("flow_share", 1.0)))

    share = np.asarray(shares, dtype=float)
    if np.any(share <= 0):
        raise ValueError("crac flow_share values must be positive")
    share = share / share.sum()
    return np.stack([np.asarray(xs), np.asarray(ys)], axis=1), share


def rows_crossed(geom: HallGeometry) -> np.ndarray:
    """(N, N) int: rack rows the air must cross from i's exhaust to j's inlet.

    Aisles are one row apart in index, so the count is the aisle-index gap. A value
    of 1 means j's inlet aisle directly abuts i's exhaust aisle across a single row
    of racks -- the over-the-top short circuit, and the dominant recirculation path.
    """
    return np.abs(geom.exhaust_aisle[:, None] - geom.inlet_aisle[None, :]).astype(int)
