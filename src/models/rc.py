"""Fitted RC network: one conductance per graph edge, one capacitance per rack.

Lumped thermal model of the hall. For rack i with capacitance C_i, conductance G_ij to
each neighbour j and G_i,sup to the supply air:

    C_i dT_i/dt  =  P_i  +  sum_j G_ij (T_j - T_i)  +  G_i,sup (T_supply - T_i)

Over a step of length H this is *linear in the parameters*, so a plain least-squares
fit recovers them:

    dT_i  =  a_i P_i  +  sum_j b_ij (T_j - T_i)  +  c_i (T_supply - T_i)

with a_i = H/C_i, b_ij = H G_ij/C_i and c_i = H G_i,sup/C_i. One independent regression
per rack, over the edges the graph gives that rack. Few parameters and physically
interpretable, which is the point of including it.

Two honest caveats, both reported rather than hidden:

*   One parameter set is fitted per horizon, rather than fitting a single-step model
    and rolling it out. That is what makes it comparable with the other baselines,
    which all predict a horizon directly.
*   The parameters are per-rack, so they do not transfer to a hall with different
    racks. RC is a per-hall fit and the transfer table has to say so.
"""

from __future__ import annotations

import numpy as np

from ..graph.build import EDGE_FEATURE_NAMES, HallGraph


class RCNetwork:
    name = "rc"

    def __init__(self, graph: HallGraph, ridge: float = 1e-3):
        self.graph = graph
        self.ridge = ridge
        self.n_racks = graph.n_racks
        self._neighbours = self._rack_neighbours(graph)
        self.coef_: list[np.ndarray] = []
        self._scale: list[np.ndarray] = []

    @staticmethod
    def _rack_neighbours(graph: HallGraph) -> list[np.ndarray]:
        """Incoming rack-to-rack neighbours for each rack, from the graph."""
        rr = graph.edge_attr[:, EDGE_FEATURE_NAMES.index("is_rack_to_rack")] > 0.5
        src, dst = graph.edge_index[0][rr], graph.edge_index[1][rr]
        out = []
        for j in range(graph.n_racks):
            out.append(np.unique(src[dst == j]))
        return out

    def _design(self, rack: int, inlet: np.ndarray, power_kw: np.ndarray,
                supply: np.ndarray) -> np.ndarray:
        """(S, 3 + k) design matrix for one rack, including an intercept.

        The intercept matters. Near equilibrium the driving terms are large and nearly
        cancel -- rack power is around 20 kW while (T_supply - T_i) is around -5 K --
        while the quantity being predicted is a few hundredths of a kelvin. Without a
        constant the fit has to reproduce that cancellation exactly from the two large
        terms, and a small error in either swamps the target. Fitted without one, this
        model scored worse than predicting zero *on its own training data*.
        """
        nb = self._neighbours[rack]
        t_i = inlet[:, rack]
        cols = [np.ones_like(t_i), power_kw[:, rack], supply - t_i]
        if nb.size:
            cols.extend((inlet[:, nb] - t_i[:, None]).T)
        return np.column_stack(cols)

    def fit(self, inlet: np.ndarray, power_kw: np.ndarray, supply: np.ndarray,
            target_delta: np.ndarray) -> "RCNetwork":
        """inlet/power (S, N), supply (S,), target_delta (S, N)."""
        if inlet.shape != power_kw.shape or inlet.shape != target_delta.shape:
            raise ValueError("inlet, power and target must share shape (S, N)")
        if inlet.shape[1] != self.n_racks:
            raise ValueError(f"expected {self.n_racks} racks, got {inlet.shape[1]}")
        self.coef_, self._scale = [], []
        for rack in range(self.n_racks):
            X = self._design(rack, inlet, power_kw, supply)
            # Standardise the non-constant columns before solving. Neighbouring racks
            # move together and the columns differ in scale by orders of magnitude, so
            # the raw normal equations are badly conditioned.
            scale = X.std(axis=0)
            scale[0] = 1.0
            scale = np.where(scale < 1e-9, 1.0, scale)
            Xs = X / scale
            gram = Xs.T @ Xs + self.ridge * np.eye(Xs.shape[1])
            beta = np.linalg.lstsq(gram, Xs.T @ target_delta[:, rack], rcond=None)[0]
            self.coef_.append(beta / scale)
            self._scale.append(scale)
        return self

    def predict_delta(self, inlet: np.ndarray, power_kw: np.ndarray,
                      supply: np.ndarray) -> np.ndarray:
        if not self.coef_:
            raise RuntimeError("call fit() before predict_delta()")
        out = np.empty((inlet.shape[0], self.n_racks))
        for rack in range(self.n_racks):
            out[:, rack] = self._design(rack, inlet, power_kw, supply) @ self.coef_[rack]
        return out

    @property
    def n_parameters(self) -> int:
        return int(sum(c.size for c in self.coef_))
