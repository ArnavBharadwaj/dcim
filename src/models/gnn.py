"""The graph model: encode, process, decode, with attention over incoming edges.

Design follows the brief.

*   **Nodes** are racks and CRAC units, encoded by type-specific encoders into a shared
    hidden space. Node features carry no identity and no absolute coordinates, so
    nothing that only makes sense in hall_a can travel with the weights.
*   **Edges** are directed and follow the air path. Edge features are distance,
    direction relative to the return flow, rows of racks crossed and edge type, all
    from the floor plan rather than from the twin's kernel.
*   **Attention over incoming edges.** Air flow is asymmetric, so mean aggregation is
    the wrong operator: a rack downstream of a hot neighbour and a rack upstream of the
    same neighbour receive very different amounts of its exhaust, and an aggregation
    that averages its neighbours identically cannot represent that. The attention
    logits see the edge features, so the model can learn to weight by direction.
*   **The target is a temperature delta**, not a temperature. A model predicting
    absolute temperature can score well by learning a constant per-hall offset, which
    is exactly the thing that must not transfer.

`layers`, `aggregation` and `use_edge_features` are constructor arguments because the
Phase 6 ablations vary them; they are not tuning knobs to be swept for the headline
number.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class EdgeAttentionLayer(MessagePassing):
    """One message-passing step with attention over each node's incoming edges."""

    def __init__(self, hidden: int, edge_dim: int, heads: int = 4,
                 aggregation: str = "attention", use_edge_features: bool = True):
        super().__init__(aggr="add", node_dim=0)
        if aggregation not in ("attention", "mean"):
            raise ValueError("aggregation must be 'attention' or 'mean'")
        if hidden % heads:
            raise ValueError(f"hidden {hidden} must be divisible by heads {heads}")
        self.hidden, self.heads = hidden, heads
        self.head_dim = hidden // heads
        self.aggregation = aggregation
        self.use_edge_features = use_edge_features
        if aggregation == "mean":
            self.aggr = "mean"

        in_dim = 2 * hidden + (edge_dim if use_edge_features else 0)
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.attn = nn.Linear(in_dim, heads) if aggregation == "attention" else None
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Residual: three layers of message passing over a dense-ish graph oversmooths
        # quickly without one, which would confound the layer-count ablation.
        return self.norm(x + self.update_mlp(torch.cat([x, out], dim=-1)))

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        parts = [x_i, x_j] + ([edge_attr] if self.use_edge_features else [])
        z = torch.cat(parts, dim=-1)
        msg = self.message_mlp(z)
        if self.aggregation == "mean":
            return msg
        alpha = softmax(self.attn(z), index, ptr, size_i)          # (E, heads)
        msg = msg.view(-1, self.heads, self.head_dim)
        return (msg * alpha.unsqueeze(-1)).reshape(-1, self.hidden)


class ThermalGNN(nn.Module):
    name = "gnn"

    def __init__(self, rack_features: int, crac_features: int, edge_dim: int,
                 hidden: int = 96, layers: int = 3, heads: int = 4,
                 aggregation: str = "attention", use_edge_features: bool = True,
                 dropout: float = 0.05):
        super().__init__()
        self.layers_n = layers
        self.use_edge_features = use_edge_features
        self.rack_encoder = nn.Sequential(
            nn.Linear(rack_features, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.crac_encoder = nn.Sequential(
            nn.Linear(crac_features, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([
            EdgeAttentionLayer(hidden, edge_dim, heads, aggregation, use_edge_features)
            for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, rack_x, crac_x, edge_index, edge_attr):
        """rack_x (B, R, Fr), crac_x (B, C, Fc) -> (B, R) predicted delta.

        The graph is the same for every sample in a batch, so nodes are flattened into
        one big disconnected batch graph by offsetting edge indices, rather than
        rebuilding a PyG Batch each step.
        """
        b, r, _ = rack_x.shape
        c = crac_x.shape[1]
        v = r + c
        h = torch.cat([self.rack_encoder(rack_x), self.crac_encoder(crac_x)], dim=1)
        h = h.reshape(b * v, -1)

        offsets = (torch.arange(b, device=rack_x.device) * v).repeat_interleave(
            edge_index.shape[1])
        ei = edge_index.repeat(1, b) + offsets.unsqueeze(0)
        ea = edge_attr.repeat(b, 1)

        for block in self.blocks:
            h = self.dropout(block(h, ei, ea))
        # Slicing off the CRAC nodes leaves a non-contiguous view, and the loss
        # functions take an internal view of their inputs.
        return self.decoder(h).reshape(b, v)[:, :r].contiguous()

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
