# cond_embedder.py
from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEmbedder(nn.Module):
    """
    将 6 维物理条件映射为高维条件向量。

    输入:
        cond: [B, 6]

    输出:
        emb: [B, embed_dim]
    """

    def __init__(
        self,
        in_dim: int = 6,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)
