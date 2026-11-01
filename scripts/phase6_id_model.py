"""Phase 6 minimal Stock-ID residual head, independent of a specific backbone.

The base remains the official Time-Series-Library model through p3.output_return.
For sample s: yhat = base(x) + Linear(Embedding(stock_id)).
This is a *static* stock-specific 5-step correction, not a dynamic cross-stock attention layer.
"""
from __future__ import annotations
import torch
from torch import nn

class StockIDResidual(nn.Module):
    def __init__(self, backbone: nn.Module, output_fn, n_stocks: int, embedding_dim: int = 8, horizon: int = 5):
        super().__init__()
        if min(n_stocks, embedding_dim, horizon) < 1:
            raise ValueError("n_stocks, embedding_dim and horizon must be positive")
        self.backbone = backbone
        self.output_fn = output_fn
        self.embedding = nn.Embedding(n_stocks, embedding_dim)
        self.id_head = nn.Linear(embedding_dim, horizon)
        # Initial forecast EXACTLY matches the no-ID backbone, for identical x and base weights.
        nn.init.zeros_(self.id_head.weight)
        nn.init.zeros_(self.id_head.bias)

    def id_correction(self, stock_ids: torch.Tensor) -> torch.Tensor:
        if stock_ids.dtype != torch.long or stock_ids.ndim != 1:
            raise ValueError("stock_ids must be a 1D torch.long tensor")
        if (stock_ids < 0).any().item() or (stock_ids >= self.embedding.num_embeddings).any().item():
            raise ValueError("stock_id outside saved vocabulary")
        return self.id_head(self.embedding(stock_ids))

    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != stock_ids.numel():
            raise ValueError("x and stock_ids batch dimensions differ")
        baseline = self.output_fn(self.backbone, x)
        correction = self.id_correction(stock_ids)
        if baseline.shape != correction.shape:
            raise ValueError(f"Backbone {tuple(baseline.shape)} vs correction {tuple(correction.shape)}")
        return baseline + correction


def stable_vocab(codes: list[str]) -> list[str]:
    vocab = sorted(set(codes))
    if len(vocab) != len(codes):
        raise ValueError("Duplicate security codes in stock vocabulary")
    if not vocab or any(not x or '.' not in x for x in vocab):
        raise ValueError("Invalid stock vocabulary")
    return vocab


def batch_ids(stock_names, indices: torch.Tensor, vocab: list[str], device: torch.device) -> torch.Tensor:
    lookup = {s: i for i, s in enumerate(vocab)}
    names = [str(stock_names[int(k)]) for k in indices.tolist()]
    missing = set(names) - set(lookup)
    if missing:
        raise ValueError(f"Stock ID unseen by model: {sorted(missing)}")
    return torch.tensor([lookup[s] for s in names], dtype=torch.long, device=device)
