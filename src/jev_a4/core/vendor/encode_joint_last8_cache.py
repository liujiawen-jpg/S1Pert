"""A4 frozen last-eight joint features: 16 condition + 16 query + 4 evidence.

Absent evidence is masked before layers 24:32. Four masked means of the
contextual result are returned; no task predictions or weights are read.
"""
from __future__ import annotations
from typing import Callable
import numpy as np


def pool_contextual_landmarks(states, mask):
    """Return condition/query/evidence/all masked means as four landmarks."""
    import torch

    if states.ndim != 3 or states.shape[1] != 36:
        raise ValueError(f"expected [batch,36,hidden] states, found {tuple(states.shape)}")
    if mask.shape != states.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("joint attention mask shape/dtype mismatch")

    def masked_mean(left: int, right: int):
        selected = states[:, left:right].float()
        selected_mask = mask[:, left:right]
        denominator = selected_mask.sum(1, keepdim=True)
        numerator = (selected * selected_mask[:, :, None]).sum(1)
        value = numerator / denominator.clamp_min(1).to(numerator.dtype)
        return torch.where(denominator > 0, value, torch.zeros_like(value))

    return torch.stack([
        masked_mean(0, 16), masked_mean(16, 32), masked_mean(32, 36), masked_mean(0, 36)
    ], dim=1)


def _run_tuple_batch(
    tuple_rows: np.ndarray, *, evidence_column: int,
    condition_states: np.ndarray, query_states: np.ndarray, card_states: np.ndarray,
    forward_last8: Callable, device: str, compute_dtype,
) -> np.ndarray:
    import torch

    condition_ids = tuple_rows[:, 0].astype(np.int64, copy=False)
    query_ids = tuple_rows[:, 1].astype(np.int64, copy=False)
    card_ids = tuple_rows[:, evidence_column].astype(np.int64, copy=False)
    host = np.zeros((len(tuple_rows), 36, condition_states.shape[2]), dtype=np.float16)
    host[:, :16] = condition_states[condition_ids]
    host[:, 16:32] = query_states[query_ids]
    valid_card = card_ids >= 0
    if valid_card.any():
        host[valid_card, 32:36] = card_states[card_ids[valid_card]]
    if not np.isfinite(host).all():
        raise FloatingPointError("non-finite block24 state before frozen last8")
    mask = np.ones((len(tuple_rows), 36), dtype=bool)
    mask[~valid_card, 32:36] = False
    states_t = torch.from_numpy(host).to(device=device, dtype=compute_dtype)
    mask_t = torch.from_numpy(mask).to(device=device)
    with torch.inference_mode():
        contextual = forward_last8(states_t, mask_t)
        if not torch.isfinite(contextual).all():
            raise FloatingPointError("non-finite frozen last8 contextual state")
        pooled = pool_contextual_landmarks(contextual, mask_t)
        if not torch.isfinite(pooled).all():
            raise FloatingPointError("non-finite pooled joint landmark")
    result = pooled.float().cpu().numpy().astype(np.float16)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite FP16 joint cache batch")
    return result
