"""Tiny helper to unwrap a saved checkpoint into a plain state_dict.

Training scripts save a dict like {'epoch', 'model', 'optimizer', 'val_loss'};
older scripts used 'model_state_dict' or 'state_dict'; some save the bare
state_dict. This helper tries those keys in order, then strips a common
'module.' / 'model.' prefix if every key has it (DDP/Lightning compat).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch


_STATE_KEYS = ('model_state_dict', 'state_dict', 'model', 'net')


def extract_state_dict(ck: Mapping | torch.nn.Module) -> dict:
    if isinstance(ck, torch.nn.Module):
        return ck.state_dict()
    if isinstance(ck, Mapping):
        for k in _STATE_KEYS:
            if k in ck and isinstance(ck[k], Mapping):
                return _strip_prefix(ck[k])
        # Heuristic: if every value is a Tensor, ck IS the state_dict.
        if ck and all(isinstance(v, torch.Tensor) for v in ck.values()):
            return _strip_prefix(ck)
    raise ValueError(f"Cannot extract state_dict from checkpoint of type {type(ck).__name__}; "
                     f"top-level keys = {list(ck.keys()) if isinstance(ck, Mapping) else 'n/a'}")


def _strip_prefix(sd: Mapping) -> dict:
    """If every key starts with the same prefix in {'module.', 'model.'}, strip it."""
    for pref in ('module.', 'model.'):
        if all(k.startswith(pref) for k in sd):
            return OrderedDict((k[len(pref):], v) for k, v in sd.items())
    return OrderedDict(sd)


def load_ckpt_into(model: torch.nn.Module, ckpt_path: str, *,
                   strict: bool = False, tag: str = '') -> tuple[list[str], list[str]]:
    """Load a checkpoint file into `model`. Returns (missing, unexpected) keys.
    Prints a one-liner summary; use `tag` to label the source in the print.
    """
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = extract_state_dict(ck)
    miss, unexp = model.load_state_dict(sd, strict=strict)
    label = tag or ckpt_path
    print(f'[ckpt] {label} → missing={len(miss)} unexpected={len(unexp)}', flush=True)
    return miss, unexp
