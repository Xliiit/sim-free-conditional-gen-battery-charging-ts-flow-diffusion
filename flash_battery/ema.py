"""Exponential moving average wrapper used during training."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA(nn.Module):
    """Maintain EMA weights for the wrapped model."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.model = model
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self._backup = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def update_ema(self) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = self.model.state_dict()
        for key, value in model_state.items():
            if not value.dtype.is_floating_point:
                ema_state[key].copy_(value)
                continue
            ema_state[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    def train(self, mode: bool = True):
        if self.training == mode:
            return super().train(mode)

        if mode:
            self.restore_to_model()
        else:
            self.backup()
            self.copy_to_model()
        return super().train(mode)

    @torch.no_grad()
    def copy_to_model(self) -> None:
        self.model.load_state_dict(self.ema_model.state_dict(), strict=False)

    @torch.no_grad()
    def backup(self) -> None:
        self._backup = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

    @torch.no_grad()
    def restore_to_model(self) -> None:
        if self._backup is None:
            return
        self.model.load_state_dict(self._backup, strict=False)
