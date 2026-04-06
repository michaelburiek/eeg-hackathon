"""
src/models/eeg_conformer.py
────────────────────────────────────────────────────────────────────────────────
EEGConformer baseline — loaded from braindecode and wrapped for training.

EEGConformer reference
----------------------
Song et al., "EEG Conformer: Convolutional Transformer for EEG Decoding
and Visualization" (IEEE TNNLS, 2023).
braindecode: https://braindecode.org/stable/generated/braindecode.models.EEGConformer.html
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class EEGConformerClassifier(nn.Module):
    """
    Thin wrapper around braindecode's EEGConformer.

    If braindecode is not installed, a lightweight fallback is used that
    matches the interface (useful for CI and import testing).

    Parameters
    ----------
    n_channels    : number of EEG channels
    n_times       : number of time samples per window
    n_classes     : number of output classes
    sfreq         : sampling frequency in Hz
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_times: int = 800,
        n_classes: int = 3,
        sfreq: float = 200.0,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
        att_depth: int = 6,
        att_heads: int = 10,
        att_drop_prob: float = 0.5,
        final_fc_length: str | int = "auto",
    ) -> None:
        super().__init__()

        try:
            from braindecode.models import EEGConformer as _EEGConformer
            self._model = _EEGConformer(
                n_outputs=n_classes,
                n_chans=n_channels,
                n_filters_time=n_filters_time,
                filter_time_length=filter_time_length,
                pool_time_length=pool_time_length,
                pool_time_stride=pool_time_stride,
                drop_prob=drop_prob,
                att_depth=att_depth,
                att_heads=att_heads,
                att_drop_prob=att_drop_prob,
                final_fc_length=final_fc_length,
                n_times=n_times,
                sfreq=sfreq,
            )
            log.info("EEGConformer loaded from braindecode.")
        except ImportError:
            log.warning("braindecode not found — using stub EEGConformer.")
            self._model = _FallbackConformer(n_channels, n_times, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)  →  logits: (B, n_classes)"""
        return self._model(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate layer features (before classification head)."""
        # braindecode EEGConformer exposes .final_layer / .classifier
        try:
            # Walk through the model and extract the feature vector
            out = self._model.ensuredims(x) if hasattr(self._model, "ensuredims") else x
            # Use the backbone up to the final FC
            for name, module in self._model.named_children():
                if name in ("fc", "classifier", "final_layer"):
                    break
                out = module(out)
            return out.flatten(1)
        except Exception:  # noqa: BLE001
            # Fallback: just run forward and return logits
            return self.forward(x)


class _FallbackConformer(nn.Module):
    """Minimal CNN + transformer fallback when braindecode is unavailable."""

    def __init__(self, n_channels: int, n_times: int, n_classes: int) -> None:
        super().__init__()
        dim = 64
        self.cnn = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(dim),
            nn.ELU(),
            nn.Conv2d(dim, dim, kernel_size=(n_channels, 1), groups=1),
            nn.BatchNorm2d(dim),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, n_times // 4)),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=128,
            dropout=0.3, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.unsqueeze(1)        # (B, 1, C, T)
        x = self.cnn(x)           # (B, dim, 1, T//4)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T//4, dim)
        x = self.transformer(x)   # (B, T//4, dim)
        x = x.mean(dim=1)         # (B, dim)
        return self.head(x)       # (B, n_classes)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_eegconformer(cfg: dict) -> EEGConformerClassifier:
    """Build EEGConformerClassifier from config dict."""
    m_cfg   = cfg.get("model", {})
    pp_cfg  = cfg.get("preprocessing", {})
    win_cfg = cfg.get("windowing", {})
    ds_cfg  = cfg.get("dataset", {})

    sfreq   = pp_cfg.get("sfreq", 200) or 500
    win_sec = win_cfg.get("window_size_sec", 4.0)
    n_times = int(win_sec * sfreq)
    n_ch    = m_cfg.get("n_channels", 19)
    n_cls   = len(ds_cfg.get("label_map", {"AD": 0, "FTD": 1, "CN": 2}))

    model = EEGConformerClassifier(
        n_channels          = n_ch,
        n_times             = n_times,
        n_classes           = m_cfg.get("n_classes", n_cls),
        sfreq               = float(sfreq),
        n_filters_time      = m_cfg.get("n_filters_time", 40),
        filter_time_length  = m_cfg.get("filter_time_length", 25),
        pool_time_length    = m_cfg.get("pool_time_length", 75),
        pool_time_stride    = m_cfg.get("pool_time_stride", 15),
        drop_prob           = m_cfg.get("drop_prob", 0.5),
        att_depth           = m_cfg.get("att_depth", 6),
        att_heads           = m_cfg.get("att_heads", 10),
        att_drop_prob       = m_cfg.get("att_drop_prob", 0.5),
        final_fc_length     = m_cfg.get("final_fc_length", "auto"),
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("EEGConformer: %.1fM parameters.", n_params)
    return model
