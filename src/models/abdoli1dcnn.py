"""Abdoli1DCNN — End-to-End 1D CNN for Environmental Sound Classification.

Faithful reimplementation of:

    Sajjad Abdoli, Patrick Cardinal, Alessandro Lameiras Koerich,
    "End-to-End Environmental Sound Classification using a 1D Convolutional
    Neural Network", Expert Systems With Applications / arXiv:1904.08990v1
    (2019).

Architecture (Table 1, 16,000-sample input @ 16 kHz)
----------------------------------------------------
Four 1D conv layers, two max-pools (after CL1 and CL2 only), then FC 128→64→C.
ReLU after every hidden layer. Softmax is applied outside the model (in the
MSLE / CE loss and at evaluation time).

Batch-norm placement follows the paper text (Sec. 2.2):
    Conv → ReLU → BN → (optional MaxPool)

Variants
--------
* variant="gamma" (16,000G): CL1 = 64 filters, k=512, s=1, initialized with a
  Gammatone filterbank (100 Hz–8 kHz). For the paper's best 89% setup the
  entire CL1 is frozen (non-trainable).
* variant="rand"  (16,000) : CL1 = 16 filters, k=64, s=2, random (Xavier) init.

Parameter counts (verified against Table 2 / Table 3)
------------------------------------------------------
* rand  : 256,538 trainable
* gamma : 550,506 trainable when CL1 weight+bias are frozen
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.gammatone import generate_gammatone_filterbank


class Abdoli1DCNN(nn.Module):
    """End-to-End 1D CNN (Abdoli et al., 2019), Table 1 / Figure 2."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        variant: str = "gamma",
        input_length: int = 16000,
        freeze_gammatone: bool | None = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant.lower()
        self.input_length = input_length

        if self.variant == "gamma":
            # Table 1 row "16,000G"
            out_c1, k1, s1 = 64, 512, 1
            # Paper Sec. 3.3: Gammatone first layer is non-trainable for the
            # best reported setup (89%). Default freeze=True for gamma.
            if freeze_gammatone is None:
                freeze_gammatone = True
        elif self.variant == "rand":
            # Table 1 row "16,000"
            out_c1, k1, s1 = 16, 64, 2
            if freeze_gammatone is None:
                freeze_gammatone = False
        else:
            raise ValueError(f"Unknown variant '{variant}'. Use 'gamma' or 'rand'.")

        self.freeze_gammatone = bool(freeze_gammatone)

        # ---- Convolutional feature extractor (Table 1) --------------------
        # CL1
        self.cl1 = nn.Conv1d(in_channels, out_c1, kernel_size=k1, stride=s1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_c1)
        self.pl1 = nn.MaxPool1d(kernel_size=8, stride=8)

        # CL2
        self.cl2 = nn.Conv1d(out_c1, 32, kernel_size=32, stride=2, bias=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.pl2 = nn.MaxPool1d(kernel_size=8, stride=8)

        # CL3
        self.cl3 = nn.Conv1d(32, 64, kernel_size=16, stride=2, bias=True)
        self.bn3 = nn.BatchNorm1d(64)

        # CL4
        self.cl4 = nn.Conv1d(64, 128, kernel_size=8, stride=2, bias=True)
        self.bn4 = nn.BatchNorm1d(128)

        # Infer flatten dim from a dummy forward (depends on variant + length)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            features = self._forward_features(dummy)
            flatten_dim = features.view(1, -1).shape[1]

        # ---- Classifier (Sec. 2.2): FC 128 → 64 → C, dropout p=0.25 --------
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(64, num_classes)

        self._init_weights()

        if self.variant == "gamma":
            self._init_gammatone(freeze=self.freeze_gammatone)

    def _init_weights(self) -> None:
        """Xavier uniform for Conv / Linear (paper does not specify init for
        non-Gammatone layers; Xavier is a neutral default compatible with ReLU).
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_gammatone(self, freeze: bool = True) -> None:
        """Initialize CL1 with 64 Gammatone impulse responses (Sec. 2.3 / 3.3).

        Paper: "initialized by 64 band-pass Gammatone filters with central
        frequency ranging from 100 Hz to 8 kHz" and for the best setup
        "make this layer non-trainable".
        """
        filters = generate_gammatone_filterbank(
            num_filters=64,
            filter_size=512,
            sample_rate=16000,
            f_min=100.0,
            f_max=8000.0,
        )
        self.cl1.weight.data.copy_(filters)
        if freeze:
            # Freeze the entire first layer (weight + bias) so trainable
            # parameter count matches paper Table 3: 550,506.
            self.cl1.weight.requires_grad = False
            if self.cl1.bias is not None:
                self.cl1.bias.requires_grad = False

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Conv stack with paper BN order: Conv → ReLU → BN → (Pool)."""
        # CL1 + PL1
        x = self.bn1(F.relu(self.cl1(x)))
        x = self.pl1(x)

        # CL2 + PL2
        x = self.bn2(F.relu(self.cl2(x)))
        x = self.pl2(x)

        # CL3
        x = self.bn3(F.relu(self.cl3(x)))

        # CL4
        x = self.bn4(F.relu(self.cl4(x)))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits (softmax applied in loss / evaluation)."""
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.fc3(x)
        return x

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
