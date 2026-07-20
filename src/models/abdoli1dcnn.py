"""Abdoli1DCNN — End-to-End 1D CNN for Environmental Sound Classification.

Reimplementation of the model from:

    Sajjad Abdoli, Patrick Cardinal, Alessandro Lameiras Koerich,
    "End-to-End Environmental Sound Classification using a 1D Convolutional Neural Network",
    Expert Systems With Applications / arXiv:1904.08990v1 (2019).

Supports both:
    1. variant="gamma" (16,000G): CL1 initialized with 64 Gammatone filters (k=512, s=1).
    2. variant="rand"  (16,000) : CL1 initialized randomly with 16 filters (k=64, s=2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.gammatone import generate_gammatone_filterbank


class Abdoli1DCNN(nn.Module):
    """End-to-End 1D CNN Architecture (Abdoli et al., 2019)."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        variant: str = "gamma",
        input_length: int = 16000,
        freeze_gammatone: bool = False,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant.lower()
        self.input_length = input_length

        if self.variant == "gamma":
            # CL1: 64 filters, kernel size 512, stride 1
            out_c1 = 64
            k1 = 512
            s1 = 1
        elif self.variant == "rand":
            # CL1: 16 filters, kernel size 64, stride 2
            out_c1 = 16
            k1 = 64
            s1 = 2
        else:
            raise ValueError(f"Unknown variant '{variant}'. Use 'gamma' or 'rand'.")

        # Layer CL1
        self.cl1 = nn.Conv1d(in_channels, out_c1, kernel_size=k1, stride=s1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_c1)
        self.pl1 = nn.MaxPool1d(kernel_size=8, stride=8)

        # Layer CL2
        self.cl2 = nn.Conv1d(out_c1, 32, kernel_size=32, stride=2, bias=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.pl2 = nn.MaxPool1d(kernel_size=8, stride=8)

        # Layer CL3
        self.cl3 = nn.Conv1d(32, 64, kernel_size=16, stride=2, bias=True)
        self.bn3 = nn.BatchNorm1d(64)

        # Layer CL4
        self.cl4 = nn.Conv1d(64, 128, kernel_size=8, stride=2, bias=True)
        self.bn4 = nn.BatchNorm1d(128)

        # Determine flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            features = self._forward_features(dummy)
            flatten_dim = features.view(1, -1).shape[1]

        # Fully Connected Classifier
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(64, num_classes)

        self._init_weights()

        if self.variant == "gamma":
            self._init_gammatone(freeze=freeze_gammatone)

    def _init_weights(self) -> None:
        """Xavier Uniform initialization for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_gammatone(self, freeze: bool = False) -> None:
        """Initializes CL1 weights with 64 Gammatone impulse response filters."""
        filters = generate_gammatone_filterbank(
            num_filters=64,
            filter_size=512,
            sample_rate=16000,
            f_min=100.0,
            f_max=8000.0,
        )
        self.cl1.weight.data.copy_(filters)
        if freeze:
            self.cl1.weight.requires_grad = False

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # CL1 + PL1
        x = F.relu(self.bn1(self.cl1(x)))
        x = self.pl1(x)

        # CL2 + PL2
        x = F.relu(self.bn2(self.cl2(x)))
        x = self.pl2(x)

        # CL3
        x = F.relu(self.bn3(self.cl3(x)))

        # CL4
        x = F.relu(self.bn4(self.cl4(x)))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.fc3(x)
        return x
