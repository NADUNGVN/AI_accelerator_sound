"""TCAM1DCNN — Lightweight Channel and Time Attention Enhanced 1D CNN.

Reimplementation of the model from:

    H. Xu, Y. Tian, H. Ren, X. Liu,
    "A Lightweight Channel and Time Attention Enhanced 1D CNN Model for
     Environmental Sound Classification",
    Expert Systems With Applications 249 (2024) 123768.

The network takes a raw audio waveform (length 8000, i.e. 0.5 s @ 16 kHz) and
classifies it into ``num_classes`` categories. It is built from a backbone of
six convolutional modules, each being ``Conv+ReLU`` followed by a Time-Channel
Attention Module (TCAM). A final convolution, global average pooling and a
linear/softmax head produce the class scores (Table 2 of the paper).

Feature-map layout follows the paper: a feature map ``Y`` has ``C`` channels and
``W`` time steps.  In PyTorch ``Conv1d`` convention this is a tensor of shape
``(batch, C, W)`` — channels first, time last.

Attention modules
-----------------
* CAM (Channel Attention Module, Sec. 2.2.1): squeeze over time -> SE-style
  gating (reduction ratio 2) -> channel recalibration + residual.  Eqs. (5)-(8).
* TAM (Time Attention Module, Sec. 2.2.2): project channels to a single time
  sequence -> sigmoid time weights -> recalibrate a conv(Y) by those weights +
  residual.  Eqs. (9)-(12).
* TCAM (Sec. 2.2.3): ``Y_TCAM = F_CAM(F_TAM(Y))`` — TAM first, then CAM. Eq. (13).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# TensorFlow-style "SAME" padding for strided 1D convolutions.
#
# The original model is implemented in Keras with padding="same". For stride > 1
# Keras produces an output length of ceil(L / stride) with (possibly asymmetric)
# zero padding. PyTorch's padding="same" only supports stride == 1, so we pad
# explicitly to reproduce the exact layer output lengths from Table 2.
# --------------------------------------------------------------------------- #
class SamePadConv1d(nn.Module):
    """1D convolution with TensorFlow ``SAME`` padding semantics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        out_length = math.ceil(length / self.stride)
        pad_total = max(
            (out_length - 1) * self.stride + self.kernel_size - length, 0
        )
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_total > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class ChannelAttentionModule(nn.Module):
    """CAM — recalibrate channels using global temporal statistics (Eqs. 5-8).

    ``z`` is obtained by global average pooling over time (Eq. 5). A two-layer
    1x1-conv gate with reduction ratio ``r`` (ReLU then sigmoid, Eq. 6) produces
    per-channel weights ``z'`` which scale ``Y`` (Eq. 7). A residual connection
    yields ``Y_CAM = Y + Y * z'`` (Eq. 8).
    """

    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        # F' and F'' are 1x1 convolutions over the length-1 squeezed vector.
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)   # F'
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)   # F''

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # Squeeze: average over the time dimension W -> (B, C, 1). Eq. (5).
        z = y.mean(dim=2, keepdim=True)
        # Gate: ReLU(F') then sigmoid(F''). Eq. (6).
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))          # z' in [0, 1], shape (B, C, 1)
        m = y * z                               # channel recalibration, Eq. (7)
        return y + m                            # residual, Eq. (8)


class TimeAttentionModule(nn.Module):
    """TAM — recalibrate time steps using cross-channel statistics (Eqs. 9-12).

    A 1x1 conv collapses the ``C`` channels into a single temporal sequence
    ``s`` (Eq. 9); a sigmoid turns it into time weights ``s'`` (Eq. 10). ``Y`` is
    passed through a separate conv ``F_z`` (Conv+ReLU) and scaled by ``s'`` to
    form ``N`` (Eq. 11). A residual connection gives ``Y_TAM = Y + N`` (Eq. 12).

    ``F_z`` uses a kernel size of 3 (not 1x1): mixing neighbouring time steps is
    what "prevents N from over-focusing on s'" per the paper, and it reproduces
    the reported ~406K parameter budget.
    """

    def __init__(self, channels: int, fz_kernel_size: int = 3) -> None:
        super().__init__()
        # F''' : C -> 1 temporal projection used to derive the time weights.
        self.proj = nn.Conv1d(channels, 1, kernel_size=1)
        # F_z : Conv+ReLU recalibration branch, keeps C channels.
        self.fz = SamePadConv1d(channels, channels, kernel_size=fz_kernel_size)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        s = self.proj(y)                        # (B, 1, W), Eq. (9)
        s = torch.sigmoid(s)                    # s' time weights, Eq. (10)
        n = F.relu(self.fz(y)) * s              # broadcast over channels, Eq. (11)
        return y + n                            # residual, Eq. (12)


class TCAM(nn.Module):
    """Time-Channel Attention Module: TAM followed by CAM (Eq. 13)."""

    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        self.tam = TimeAttentionModule(channels)
        self.cam = ChannelAttentionModule(channels, reduction=reduction)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.cam(self.tam(y))


TCAMBlock = TCAM


class ConvTCAMBlock(nn.Module):
    """One backbone module: SAME-padded Conv -> BN -> ReLU -> TCAM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        reduction: int = 2,
    ) -> None:
        super().__init__()
        self.conv = SamePadConv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.tcam = TCAM(out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        return self.tcam(x)


class TCAM1DCNN(nn.Module):
    """The full TCAM-enhanced 1D CNN (Table 2 of the paper).

    Args:
        num_classes: number of output categories (10 for ESC-10 / UrbanSound8K).
        in_channels: input channels of the raw waveform (1).

    Input:  ``(batch, in_channels, 8000)`` raw waveform.
    Output: ``(batch, num_classes)`` logits (apply softmax / cross-entropy).
    """

    # (out_channels, kernel, stride) for the six backbone Conv+TCAM modules,
    # transcribed from Table 2 (layers 1-18). Input channels are chained from
    # the previous module (the first from ``in_channels``).
    _BACKBONE = [
        (32, 32, 1),   # layer 1  : 8000xC_in -> 8000x32
        (32, 16, 2),   # layer 4  : 8000x32   -> 4000x32
        (64, 9, 2),    # layer 7  : 4000x32   -> 2000x64
        (64, 6, 2),    # layer 10 : 2000x64   -> 1000x64
        (128, 3, 5),   # layer 13 : 1000x64   -> 200x128
        (128, 3, 5),   # layer 16 : 200x128   -> 40x128
    ]

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()

        modules = []
        prev_out = in_channels
        for out_c, k, s in self._BACKBONE:
            modules.append(ConvTCAMBlock(prev_out, out_c, k, s))
            prev_out = out_c
        self.backbone = nn.Sequential(*modules)

        # Layer 19: final convolution 3x1, 256 channels, stride 2 (40 -> 20).
        self.head_conv = SamePadConv1d(prev_out, 256, kernel_size=3, stride=2)

        # Global average pooling + linear softmax classifier (Table 2 tail).
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Glorot (Xavier) initialisation, as used in the paper."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)                    # (B, 128, 40)
        # Table 2 attaches ReLU only to the six backbone convs; the final
        # convolution (L19) feeds global average pooling directly.
        x = self.head_conv(x)                   # (B, 256, 20)
        x = x.mean(dim=2)                       # global average pooling -> (B, 256)
        x = self.dropout(x)
        return self.classifier(x)               # (B, num_classes) logits
