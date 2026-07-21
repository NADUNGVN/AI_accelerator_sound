"""DS-Res1D-SE — pure Conv1D depthwise-separable residual network with SE.

Paper name / registry key
-------------------------
* **Paper name:** DS-Res1D-SE Network
* **Class:** ``DSRes1DSENet``
* **Config ``model_name``:** ``ds_res1d_se``
* **Legacy aliases:** ``EfficientAudioCNN1D``, ``efficient_audio_cnn1d``

Layer character
---------------
Native **Conv1d** stack (not Conv2D-H1): multi-scale stem, depthwise-separable
residual blocks with Squeeze-and-Excitation, global avg+max pooling head.
Same full-clip 4 s input as the DS-Conv2D-H1 student; software-only no-teacher
baseline that does **not** target DPU kernel packing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size // 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x):
        scale = x.mean(dim=-1, keepdim=True)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class DSResBlock(nn.Module):
    """
    Depthwise-separable residual block for low-MAC raw waveform modeling.
    """
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=9, se_reduction=8, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm1d(in_channels)
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcite1d(out_channels, reduction=se_reduction)
        self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.act(self.dw_bn(self.dw(x)))
        y = self.act(self.pw_bn(self.pw(y)))
        y = self.se(y)
        y = self.dropout(y)
        return self.act(y + residual)


class MultiScaleStem1d(nn.Module):
    """
    Three cheap raw-waveform branches. Since input has one channel, larger
    first-layer kernels add little parameter cost but improve event-scale coverage.
    """
    def __init__(self, out_channels=24, stride=4):
        super().__init__()
        branch_channels = out_channels // 3
        remainder = out_channels - branch_channels * 3
        widths = [branch_channels, branch_channels, branch_channels + remainder]
        kernels = [9, 31, 63]
        self.branches = nn.ModuleList(
            [ConvBNAct(1, width, kernel_size=kernel, stride=stride) for width, kernel in zip(widths, kernels)]
        )

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class DSRes1DSENet(nn.Module):
    """
    Low-parameter, low-MAC pure-Conv1D network for 4 s raw UrbanSound8K clips.

    Processes a full 4-second waveform in one pass (no 15-frame SUM loop).
    Layer family: MultiScaleStem1d + DSResBlock (DW/PW + SE + residual).
    """
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.25):
        super().__init__()

        def c(channels):
            return max(8, int(round(channels * width_mult)))

        channels = [c(v) for v in [24, 32, 48, 64, 96, 128, 160]]
        self.stem = MultiScaleStem1d(out_channels=channels[0], stride=4)
        self.blocks = nn.Sequential(
            DSResBlock(channels[0], channels[1], stride=2, kernel_size=15, dropout=dropout * 0.25),
            DSResBlock(channels[1], channels[2], stride=2, kernel_size=15, dropout=dropout * 0.25),
            DSResBlock(channels[2], channels[3], stride=2, kernel_size=11, dropout=dropout * 0.5),
            DSResBlock(channels[3], channels[4], stride=2, kernel_size=9, dropout=dropout * 0.5),
            DSResBlock(channels[4], channels[5], stride=2, kernel_size=9, dropout=dropout),
            DSResBlock(channels[5], channels[6], stride=2, kernel_size=7, dropout=dropout),
            DSResBlock(channels[6], channels[6], stride=1, kernel_size=7, dropout=dropout),
        )
        self.head_bn = nn.BatchNorm1d(channels[-1] * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1] * 2, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        avg_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.head_bn(x)
        x = self.dropout(x)
        return self.fc(x)

# Backward-compatible aliases
EfficientAudioCNN1D = DSRes1DSENet

