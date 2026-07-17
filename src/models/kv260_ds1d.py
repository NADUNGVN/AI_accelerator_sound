import torch
import torch.nn as nn


class ConvBNReLU2dH1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, kernel_size // 2),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DSBlock2dH1(nn.Module):
    """
    DPU-friendly depthwise-separable temporal block:
    Conv2D-H1 depthwise -> BN -> ReLU -> Conv2D 1x1 -> BN -> ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.depthwise = ConvBNReLU2dH1(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBNReLU2dH1(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class KV260AudioNetDS1D(nn.Module):
    """
    KV260-oriented 1D-CNN represented as Conv2D with height=1.

    Logical input is still a mono 1D waveform [B, 1, T]. The model reshapes it
    internally to [B, 1, 1, T] so every convolution has kernel (1, k), i.e. it
    only slides along time.
    """
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.15):
        super().__init__()

        def c(channels):
            return max(8, int(round(channels * width_mult)))

        channels = [c(v) for v in [24, 32, 48, 64, 96, 128, 160]]
        self.stem = ConvBNReLU2dH1(1, channels[0], kernel_size=31, stride=4)
        self.blocks = nn.Sequential(
            DSBlock2dH1(channels[0], channels[1], kernel_size=15, stride=2),
            DSBlock2dH1(channels[1], channels[2], kernel_size=15, stride=2),
            DSBlock2dH1(channels[2], channels[3], kernel_size=11, stride=2),
            DSBlock2dH1(channels[3], channels[4], kernel_size=9, stride=2),
            DSBlock2dH1(channels[4], channels[5], kernel_size=9, stride=2),
            DSBlock2dH1(channels[5], channels[6], kernel_size=7, stride=2),
            DSBlock2dH1(channels[6], channels[6], kernel_size=15, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(2)
        elif x.dim() != 4:
            raise ValueError(f"Expected input [B,C,T] or [B,C,1,T], got shape {tuple(x.shape)}")

        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
