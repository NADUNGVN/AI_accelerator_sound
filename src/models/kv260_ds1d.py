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


class MultiScaleStem2dH1(nn.Module):
    """
    Parallel temporal kernels over the raw waveform. The operation is still
    logical 1D convolution because every kernel has height=1.
    """
    def __init__(self, out_channels, kernels=(15, 31, 63), stride=4):
        super().__init__()
        branch_count = len(kernels)
        base = out_channels // branch_count
        widths = [base for _ in kernels]
        widths[-1] += out_channels - sum(widths)
        self.branches = nn.ModuleList(
            [
                ConvBNReLU2dH1(1, width, kernel_size=kernel, stride=stride)
                for width, kernel in zip(widths, kernels)
            ]
        )

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


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
    def __init__(
        self,
        num_classes=10,
        width_mult=1.0,
        dropout=0.15,
        pool_type="avg",
        pool_bins=None,
        stem_type="single",
    ):
        super().__init__()
        pool_type = pool_type.lower()
        if pool_type not in {"avg", "avgmax", "pyramid_avgmax"}:
            raise ValueError(f"Unsupported pool_type '{pool_type}'. Use 'avg', 'avgmax', or 'pyramid_avgmax'.")
        self.pool_type = pool_type
        self.pool_bins = [int(v) for v in (pool_bins or [1, 2, 4])]

        def c(channels):
            return max(8, int(round(channels * width_mult)))

        channels = [c(v) for v in [24, 32, 48, 64, 96, 128, 160]]
        stem_type = stem_type.lower()
        if stem_type == "single":
            self.stem = ConvBNReLU2dH1(1, channels[0], kernel_size=31, stride=4)
        elif stem_type == "multiscale":
            self.stem = MultiScaleStem2dH1(channels[0], kernels=(15, 31, 63), stride=4)
        else:
            raise ValueError(f"Unsupported stem_type '{stem_type}'. Use 'single' or 'multiscale'.")
        self.blocks = nn.Sequential(
            DSBlock2dH1(channels[0], channels[1], kernel_size=15, stride=2),
            DSBlock2dH1(channels[1], channels[2], kernel_size=15, stride=2),
            DSBlock2dH1(channels[2], channels[3], kernel_size=11, stride=2),
            DSBlock2dH1(channels[3], channels[4], kernel_size=9, stride=2),
            DSBlock2dH1(channels[4], channels[5], kernel_size=9, stride=2),
            DSBlock2dH1(channels[5], channels[6], kernel_size=7, stride=2),
            DSBlock2dH1(channels[6], channels[6], kernel_size=15, stride=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1)) if pool_type == "avgmax" else None
        self.dropout = nn.Dropout(dropout)
        if pool_type == "pyramid_avgmax":
            head_features = channels[-1] * 2 * sum(self.pool_bins)
        else:
            head_features = channels[-1] * (2 if pool_type == "avgmax" else 1)
        self.fc = nn.Linear(head_features, num_classes)

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
        if self.pool_type == "pyramid_avgmax":
            pooled = []
            for bin_count in self.pool_bins:
                pooled.append(nn.functional.adaptive_avg_pool2d(x, (1, bin_count)).flatten(1))
                pooled.append(nn.functional.adaptive_max_pool2d(x, (1, bin_count)).flatten(1))
            x = torch.cat(pooled, dim=1)
        elif self.pool_type == "avgmax":
            x = torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)
        else:
            x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class KV260LogMelNetDS1D(nn.Module):
    """
    KV260-oriented temporal 1D-CNN for log-mel inputs.

    The input shape is [B, n_mels, T]. It is reshaped to [B, n_mels, 1, T],
    so every convolution still slides only along time with kernel (1, k).
    Log-mel extraction is intentionally kept outside the model.
    """
    def __init__(
        self,
        num_classes=10,
        input_channels=64,
        width_mult=1.0,
        dropout=0.20,
        pool_type="avgmax",
    ):
        super().__init__()
        pool_type = pool_type.lower()
        if pool_type not in {"avg", "avgmax"}:
            raise ValueError(f"Unsupported pool_type '{pool_type}'. Use 'avg' or 'avgmax'.")
        self.pool_type = pool_type

        def c(channels):
            return max(8, int(round(channels * width_mult)))

        channels = [c(v) for v in [48, 64, 80, 96, 128, 160]]
        self.stem = ConvBNReLU2dH1(input_channels, channels[0], kernel_size=5, stride=1)
        self.blocks = nn.Sequential(
            DSBlock2dH1(channels[0], channels[1], kernel_size=5, stride=2),
            DSBlock2dH1(channels[1], channels[2], kernel_size=5, stride=2),
            DSBlock2dH1(channels[2], channels[3], kernel_size=5, stride=2),
            DSBlock2dH1(channels[3], channels[4], kernel_size=3, stride=2),
            DSBlock2dH1(channels[4], channels[5], kernel_size=3, stride=2),
            DSBlock2dH1(channels[5], channels[5], kernel_size=3, stride=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1)) if pool_type == "avgmax" else None
        self.dropout = nn.Dropout(dropout)
        head_features = channels[-1] * (2 if pool_type == "avgmax" else 1)
        self.fc = nn.Linear(head_features, num_classes)

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
        if self.pool_type == "avgmax":
            x = torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)
        else:
            x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
