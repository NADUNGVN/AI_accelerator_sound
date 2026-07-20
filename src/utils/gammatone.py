"""Gammatone Filterbank Generator for 1D CNN First Layer Initialization.

Ref: Abdoli et al. (2019), "End-to-End Environmental Sound Classification
using a 1D Convolutional Neural Network", Section 2.3 & 3.3.
"""

import math
import torch


def erb(fc: float) -> float:
    """Computes Equivalent Rectangular Bandwidth (ERB) for center frequency fc (Hz).

    Glasberg & Moore (1990) formula.
    """
    return 24.7 * (4.37 * (fc / 1000.0) + 1.0)


def generate_gammatone_filterbank(
    num_filters: int = 64,
    filter_size: int = 512,
    sample_rate: int = 16000,
    f_min: float = 100.0,
    f_max: float = 8000.0,
    order: int = 4,
) -> torch.Tensor:
    """Generates a Gammatone filterbank of shape (num_filters, 1, filter_size).

    Parameters:
        num_filters: Number of bandpass filters (default 64).
        filter_size: Length of impulse response in samples (default 512).
        sample_rate: Sampling frequency in Hz (default 16000).
        f_min: Minimum center frequency in Hz (default 100.0).
        f_max: Maximum center frequency in Hz (default 8000.0).
        order: Filter order (default 4).

    Returns:
        Tensor of shape (num_filters, 1, filter_size) containing normalized impulse responses.
    """
    # Convert f_min and f_max to ERB scale to space center frequencies linearly in ERB
    # ERB_number(f) = 21.4 * log10(4.37 * f/1000 + 1)
    erb_min = 21.4 * math.log10(4.37 * (f_min / 1000.0) + 1.0)
    erb_max = 21.4 * math.log10(4.37 * (f_max / 1000.0) + 1.0)

    erb_space = torch.linspace(erb_min, erb_max, num_filters)
    # Inverse ERB scale to get center frequencies fc
    fc_list = (10.0 ** (erb_space / 21.4) - 1.0) / 0.00437

    t = torch.arange(0, filter_size, dtype=torch.float32) / float(sample_rate)
    filters = torch.zeros(num_filters, 1, filter_size, dtype=torch.float32)

    for idx, fc in enumerate(fc_list):
        fc_val = float(fc)
        b = 1.019 * erb(fc_val)
        # Impulse response g(t) = t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*fc*t)
        envelope = (t ** (order - 1)) * torch.exp(-2.0 * math.pi * b * t)
        carrier = torch.cos(2.0 * math.pi * fc_val * t)
        gt = envelope * carrier

        # Normalize filter to unit energy (L2 norm)
        norm = torch.sqrt(torch.sum(gt ** 2))
        if norm > 1e-8:
            gt = gt / norm

        filters[idx, 0, :] = gt

    return filters
