import torch
import torch.nn as nn
import torchaudio


class LogMelFeatureExtractor(nn.Module):
    """
    Differentiable log-mel preprocessing kept outside the classifier model.

    The classifier remains a 1D-CNN over time. This module converts raw waveform
    batches [B, 1, T] into [B, n_mels, frames] before the model sees them.
    """
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        win_length=None,
        n_mels=64,
        f_min=40.0,
        f_max=None,
        eps=1e-6,
        normalize=True,
    ):
        super().__init__()
        self.eps = float(eps)
        self.normalize = bool(normalize)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(sample_rate),
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length) if win_length is not None else int(n_fft),
            f_min=float(f_min),
            f_max=f_max,
            n_mels=int(n_mels),
            power=2.0,
            center=True,
            normalized=False,
        )

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x[:, 0, :]
        elif x.dim() != 2:
            raise ValueError(f"Expected waveform [B,1,T] or [B,T], got {tuple(x.shape)}")

        mel = self.mel(x)
        log_mel = torch.log(mel.clamp_min(self.eps))
        if self.normalize:
            mean = log_mel.mean(dim=(-2, -1), keepdim=True)
            std = log_mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            log_mel = (log_mel - mean) / std
        return log_mel
