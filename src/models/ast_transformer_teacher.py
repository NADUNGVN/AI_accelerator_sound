"""AST Transformer Teacher — spectrogram-patch Transformer for KD only.

Paper name / registry key
-------------------------
* **Paper name:** AST Transformer Teacher
* **Class:** ``ASTTransformerTeacher``
* **Config / tools:** HuggingFace ``MIT/ast-finetuned-audioset-10-10-0.4593``
* **Role:** train-time teacher only; **deploy never ships teacher weights**

Layer character
---------------
Unlike the two no-teacher 1D-CNN students (raw-waveform Conv1d / Conv2D-H1),
AST is a **Transformer over log-mel spectrogram patches** (Gong et al.,
Audio Spectrogram Transformer). Primary computational operators are:

* log-mel feature extraction (external / HF feature extractor)
* linear patch embedding + positional embeddings
* multi-head self-attention blocks (Transformer encoder)
* classification head (AudioSet-pretrained, fine-tuned on US8K)

This module wraps HuggingFace ``AutoModelForAudioClassification`` so the
paper naming matches the layer family used in Track 3 (KD).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class ASTTransformerTeacher(nn.Module):
    """Train-time Audio Spectrogram Transformer teacher.

    Parameters
    ----------
    pretrained_name:
        HuggingFace model id. Default is AudioSet-finetuned AST.
    num_labels:
        Number of target classes (UrbanSound8K = 10).
    hf_model:
        Optional already-constructed HF model (for loading checkpoints).
    """

    DEFAULT_PRETRAINED = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(
        self,
        pretrained_name: str = DEFAULT_PRETRAINED,
        num_labels: int = 10,
        hf_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.pretrained_name = pretrained_name
        self.num_labels = int(num_labels)
        if hf_model is not None:
            self.backbone = hf_model
        else:
            from transformers import AutoModelForAudioClassification

            self.backbone = AutoModelForAudioClassification.from_pretrained(
                pretrained_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True,
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name: str = DEFAULT_PRETRAINED,
        num_labels: int = 10,
        **kwargs: Any,
    ) -> "ASTTransformerTeacher":
        return cls(pretrained_name=pretrained_name, num_labels=num_labels, **kwargs)

    def forward(self, input_values: torch.Tensor, **kwargs: Any):
        """Forward HF-style inputs (preprocessed log-mel / waveform per extractor)."""
        outputs = self.backbone(input_values=input_values, **kwargs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs

    def freeze_backbone(self, n_layers: Optional[int] = None) -> None:
        """Optionally freeze early encoder layers for fine-tuning recipes."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if n_layers is None:
            return
        # Unfreeze last n encoder layers + classifier when present.
        encoder = getattr(getattr(self.backbone, "audio_spectrogram_transformer", None), "encoder", None)
        if encoder is None or not hasattr(encoder, "layer"):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            return
        layers = list(encoder.layer)
        for layer in layers[-int(n_layers) :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        classifier = getattr(self.backbone, "classifier", None)
        if classifier is not None:
            for parameter in classifier.parameters():
                parameter.requires_grad = True


# Short alias used in docs
ASTTeacher = ASTTransformerTeacher
