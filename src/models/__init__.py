"""Model registry — names reflect layer families used in the paper.

Primary (paper)
---------------
* ``DSConv2DH1PyramidNet`` / ``ds_conv2d_h1_pyramid`` — deployable student
* ``DSRes1DSENet`` / ``ds_res1d_se`` — pure-Conv1D no-teacher baseline
* ``ASTTransformerTeacher`` — spectrogram Transformer teacher (KD only)

Literature baseline
-------------------
* ``TCAMAttn1DNet`` / ``tcam_attn1d`` — Xu et al. TCAM reimplementation

Legacy aliases (configs, checkpoints, older docs) remain exported.
"""

from .tcam_attn1d import (
    TCAMAttn1DNet,
    TCAM1DCNN,
    TimeAttentionModule,
    ChannelAttentionModule,
    TCAMBlock,
)
from .ds_res1d_se import DSRes1DSENet, EfficientAudioCNN1D
from .ds_conv2d_h1_pyramid import (
    DSConv2DH1PyramidNet,
    DSConv2DH1PyramidNetDeep,
    DSConv2DH1LogMelNet,
    KV260AudioNetDS1D,
    KV260AudioNetDS1DDeep,
    KV260LogMelNetDS1D,
)
from .ast_transformer_teacher import ASTTransformerTeacher, ASTTeacher

__all__ = [
    # paper names
    "DSConv2DH1PyramidNet",
    "DSConv2DH1PyramidNetDeep",
    "DSConv2DH1LogMelNet",
    "DSRes1DSENet",
    "ASTTransformerTeacher",
    "ASTTeacher",
    "TCAMAttn1DNet",
    # legacy aliases
    "TCAM1DCNN",
    "TimeAttentionModule",
    "ChannelAttentionModule",
    "TCAMBlock",
    "EfficientAudioCNN1D",
    "KV260AudioNetDS1D",
    "KV260AudioNetDS1DDeep",
    "KV260LogMelNetDS1D",
]
