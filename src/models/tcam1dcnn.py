"""Legacy import path — use ``tcam_attn1d``."""
from .tcam_attn1d import *  # noqa: F401,F403
from .tcam_attn1d import (
    TCAMAttn1DNet,
    TCAM1DCNN,
    TimeAttentionModule,
    ChannelAttentionModule,
    TCAMBlock,
)
