import os
import random
import numpy as np
import torch

def set_seed(seed=83):
    """
    Ensures reproducibility of experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_dirs():
    """
    Ensures that checkpoints, logs, and results subdirectories exist.
    """
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
