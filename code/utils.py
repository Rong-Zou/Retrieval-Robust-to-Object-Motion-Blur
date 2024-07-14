import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    # manual seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def assure_dir(dir: str):
    if not os.path.isdir(dir):
        os.makedirs(dir)
