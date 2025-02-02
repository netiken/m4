import torch
import random
import numpy as np
from scipy.stats import rankdata
import logging
import struct


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def create_logger(log_name):
    logging.basicConfig(
        format="%(asctime)s|%(filename)s:%(lineno)d|%(message)s",
        datefmt="%m-%d:%H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_name, mode="a"), logging.StreamHandler()],
    )
    for handler in logging.root.handlers:
        handler.addFilter(fileFilter())
