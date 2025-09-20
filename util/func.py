import torch
import random
import numpy as np
from scipy.stats import rankdata
import logging


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

def map_percentiles(arr,arr_std):
    assert len(arr) == len(arr_std)
    sorted_arr_std = np.sort(arr_std)
    ranks = rankdata(arr, method='ordinal')-1  # Get the ranks of the elements
    res = sorted_arr_std[ranks]
    return res