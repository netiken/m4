import torch
import random
import numpy as np
from scipy.stats import rankdata

def fix_seed(seed):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    
def decode_dict(d, encoding_used="utf-8"):
    return {
        k.decode(encoding_used): (
            v.decode(encoding_used) if isinstance(v, bytes) else v
        )
        for k, v in d.items()
    }

def calculate_percentiles(arr):
    ranks = rankdata(arr, method='min')  # Get the ranks of the elements
    percentiles = (ranks - 1) / (len(arr) - 1) * 100  # Convert ranks to percentiles
    return percentiles

def map_percentiles(arr,arr_std):
    assert len(arr) == len(arr_std)
    sorted_arr_std = np.sort(arr_std)
    ranks = rankdata(arr, method='ordinal')-1  # Get the ranks of the elements
    res = sorted_arr_std[ranks]
    return res