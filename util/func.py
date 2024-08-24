import torch
import random
import numpy as np
from scipy.stats import rankdata
import logging
import struct


def z_score_normalization(data):
    """
    Perform Z-Score normalization (standardization) on the input data.

    Args:
        data (np.array): The input data array to be normalized.

    Returns:
        normalized_data (np.array): The Z-Score normalized data.
        mean (float): The mean of the input data.
        std (float): The standard deviation of the input data.
    """
    mean = data.mean()
    std = data.std()
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


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
    ranks = rankdata(arr, method="min")  # Get the ranks of the elements
    percentiles = (ranks - 1) / (len(arr) - 1) * 100  # Convert ranks to percentiles
    return percentiles


def map_percentiles(arr, arr_std):
    assert len(arr) == len(arr_std)
    sorted_arr_std = np.sort(arr_std)
    ranks = rankdata(arr, method="ordinal") - 1  # Get the ranks of the elements
    res = sorted_arr_std[ranks]
    return res


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def parse_output_get_input(res, n_flows):
    size_tmp = res.size_tmp
    event_times = np.fromiter(res.event_times, dtype=np.float64, count=2 * n_flows)
    enq_indices = np.fromiter(res.enq_indices, dtype=np.uint, count=n_flows).astype(
        np.int64
    )
    deq_indices = np.fromiter(res.deq_indices, dtype=np.uint, count=n_flows).astype(
        np.int64
    )
    num_active_flows = np.fromiter(
        res.num_active_flows, dtype=np.uint, count=(2 * n_flows - 1)
    ).astype(np.int64)
    weight_scatter_indices = np.fromiter(
        res.weight_scatter_indices, dtype=np.uint, count=size_tmp
    ).astype(np.int64)
    active_flows = np.fromiter(res.active_flows, dtype=np.uint, count=size_tmp).astype(
        np.int64
    )
    return (
        event_times,
        enq_indices,
        deq_indices,
        num_active_flows,
        weight_scatter_indices,
        active_flows,
    )


class fileFilter(logging.Filter):
    def filter(self, record):
        # return (not record.getMessage().startswith("Added")) and (
        #     not record.getMessage().startswith("Rank ")
        # )
        return True


def create_logger(log_name):
    logging.basicConfig(
        # format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        # format="%(asctime)s|%(levelname)s| %(processName)s [%(filename)s:%(lineno)d] %(message)s",
        format="%(asctime)s|%(filename)s:%(lineno)d|%(message)s",
        # datefmt="%Y-%m-%d:%H:%M:%S",
        datefmt="%m-%d:%H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_name, mode="a"), logging.StreamHandler()],
    )
    for handler in logging.root.handlers:
        handler.addFilter(fileFilter())
