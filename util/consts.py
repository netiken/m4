import numpy as np

UNIT_G = 1000000000
UNIT_M = 1000000
UNIT_K = 1000

MTU = UNIT_K
BDP = 10 * MTU

HEADER_SIZE = 48
BYTE_TO_BIT = 8

DELAY_PROPAGATION_BASE = {"link": 1000, "path": 1000}  # 1us

EPS = 1e-12

balance_len_bins = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    15,
    20,
    50,
    100,
    500,
]


def get_base_delay_transmission(sizes, lr_bottleneck):
    return (sizes + np.ceil(sizes / MTU) * HEADER_SIZE) * BYTE_TO_BIT / lr_bottleneck


def get_base_delay_path(sizes, n_links_passed, lr_bottleneck, switch_to_host=1):
    pkt_head = np.clip(sizes, a_min=0, a_max=MTU)
    delay_propagation = DELAY_PROPAGATION_BASE["path"] * n_links_passed
    pkt_size = (pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    delay_transmission = pkt_size / lr_bottleneck + pkt_size / (
        lr_bottleneck * switch_to_host
    ) * (n_links_passed - 2)

    return delay_propagation + delay_transmission
