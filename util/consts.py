import numpy as np
from enum import Enum

PLACEHOLDER = 0.1

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
balance_len_bins_label = [
    "(0,1)",
    "[1,2)",
    "[2,4)",
    "[4,6)",
    "[6,8)",
    "[8,10)",
    "[10,15)",
    "[15,30)",
    "[30,60)",
    "[60,100)",
    "[100,200)",
    "[200,400)",
    "[400,600)",
    "[600,1000)",
    "[1000,2000)",
    "[2000,3000)",
    "[3000,4000)",
    "[4000, $\infty$)",
]

balance_len_bins_list = [200, 500]

balance_len_bins_test = [100]
balance_len_bins_test_label = [
    "(0, 100)",
    "[100,$\infty$)",
]

balance_size_bins = [200000, 500000, 1000000]
balance_size_bins_label = [
    "(0, 200KB)",
    "[200KB,500KB)",
    "[500KB,1MB)",
    "[1MB, $\infty$)",
]


def get_base_delay_transmission(sizes, lr_bottleneck):
    return (sizes + np.ceil(sizes / MTU) * HEADER_SIZE) * BYTE_TO_BIT / lr_bottleneck


def get_base_delay_link(sizes, n_links_passed, lr_bottleneck):
    pkt_head = np.clip(sizes, a_min=0, a_max=MTU)
    delay_propagation = DELAY_PROPAGATION_BASE["link"] * n_links_passed
    pkt_size = (pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    delay_transmission = pkt_size / lr_bottleneck

    return delay_propagation + delay_transmission


def get_base_delay_path(sizes, n_links_passed, lr_bottleneck, switch_to_host=1):
    pkt_head = np.clip(sizes, a_min=0, a_max=MTU)
    delay_propagation = DELAY_PROPAGATION_BASE["path"] * n_links_passed
    pkt_size = (pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    delay_transmission = pkt_size / lr_bottleneck + pkt_size / (
        lr_bottleneck * switch_to_host
    ) * (n_links_passed - 2)

    return delay_propagation + delay_transmission
