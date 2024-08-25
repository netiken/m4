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

DELAY_PROPAGATION_BASE = {"link": 1000, "path": 10000}  # 1us

EPS = 1e-12

# SIZE_BUCKET_LIST_OUTPUT = ["(0, MTU)", "(MTU, BDP)", "(BDP, 5BDP)", "(5BDP, $\infty$)"]

balance_len_bins = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100]
balance_len_bins_label = [
    "(0, 2)",
    "[2,4)",
    "[4,6)",
    "[6,8)",
    "[8,10)",
    "[10,15)",
    "[15,20)",
    "[20,30)",
    "[30,40)",
    "[40,50)",
    "[50,75)",
    "[75,100)",
    "[100, $\infty$)",
]
# balance_len_bins = [2, 4, 8, 16, 32, 64, 128, 256]
# balance_len_bins_label = [
#     "(0, 2)",
#     "[2,4)",
#     "[4,8)",
#     "[8,16)",
#     "[16,32)",
#     "[32,64)",
#     "[64,128)",
#     "[128,256)",
#     "[256, $\infty$)",
# ]

balance_size_bins = [1000, 5000, 10000, 20000, 50000, 200000, 1000000]
balance_size_bins_label = [
    "(0, 1MTU)",
    "[1MTU, 5MTU)",
    "[5MTU, 10MTU)",
    "[10MTU, 20MTU)",
    "[20MTU, 50MTU)",
    "[50MTU, 200MTU)",
    "[200MTU, 1000MTU)",
    "[1000MTU, $\infty$)",
]


class QueueEvent(Enum):
    ARRIVAL_FIRST_PKT = 1
    ARRIVAL_LAST_PKT = 2
    QUEUE_START = 3
    QUEUE_END = 4


# SIZE_BUCKET_LIST_LABEL = [
#     "(0, 0.25MTU)",
#     "(0.25MTU, 0.5MTU)",
#     "(0.5MTU, 0.75MTU)",
#     "(0.75MTU, MTU)",
#     "(MTU, 0.2BDP)",
#     "(0.2BD, 0.5BDP)",
#     "(0.5BDP, 0.75BDP)",
#     "(0.75BDP, BDP)",
#     "(BDP, 5BDP)",
#     "(5BDP, INF)",
# ]

# SIZE_BUCKET_LIST_LABEL_OUTPUT = ["(0, MTU)", "(MTU, BDP)", "(BDP, 5BDP)", "(5BDP, INF)"]

# LINK_TO_DELAY_DICT={
#     3:np.array([0,0,0]),
#     5:np.array([0,0,1*DELAY_PROPAGATION_BASE,0,0]),
#     7:np.array([0,0,1*DELAY_PROPAGATION_BASE,2*DELAY_PROPAGATION_BASE,1*DELAY_PROPAGATION_BASE,0,0]),
# }

P99_PERCENTILE_LIST = np.arange(10, 101, 10)
PERCENTILE_METHOD = "nearest"
N_BACKGROUND = 58
# BDP_DICT = {
#     3: 5 * MTU,
#     5: 10 * MTU,
#     7: 15 * MTU,
# }


def get_base_delay(
    sizes,
    n_links_passed,
    lr_bottleneck,
    flow_idx_target,
    flow_idx_nontarget_internal,
    switch_to_host=4,
):
    pkt_head = np.clip(sizes, a_min=0, a_max=MTU)
    delay_propagation = DELAY_PROPAGATION_BASE * n_links_passed
    pkt_size = (pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    delay_transmission = (
        np.multiply(pkt_size / lr_bottleneck, flow_idx_target)
        + pkt_size / (lr_bottleneck * switch_to_host) * (n_links_passed - 2)
        - np.multiply(pkt_size / lr_bottleneck, flow_idx_nontarget_internal)
    )

    return delay_propagation + delay_transmission


def get_base_delay_transmission(sizes, lr_bottleneck):
    return (sizes + np.ceil(sizes / MTU) * HEADER_SIZE) * BYTE_TO_BIT / lr_bottleneck


# def get_size_bucket_list(mtu, bdp):
#     return np.array(
#         [
#             mtu // 4,
#             mtu // 2,
#             mtu * 3 // 4,
#             mtu,
#             bdp // 5,
#             bdp // 2,
#             bdp * 3 // 4,
#             bdp,
#             5 * bdp,
#         ]
#     )
# def get_size_bucket_list_output(mtu, bdp):
#     return np.array([mtu, bdp, 5 * bdp])


def get_base_delay_pmn(
    sizes,
    n_links_passed,
    lr_bottleneck,
    flow_idx_target,
    flow_idx_nontarget_internal,
    switch_to_host=4,
):
    pkt_head = np.clip(sizes, a_min=0, a_max=MTU)
    delay_propagation = DELAY_PROPAGATION_BASE["link"] * n_links_passed
    pkt_size = (pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    delay_transmission = (
        np.multiply(pkt_size / lr_bottleneck, flow_idx_target)
        + pkt_size / (lr_bottleneck * switch_to_host) * (n_links_passed - 2)
        - np.multiply(pkt_size / lr_bottleneck, flow_idx_nontarget_internal)
    )

    return delay_propagation + delay_transmission


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
    delay_transmission = pkt_size / lr_bottleneck + +pkt_size / (
        lr_bottleneck * switch_to_host
    ) * (n_links_passed - 2)

    return delay_propagation + delay_transmission
