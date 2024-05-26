import numpy as np

UNIT_G = 1000000000
UNIT_M = 1000000
UNIT_K = 1000

MTU = UNIT_K
BDP = 10 * MTU

HEADER_SIZE = 48
BYTE_TO_BIT = 8

DELAY_PROPAGATION_BASE = 1000  # 1us

SIZE_BUCKET_LIST_OUTPUT = ["(0, MTU)", "(MTU, BDP)", "(BDP, 5BDP)", "(5BDP, $\infty$)"]

LINK_TO_DELAY_DICT={
    3:np.array([0,0,0]),
    5:np.array([0,0,1*DELAY_PROPAGATION_BASE,0,0]),
    7:np.array([0,0,1*DELAY_PROPAGATION_BASE,2*DELAY_PROPAGATION_BASE,1*DELAY_PROPAGATION_BASE,0,0]),
}

def get_base_delay(sizes, n_links_passed, lr_bottleneck,flow_idx_target,flow_idx_nontarget_internal):
    pkt_head = np.clip(sizes, a_min=0, a_max=MTU)
    delay_propagation = DELAY_PROPAGATION_BASE * n_links_passed
    pkt_size=(pkt_head + HEADER_SIZE) * BYTE_TO_BIT
    delay_transmission = np.multiply(pkt_size / lr_bottleneck,flow_idx_target) + pkt_size / (lr_bottleneck*4)*(n_links_passed-2)-np.multiply(pkt_size / lr_bottleneck,flow_idx_nontarget_internal)

    return delay_propagation + delay_transmission

def get_base_delay_transmission(sizes, lr_bottleneck):
    return (sizes + np.ceil(sizes / MTU) * HEADER_SIZE) * BYTE_TO_BIT / lr_bottleneck