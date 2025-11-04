import argparse
import sys
import os
from consts import CC_LIST, CONFIG_TO_PARAM_DICT, DEFAULT_PARAM_VEC
import numpy as np
import random


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


from os.path import abspath, dirname

cur_dir = dirname(abspath(__file__))
os.chdir(cur_dir)

local_dir = "%s/mix_m3" % (cur_dir)

config_template = """ENABLE_QCN {enable_qcn}
ENABLE_PFC {enable_pfc}

PACKET_PAYLOAD_SIZE 1000

TOPOLOGY_FILE {root}/{topo}.txt
FLOW_FILE {root}/{trace}.txt
TRACE_FILE {local_dir}/trace.txt
TRACE_OUTPUT_FILE {root}/mix_{topo}_{trace}{failure}{config_specs}.tr
FCT_OUTPUT_FILE {root}/fct_{topo}_{trace}{failure}{config_specs}.txt
PFC_OUTPUT_FILE {root}/pfc_{topo}_{trace}{failure}{config_specs}.txt

SIMULATOR_STOP_TIME {duration}

CC_MODE {mode}
ALPHA_RESUME_INTERVAL {t_alpha}
RATE_DECREASE_INTERVAL {t_dec}
CLAMP_TARGET_RATE 0
RP_TIMER {t_inc}
EWMA_GAIN {g}
FAST_RECOVERY_TIMES 1
RATE_AI {ai}Mb/s
RATE_HAI {hai}Mb/s
MIN_RATE 1000Mb/s
MAX_RATE 10000Mb/s
DCTCP_RATE_AI {dctcp_ai}Mb/s
TIMELY_T_HIGH {timely_t_high}
TIMELY_T_LOW {timely_t_low}
TIMELY_BETA {timely_beta}

ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 1
L2_BACK_TO_ZERO 0

HAS_WIN {has_win}
GLOBAL_T 1
VAR_WIN {vwin}
FAST_REACT {us}
U_TARGET {u_tgt}
MI_THRESH {mi}
INT_MULTI {int_multi}
MULTI_RATE 0
SAMPLE_FEEDBACK 0
PINT_LOG_BASE {pint_log_base}
PINT_PROB {pint_prob}

RATE_BOUND 1

ACK_HIGH_PRIO {ack_prio}

LINK_DOWN {link_down}

ENABLE_TRACE {enable_tr}

KMAX_MAP {kmax_map}
KMIN_MAP {kmin_map}
PMAX_MAP {pmax_map}
BUFFER_SIZE {buffer_size}
QLEN_MON_FILE {root}/qlen_{topo}_{trace}{failure}{config_specs}.txt
QLEN_MON_START 1000000000
QLEN_MON_END 3000000000

FIXED_WIN {fwin}
BASE_RTT {base_rtt}
MAX_INFLIGHT_FLOWS {max_inflight_flows}
N_CLIENTS_PER_RACK_FOR_CLOSED_LOOP {n_clients_per_rack_for_closed_loop}
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run simulation")
    parser.add_argument(
        "--cc",
        dest="cc",
        action="store",
        default="hp",
        help="hp/dcqcn/timely/dctcp/hpccPint",
    )
    parser.add_argument(
        "--param_1",
        dest="param_1",
        action="store",
        type=float,
        default=30.0,
        help="CC param 1",
    )
    parser.add_argument(
        "--param_2",
        dest="param_2",
        action="store",
        type=float,
        default=0.0,
        help="CC param 2",
    )
    parser.add_argument(
        "--bfsz",
        dest="bfsz",
        action="store",
        type=float,
        default=300.0,
        help="buffer size",
    )
    parser.add_argument(
        "--fwin",
        dest="fwin",
        action="store",
        type=float,
        default=18000.0,
        help="fixed window size",
    )
    parser.add_argument(
        "--enable_pfc",
        dest="enable_pfc",
        action="store",
        type=float,
        default=1.0,
        help="enabel PFC",
    )
    parser.add_argument(
        "--random_seed",
        dest="random_seed",
        type=int,
        default=0,
        help="random random_seed",
    )
    parser.add_argument(
        "--shard_cc", dest="shard_cc", type=int, default=0, help="random seed"
    )
    parser.add_argument(
        "--max_inflight_flows",
        dest="max_inflight_flows",
        type=int,
        default=0,
        help="max inflgiht flows for close-loop traffic",
    )
    parser.add_argument(
        "--n_clients_per_rack_for_closed_loop",
        dest="n_clients_per_rack_for_closed_loop",
        type=int,
        default=1,
        help="number of clients per rack for closed-loop traffic",
    )
    parser.add_argument(
        "--trace",
        dest="trace",
        action="store",
        default="flow",
        help="the name of the flow file",
    )
    parser.add_argument(
        "--bw", dest="bw", action="store", default="10", help="the NIC bandwidth"
    )
    parser.add_argument(
        "--down", dest="down", action="store", default="0 0 0", help="link down event"
    )
    parser.add_argument(
        "--topo",
        dest="topo",
        action="store",
        default="fat",
        help="the name of the topology file",
    )
    parser.add_argument(
        "--utgt", dest="utgt", action="store", type=int, default=95, help="eta of HPCC"
    )
    parser.add_argument(
        "--mi", dest="mi", action="store", type=int, default=0, help="MI_THRESH"
    )
    parser.add_argument(
        "--hpai", dest="hpai", action="store", type=int, default=50, help="AI for HPCC"
    )
    parser.add_argument(
        "--pint_log_base",
        dest="pint_log_base",
        action="store",
        type=float,
        default=1.05,
        help="PINT's log_base",
    )
    parser.add_argument(
        "--pint_prob",
        dest="pint_prob",
        action="store",
        type=float,
        default=1.0,
        help="PINT's sampling probability",
    )
    parser.add_argument(
        "--enable_tr",
        dest="enable_tr",
        action="store",
        type=int,
        default=0,
        help="enable packet-level events dump",
    )
    parser.add_argument(
        "--enable_debug",
        dest="enable_debug",
        action="store",
        type=int,
        default=0,
        help="enable debug for parameter sample space",
    )
    parser.add_argument(
        "--root",
        dest="root",
        action="store",
        default="mix",
        help="the root directory for configs and results",
    )
    parser.add_argument(
        "--base_rtt",
        dest="base_rtt",
        action="store",
        type=int,
        default=14400,
        help="the base RTT",
    )
    args = parser.parse_args()
    fix_seed(int(args.random_seed))

    seed = int(args.shard_cc)
    enable_debug = args.enable_debug
    enable_tr = args.enable_tr
    max_inflight_flows = args.max_inflight_flows
    n_clients_per_rack_for_closed_loop = args.n_clients_per_rack_for_closed_loop

    root = args.root
    topo = args.topo
    bw = int(args.bw)
    trace = args.trace
    # bfsz = 16 if bw==50 else 32
    # bfsz = int(16 * bw / 50)
    mi = args.mi
    pint_log_base = args.pint_log_base
    pint_prob = args.pint_prob

    bfsz = args.bfsz * 10
    fwin = args.fwin
    base_rtt = args.base_rtt
    enable_pfc = args.enable_pfc
    dctcp_k = 30
    timely_t_low = 10000
    timely_t_high = 50000
    timely_beta = 0.8
    dcqcn_k_min = 10
    dcqcn_k_max = 40
    hpai = 25
    u_tgt = args.utgt / 100.0

    failure = ""
    if args.down != "0 0 0":
        failure = "_down"

    bfsz_idx = CONFIG_TO_PARAM_DICT["bfsz"]
    fwin_idx = CONFIG_TO_PARAM_DICT["fwin"]
    # pfc_idx = CONFIG_TO_PARAM_DICT["pfc"]

    cc = args.cc
    cc_idx = CONFIG_TO_PARAM_DICT["cc"] + CC_LIST.index(cc)
    DEFAULT_PARAM_VEC[cc_idx] = 1.0
    enable_qcn = 1
    if cc == "dctcp":
        cc_idx = CONFIG_TO_PARAM_DICT["dctcp_k"]
        dctcp_k = args.param_1
        DEFAULT_PARAM_VEC[cc_idx] = float(dctcp_k)
    elif cc.startswith("dcqcn"):
        cc_idx = CONFIG_TO_PARAM_DICT["dcqcn_k_min"]
        dcqcn_k_min = args.param_1
        DEFAULT_PARAM_VEC[cc_idx] = float(dcqcn_k_min)

        cc_idx = CONFIG_TO_PARAM_DICT["dcqcn_k_max"]
        dcqcn_k_max = args.param_2
        DEFAULT_PARAM_VEC[cc_idx] = float(dcqcn_k_max)
        enable_pfc = 1
    elif cc.startswith("hp"):
        cc_idx = CONFIG_TO_PARAM_DICT["u_tgt"]
        u_tgt = args.param_1 / 100.0
        DEFAULT_PARAM_VEC[cc_idx] = float(args.param_1)

        cc_idx = CONFIG_TO_PARAM_DICT["hpai"]
        hpai = args.param_2 * 10
        DEFAULT_PARAM_VEC[cc_idx] = float(args.param_2)
    elif cc.startswith("timely"):
        cc_idx = CONFIG_TO_PARAM_DICT["timely_t_low"]
        timely_t_low = args.param_1 * 1000
        DEFAULT_PARAM_VEC[cc_idx] = float(args.param_1)

        cc_idx = CONFIG_TO_PARAM_DICT["timely_t_high"]
        timely_t_high = args.param_2 * 1000
        DEFAULT_PARAM_VEC[cc_idx] = float(args.param_2)

    DEFAULT_PARAM_VEC[bfsz_idx] = float(bfsz / 10.0)
    DEFAULT_PARAM_VEC[fwin_idx] = float(fwin / 1000.0)
    # DEFAULT_PARAM_VEC[pfc_idx] = enable_pfc

    config_specs = ""
    config_name = "%s/config_%s_%s%s%s.txt" % (root, topo, trace, failure, config_specs)

    kmax_map = "3 %d %d %d %d %d %d" % (
        bw * 1000000000,
        dcqcn_k_max * bw / 25,
        bw * 4 * 1000000000,
        dcqcn_k_max * bw * 4 / 25,
        bw * 1000000000 * 10000,
        dcqcn_k_max * bw / 25 * 10000,
    )
    kmin_map = "3 %d %d %d %d %d %d" % (
        bw * 1000000000,
        dcqcn_k_min * bw / 25,
        bw * 4 * 1000000000,
        dcqcn_k_min * bw * 4 / 25,
        bw * 1000000000 * 10000,
        dcqcn_k_min * bw / 25 * 10000,
    )
    pmax_map = "3 %d %.2f %d %.2f %d %.2f" % (
        bw * 1000000000,
        0.2,
        bw * 4 * 1000000000,
        0.2,
        bw * 1000000000 * 10000,
        0.2,
    )

    duration = 600
    with open("%s/%s.txt" % (root, trace), "rb") as f:
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline()
        duration = int(float(last_line.split()[-1]) + 2)

    if args.cc.startswith("dcqcn"):
        ai = 5 * bw / 25
        hai = 50 * bw / 25

        if args.cc == "dcqcn":
            config = config_template.format(
                local_dir=local_dir,
                root=root,
                bw=bw,
                trace=trace,
                topo=topo,
                cc=args.cc,
                mode=1,
                t_alpha=1,
                t_dec=4,
                t_inc=300,
                g=0.00390625,
                ai=ai,
                hai=hai,
                dctcp_ai=1000,
                has_win=0,
                vwin=0,
                us=0,
                u_tgt=u_tgt,
                mi=mi,
                int_multi=1,
                pint_log_base=pint_log_base,
                pint_prob=pint_prob,
                ack_prio=1,
                link_down=args.down,
                failure=failure,
                kmax_map=kmax_map,
                kmin_map=kmin_map,
                pmax_map=pmax_map,
                buffer_size=bfsz,
                enable_tr=enable_tr,
                fwin=fwin,
                base_rtt=base_rtt,
                duration=duration,
                config_specs=config_specs,
                timely_t_high=timely_t_high,
                timely_t_low=timely_t_low,
                timely_beta=timely_beta,
                enable_pfc=enable_pfc,
                enable_qcn=enable_qcn,
                max_inflight_flows=max_inflight_flows,
                n_clients_per_rack_for_closed_loop=n_clients_per_rack_for_closed_loop,
            )
        elif args.cc == "dcqcn_paper":
            config = config_template.format(
                local_dir=local_dir,
                root=root,
                bw=bw,
                trace=trace,
                topo=topo,
                cc=args.cc,
                mode=1,
                t_alpha=50,
                t_dec=50,
                t_inc=55,
                g=0.00390625,
                ai=ai,
                hai=hai,
                dctcp_ai=1000,
                has_win=0,
                vwin=0,
                us=0,
                u_tgt=u_tgt,
                mi=mi,
                int_multi=1,
                pint_log_base=pint_log_base,
                pint_prob=pint_prob,
                ack_prio=1,
                link_down=args.down,
                failure=failure,
                kmax_map=kmax_map,
                kmin_map=kmin_map,
                pmax_map=pmax_map,
                buffer_size=bfsz,
                enable_tr=enable_tr,
                fwin=fwin,
                base_rtt=base_rtt,
                duration=duration,
                config_specs=config_specs,
                timely_t_high=timely_t_high,
                timely_t_low=timely_t_low,
                timely_beta=timely_beta,
                enable_pfc=enable_pfc,
                enable_qcn=enable_qcn,
                max_inflight_flows=max_inflight_flows,
                n_clients_per_rack_for_closed_loop=n_clients_per_rack_for_closed_loop,
            )
        elif args.cc == "dcqcn_vwin":
            config = config_template.format(
                local_dir=local_dir,
                root=root,
                bw=bw,
                trace=trace,
                topo=topo,
                cc=args.cc,
                mode=1,
                t_alpha=1,
                t_dec=4,
                t_inc=300,
                g=0.00390625,
                ai=ai,
                hai=hai,
                dctcp_ai=1000,
                has_win=1,
                vwin=1,
                us=0,
                u_tgt=u_tgt,
                mi=mi,
                int_multi=1,
                pint_log_base=pint_log_base,
                pint_prob=pint_prob,
                ack_prio=0,
                link_down=args.down,
                failure=failure,
                kmax_map=kmax_map,
                kmin_map=kmin_map,
                pmax_map=pmax_map,
                buffer_size=bfsz,
                enable_tr=enable_tr,
                fwin=fwin,
                base_rtt=base_rtt,
                duration=duration,
                config_specs=config_specs,
                timely_t_high=timely_t_high,
                timely_t_low=timely_t_low,
                timely_beta=timely_beta,
                enable_pfc=enable_pfc,
                enable_qcn=enable_qcn,
                max_inflight_flows=max_inflight_flows,
                n_clients_per_rack_for_closed_loop=n_clients_per_rack_for_closed_loop,
            )
        elif args.cc == "dcqcn_paper_vwin":
            config = config_template.format(
                local_dir=local_dir,
                root=root,
                bw=bw,
                trace=trace,
                topo=topo,
                cc=args.cc,
                mode=1,
                t_alpha=50,
                t_dec=50,
                t_inc=55,
                g=0.00390625,
                ai=ai,
                hai=hai,
                dctcp_ai=1000,
                has_win=1,
                vwin=1,
                us=0,
                u_tgt=u_tgt,
                mi=mi,
                int_multi=1,
                pint_log_base=pint_log_base,
                pint_prob=pint_prob,
                ack_prio=0,
                link_down=args.down,
                failure=failure,
                kmax_map=kmax_map,
                kmin_map=kmin_map,
                pmax_map=pmax_map,
                buffer_size=bfsz,
                enable_tr=enable_tr,
                fwin=fwin,
                base_rtt=base_rtt,
                duration=duration,
                config_specs=config_specs,
                timely_t_high=timely_t_high,
                timely_t_low=timely_t_low,
                timely_beta=timely_beta,
                enable_pfc=enable_pfc,
                enable_qcn=enable_qcn,
                max_inflight_flows=max_inflight_flows,
                n_clients_per_rack_for_closed_loop=n_clients_per_rack_for_closed_loop,
            )
    elif args.cc == "hp":
        ai = 10 * bw / 25
        # if args.hpai > 0:
        #     ai = args.hpai
        if hpai > 0:
            ai = hpai
        hai = ai  # useless
        int_multi = max(bw / 25, 1)
        cc = "%s%d" % (args.cc, args.utgt)
        if mi > 0:
            cc += "mi%d" % mi
        # if args.hpai > 0:
        #     cc += "ai%d"%ai
        if hpai > 0:
            cc += "ai%d" % ai
        # config_name = "%s/config_%s_%s_%s%s%s.txt"%(root, topo, trace, cc, failure, config_specs)
        print("cc:", cc)
        config = config_template.format(
            local_dir=local_dir,
            root=root,
            bw=bw,
            trace=trace,
            topo=topo,
            cc=args.cc,
            mode=3,
            t_alpha=1,
            t_dec=4,
            t_inc=300,
            g=0.00390625,
            ai=ai,
            hai=hai,
            dctcp_ai=1000,
            has_win=1,
            vwin=1,
            us=1,
            u_tgt=u_tgt,
            mi=mi,
            int_multi=int_multi,
            pint_log_base=pint_log_base,
            pint_prob=pint_prob,
            ack_prio=0,
            link_down=args.down,
            failure=failure,
            kmax_map=kmax_map,
            kmin_map=kmin_map,
            pmax_map=pmax_map,
            buffer_size=bfsz,
            enable_tr=enable_tr,
            fwin=fwin,
            base_rtt=base_rtt,
            duration=duration,
            config_specs=config_specs,
            timely_t_high=timely_t_high,
            timely_t_low=timely_t_low,
            timely_beta=timely_beta,
            enable_pfc=enable_pfc,
            enable_qcn=enable_qcn,
            max_inflight_flows=max_inflight_flows,
            n_clients_per_rack_for_closed_loop=n_clients_per_rack_for_closed_loop,
        )
    elif args.cc == "dctcp":
        ai = 10  # ai is useless for dctcp
        hai = ai  # also useless
        dctcp_ai = 615  # calculated from RTT=13us and MTU=1KB, because DCTCP add 1 MTU per RTT.
        # masking_threshold_K={dctcp_k} KB (e.g., 30KB)
        kmax_map = "3 %d %d %d %d %d %d" % (
            bw * 1000000000,
            dctcp_k * bw / 10,
            bw * 4 * 1000000000,
            dctcp_k * bw * 4 / 10,
            bw * 1000000000 * 10000,
            dctcp_k * bw / 10 * 10000,
        )
        kmin_map = "3 %d %d %d %d %d %d" % (
            bw * 1000000000,
            dctcp_k * bw / 10,
            bw * 4 * 1000000000,
            dctcp_k * bw * 4 / 10,
            bw * 1000000000 * 10000,
            dctcp_k * bw / 10 * 10000,
        )
        pmax_map = "3 %d %.2f %d %.2f %d %.2f" % (
            bw * 1000000000,
            1.0,
            bw * 4 * 1000000000,
            1.0,
            bw * 1000000000 * 10000,
            1.0,
        )
        config = config_template.format(
            local_dir=local_dir,
            root=root,
            bw=bw,
            trace=trace,
            topo=topo,
            cc=args.cc,
            mode=8,
            t_alpha=1,
            t_dec=4,
            t_inc=300,
            g=0.0625,
            ai=ai,
            hai=hai,
            dctcp_ai=dctcp_ai,
            has_win=1,
            vwin=1,
            us=0,
            u_tgt=u_tgt,
            mi=mi,
            int_multi=1,
            pint_log_base=pint_log_base,
            pint_prob=pint_prob,
            ack_prio=1,
            link_down=args.down,
            failure=failure,
            kmax_map=kmax_map,
            kmin_map=kmin_map,
            pmax_map=pmax_map,
            buffer_size=bfsz,
            enable_tr=enable_tr,
            fwin=fwin,
            base_rtt=base_rtt,
            duration=duration,
            config_specs=config_specs,
            timely_t_high=timely_t_high,
            timely_t_low=timely_t_low,
            timely_beta=timely_beta,
            enable_pfc=enable_pfc,
            enable_qcn=enable_qcn,
            max_inflight_flows=max_inflight_flows,
            n_clients_per_rack_for_closed_loop=n_clients_per_rack_for_closed_loop,
        )
    elif args.cc == "timely":
        ai = 10 * bw / 10
        hai = 50 * bw / 10
        config = config_template.format(
            local_dir=local_dir,
            root=root,
            bw=bw,
            trace=trace,
            topo=topo,
            cc=args.cc,
            mode=7,
            t_alpha=1,
            t_dec=4,
            t_inc=300,
            g=0.00390625,
            ai=ai,
            hai=hai,
            dctcp_ai=1000,
            has_win=0,
            vwin=0,
            us=0,
            u_tgt=u_tgt,
            mi=mi,
            int_multi=1,
            pint_log_base=pint_log_base,
            pint_prob=pint_prob,
            ack_prio=1,
            link_down=args.down,
            failure=failure,
            kmax_map=kmax_map,
            kmin_map=kmin_map,
            pmax_map=pmax_map,
            buffer_size=bfsz,
            enable_tr=enable_tr,
            fwin=fwin,
            base_rtt=base_rtt,
            duration=duration,
            config_specs=config_specs,
            timely_t_high=timely_t_high,
            timely_t_low=timely_t_low,
            timely_beta=timely_beta,
            enable_pfc=enable_pfc,
            enable_qcn=enable_qcn,
            max_inflight_flows=max_inflight_flows,
            n_clients_per_rack_for_closed_loop=n_clients_per_rack_for_closed_loop,
        )
    elif args.cc == "timely_vwin":
        ai = 10 * bw / 10
        hai = 50 * bw / 10
        config = config_template.format(
            local_dir=local_dir,
            root=root,
            bw=bw,
            trace=trace,
            topo=topo,
            cc=args.cc,
            mode=7,
            t_alpha=1,
            t_dec=4,
            t_inc=300,
            g=0.00390625,
            ai=ai,
            hai=hai,
            dctcp_ai=1000,
            has_win=1,
            vwin=1,
            us=0,
            u_tgt=u_tgt,
            mi=mi,
            int_multi=1,
            pint_log_base=pint_log_base,
            pint_prob=pint_prob,
            ack_prio=1,
            link_down=args.down,
            failure=failure,
            kmax_map=kmax_map,
            kmin_map=kmin_map,
            pmax_map=pmax_map,
            buffer_size=bfsz,
            enable_tr=enable_tr,
            fwin=fwin,
            base_rtt=base_rtt,
            duration=duration,
            config_specs=config_specs,
            timely_t_high=timely_t_high,
            timely_t_low=timely_t_low,
            timely_beta=timely_beta,
            enable_pfc=enable_pfc,
            enable_qcn=enable_qcn,
            max_inflight_flows=max_inflight_flows,
            n_clients_per_rack_for_closed_loop=n_clients_per_rack_for_closed_loop,
        )
    elif args.cc == "hpccPint":
        ai = 10 * bw / 25
        # if args.hpai > 0:
        #     ai = args.hpai
        if hpai > 0:
            ai = hpai
        hai = ai  # useless
        int_multi = max(bw / 25, 1)
        cc = "%s%d" % (args.cc, args.utgt)
        if mi > 0:
            cc += "mi%d" % mi
        # if args.hpai > 0:
        #     cc += "ai%d"%ai
        if hpai > 0:
            cc += "ai%d" % ai
        cc += "log%.3f" % pint_log_base
        cc += "p%.3f" % pint_prob
        # config_name = "%s/config_%s_%s_%s%s%s.txt"%(root, topo, trace, cc, failure, config_specs)
        print("cc:", cc)
        config = config_template.format(
            local_dir=local_dir,
            root=root,
            bw=bw,
            trace=trace,
            topo=topo,
            cc=args.cc,
            mode=10,
            t_alpha=1,
            t_dec=4,
            t_inc=300,
            g=0.00390625,
            ai=ai,
            hai=hai,
            dctcp_ai=1000,
            has_win=1,
            vwin=1,
            us=1,
            u_tgt=u_tgt,
            mi=mi,
            int_multi=int_multi,
            pint_log_base=pint_log_base,
            pint_prob=pint_prob,
            ack_prio=0,
            link_down=args.down,
            failure=failure,
            kmax_map=kmax_map,
            kmin_map=kmin_map,
            pmax_map=pmax_map,
            buffer_size=bfsz,
            enable_tr=enable_tr,
            fwin=fwin,
            base_rtt=base_rtt,
            duration=duration,
            config_specs=config_specs,
            timely_t_high=timely_t_high,
            timely_t_low=timely_t_low,
            timely_beta=timely_beta,
            enable_pfc=enable_pfc,
            enable_qcn=enable_qcn,
            max_inflight_flows=max_inflight_flows,
            n_clients_per_rack_for_closed_loop=n_clients_per_rack_for_closed_loop,
        )
    else:
        print("unknown cc:", args.cc)
        sys.exit(1)

    with open(config_name, "w") as file:
        file.write(config)
    with open(
        "%s/param_%s_%s%s%s.txt" % (root, topo, trace, failure, config_specs), "w"
    ) as file:
        file.write(" ".join(map(str, DEFAULT_PARAM_VEC)) + "\n")
        file.write(
            "{} {} {} {} {} {} {} {} {} {}\n".format(
                bfsz,
                fwin,
                # enable_pfc,
                cc,
                dctcp_k,
                dcqcn_k_min,
                dcqcn_k_max,
                u_tgt,
                hpai,
                timely_t_low,
                timely_t_high,
            )
        )
    np.save(
        "%s/param_%s_%s%s%s.npy" % (root, topo, trace, failure, config_specs),
        DEFAULT_PARAM_VEC,
    )
    os.system("./waf --run 'scratch/third %s'" % (config_name))
