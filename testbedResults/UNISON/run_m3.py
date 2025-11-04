import argparse
import sys
import os
from consts import CC_LIST, PARAM_LIST, CONFIG_TO_PARAM_DICT, DEFAULT_PARAM_VEC
import numpy as np
import random
def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
from os.path import abspath, dirname
cur_dir=dirname(abspath(__file__))
os.chdir(cur_dir)

local_dir="%s/mix_m3"%(cur_dir)

config_template="""ENABLE_QCN {enable_qcn}
ENABLE_PFC {enable_pfc}
USE_DYNAMIC_PFC_THRESHOLD 1

PACKET_PAYLOAD_SIZE 1000

TOPOLOGY_FILE {local_dir}/{topo}.txt
FLOW_FILE {root}/{trace}.txt
FLOW_ON_PATH_FILE {root}/{trace}_on_path.txt
FLOW_PATH_MAP_FILE {root}/{trace}_path_map.txt
TRACE_FILE {local_dir}/{trace_track}.txt
TRACE_OUTPUT_FILE {root}/mix_{topo}{failure}{config_specs}.tr
FCT_OUTPUT_FILE {root}/fct_{topo}{failure}{config_specs}.txt
PFC_OUTPUT_FILE {root}/pfc_{topo}{failure}{config_specs}.txt

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
QLEN_MON_FILE {root}/qlen_{topo}{failure}{config_specs}.txt
QLEN_MON_START 1000000000
QLEN_MON_END 3000000000

FIXED_WIN {fwin}
BASE_RTT {base_rtt}
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run simulation')
    parser.add_argument("--shard_total", dest = "shard_total",type=int, default=0, help="random shard_total")
    parser.add_argument("--shard_cc", dest="shard_cc", type=int, default=0, help="random seed")
    parser.add_argument('--trace', dest='trace', action='store', default='flow', help="the name of the flow file")
    parser.add_argument('--bw', dest="bw", action='store', default='10', help="the NIC bandwidth")
    parser.add_argument('--down', dest='down', action='store', default='0 0 0', help="link down event")
    parser.add_argument('--topo', dest='topo', action='store', default='fat', help="the name of the topology file")
    parser.add_argument('--utgt', dest='utgt', action='store', type=int, default=95, help="eta of HPCC")
    parser.add_argument('--mi', dest='mi', action='store', type=int, default=0, help="MI_THRESH")
    parser.add_argument('--hpai', dest='hpai', action='store', type=int, default=50, help="AI for HPCC")
    parser.add_argument('--pint_log_base', dest='pint_log_base', action='store', type=float, default=1.05, help="PINT's log_base")
    parser.add_argument('--pint_prob', dest='pint_prob', action='store', type=float, default=1.0, help="PINT's sampling probability")
    parser.add_argument('--enable_tr', dest='enable_tr', action='store', type=int, default=0, help="enable packet-level events dump")
    parser.add_argument('--enable_debug', dest='enable_debug', action='store', type=int, default=0, help="enable debug for parameter sample space")
    parser.add_argument('--root', dest='root', action='store', default='mix', help="the root directory for configs and results")
    parser.add_argument('--base_rtt', dest='base_rtt', action='store', type=int, default=8000, help="the base RTT")
    args = parser.parse_args()
    
    seed=int(args.shard_cc)
    fix_seed(int(args.shard_total))

    enable_debug=args.enable_debug
    enable_tr = args.enable_tr

    root = args.root
    topo=args.topo
    bw = int(args.bw)
    trace = args.trace
    #bfsz = 16 if bw==50 else 32
    # bfsz = int(16 * bw / 50)
    mi=args.mi
    pint_log_base=args.pint_log_base
    pint_prob = args.pint_prob

    # fwin = args.fwin
    base_rtt = args.base_rtt

    failure = ''
    if args.down != '0 0 0':
        failure = '_down'

    bfsz_idx=CONFIG_TO_PARAM_DICT['bfsz']
    fwin_idx=CONFIG_TO_PARAM_DICT['fwin']
    pfc_idx=CONFIG_TO_PARAM_DICT['pfc']
    if enable_debug:
        bfsz=int(PARAM_LIST[bfsz_idx][seed%2]*PARAM_LIST[bfsz_idx][2])
        fwin=int(PARAM_LIST[fwin_idx][seed%2]*PARAM_LIST[fwin_idx][2])
        enable_pfc=int(PARAM_LIST[pfc_idx][seed%2])
    else:
        bfsz=int(np.random.uniform(PARAM_LIST[bfsz_idx][0],PARAM_LIST[bfsz_idx][1])*PARAM_LIST[bfsz_idx][2])
        fwin=int(np.random.uniform(PARAM_LIST[fwin_idx][0],PARAM_LIST[fwin_idx][1])*PARAM_LIST[fwin_idx][2])
        enable_pfc=int(np.random.choice(PARAM_LIST[pfc_idx],1)[0])
    
    dctcp_k=20
    timely_t_low=10000
    timely_t_high=50000
    timely_beta=0.8
    dcqcn_k_min=10
    dcqcn_k_max=40
    hpai=25
    u_tgt=args.utgt/100.
 
    cc=np.random.choice(CC_LIST,1)[0]
    
    cc_idx=CONFIG_TO_PARAM_DICT["cc"]+CC_LIST.index(cc)
    DEFAULT_PARAM_VEC[cc_idx]=1.0
    args.cc=cc
    enable_qcn=1
    if cc=="dctcp":
        cc_idx=CONFIG_TO_PARAM_DICT['dctcp_k']
        if enable_debug:
            dctcp_k=int(PARAM_LIST[cc_idx][seed%2]*PARAM_LIST[cc_idx][2])
        else:
            dctcp_k=int(np.random.uniform(PARAM_LIST[cc_idx][0],PARAM_LIST[cc_idx][1])*PARAM_LIST[cc_idx][2])
        DEFAULT_PARAM_VEC[cc_idx]=float(dctcp_k)/PARAM_LIST[cc_idx][2]
    elif cc.startswith("dcqcn"):
        cc_idx=CONFIG_TO_PARAM_DICT['dcqcn_k_min']
        if enable_debug:
            dcqcn_k_min=int(PARAM_LIST[cc_idx][seed%2]*PARAM_LIST[cc_idx][2])
        else:
            dcqcn_k_min=int(np.random.uniform(PARAM_LIST[cc_idx][0],PARAM_LIST[cc_idx][1])*PARAM_LIST[cc_idx][2])
        DEFAULT_PARAM_VEC[cc_idx]=float(dcqcn_k_min)/PARAM_LIST[cc_idx][2]
  
        cc_idx=CONFIG_TO_PARAM_DICT['dcqcn_k_max']
        if enable_debug:
            dcqcn_k_max=int(PARAM_LIST[cc_idx][seed%2]*PARAM_LIST[cc_idx][2])
        else:
            dcqcn_k_max=int(np.random.uniform(PARAM_LIST[cc_idx][0],PARAM_LIST[cc_idx][1])*PARAM_LIST[cc_idx][2])
        DEFAULT_PARAM_VEC[cc_idx]=float(dcqcn_k_max)/PARAM_LIST[cc_idx][2]
        enable_pfc=1
    elif cc.startswith("hp"):
        cc_idx=CONFIG_TO_PARAM_DICT['u_tgt']
        if enable_debug:
            u_tgt=(PARAM_LIST[cc_idx][seed%2]*PARAM_LIST[cc_idx][2])
        else:
            u_tgt=(np.random.uniform(PARAM_LIST[cc_idx][0],PARAM_LIST[cc_idx][1])*PARAM_LIST[cc_idx][2])
        DEFAULT_PARAM_VEC[cc_idx]=(u_tgt)/PARAM_LIST[cc_idx][2]

        cc_idx=CONFIG_TO_PARAM_DICT['hpai']
        if enable_debug:
            hpai=int(PARAM_LIST[cc_idx][seed%2]*PARAM_LIST[cc_idx][2])
        else:
            hpai=int(np.random.uniform(PARAM_LIST[cc_idx][0],PARAM_LIST[cc_idx][1])*PARAM_LIST[cc_idx][2])
        DEFAULT_PARAM_VEC[cc_idx]=float(hpai)/PARAM_LIST[cc_idx][2]
    elif cc.startswith("timely"):
        cc_idx=CONFIG_TO_PARAM_DICT['timely_t_low']
        if enable_debug:
            timely_t_low=int(PARAM_LIST[cc_idx][seed%2]*PARAM_LIST[cc_idx][2])
        else:
            timely_t_low=int(np.random.uniform(PARAM_LIST[cc_idx][0],PARAM_LIST[cc_idx][1])*PARAM_LIST[cc_idx][2])
        DEFAULT_PARAM_VEC[cc_idx]=float(timely_t_low)/PARAM_LIST[cc_idx][2]
  
        cc_idx=CONFIG_TO_PARAM_DICT['timely_t_high']
        if enable_debug:
            timely_t_high=int(PARAM_LIST[cc_idx][seed%2]*PARAM_LIST[cc_idx][2])
        else:
            timely_t_high=int(np.random.uniform(PARAM_LIST[cc_idx][0],PARAM_LIST[cc_idx][1])*PARAM_LIST[cc_idx][2])
        DEFAULT_PARAM_VEC[cc_idx]=float(timely_t_high)/PARAM_LIST[cc_idx][2]
      
    DEFAULT_PARAM_VEC[bfsz_idx]=float(bfsz)/PARAM_LIST[bfsz_idx][2]
    DEFAULT_PARAM_VEC[fwin_idx]=float(fwin)/PARAM_LIST[fwin_idx][2]
    DEFAULT_PARAM_VEC[pfc_idx]=enable_pfc

    config_specs="_s%d"%(seed)
    config_name = "%s/config_%s_%s%s%s.txt"%(root, topo, trace, failure, config_specs)

    kmax_map = "2 %d %d %d %d"%(bw*1000000000, dcqcn_k_max*bw/25, bw*4*1000000000, dcqcn_k_max*bw*4/25)
    kmin_map = "2 %d %d %d %d"%(bw*1000000000, dcqcn_k_min*bw/25, bw*4*1000000000, dcqcn_k_min*bw*4/25)
    pmax_map = "2 %d %.2f %d %.2f"%(bw*1000000000, 0.2, bw*4*1000000000, 0.2)

    duration=600
    with open("%s/%s.txt"%(root, trace), 'rb') as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline()
        duration=int(float(last_line.split()[-1])+2)

    if (args.cc.startswith("dcqcn")):
        ai = 5 * bw / 25
        hai = 50 * bw /25

        if args.cc == "dcqcn":
            config = config_template.format(local_dir=local_dir,root=root, bw=bw, trace=trace, topo=topo, trace_track=topo.replace("topo","trace"), cc=args.cc, mode=1, t_alpha=1, t_dec=4, t_inc=300, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=0, vwin=0, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=1, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr, fwin=fwin, base_rtt=base_rtt,duration=duration,config_specs=config_specs,timely_t_high=timely_t_high,timely_t_low=timely_t_low, timely_beta=timely_beta,enable_pfc=enable_pfc,enable_qcn=enable_qcn)
        elif args.cc == "dcqcn_paper":
            config = config_template.format(local_dir=local_dir,root=root, bw=bw, trace=trace, topo=topo, trace_track=topo.replace("topo","trace"), cc=args.cc, mode=1, t_alpha=50, t_dec=50, t_inc=55, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=0, vwin=0, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=1, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr, fwin=fwin, base_rtt=base_rtt,duration=duration,config_specs=config_specs,timely_t_high=timely_t_high,timely_t_low=timely_t_low, timely_beta=timely_beta,enable_pfc=enable_pfc,enable_qcn=enable_qcn)
        elif args.cc == "dcqcn_vwin":
            config = config_template.format(local_dir=local_dir,root=root, bw=bw, trace=trace, topo=topo, trace_track=topo.replace("topo","trace"), cc=args.cc, mode=1, t_alpha=1, t_dec=4, t_inc=300, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=1, vwin=1, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=0, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr, fwin=fwin, base_rtt=base_rtt,duration=duration,config_specs=config_specs,timely_t_high=timely_t_high,timely_t_low=timely_t_low, timely_beta=timely_beta,enable_pfc=enable_pfc,enable_qcn=enable_qcn)
        elif args.cc == "dcqcn_paper_vwin":
            config = config_template.format(local_dir=local_dir,root=root, bw=bw, trace=trace, topo=topo, trace_track=topo.replace("topo","trace"), cc=args.cc, mode=1, t_alpha=50, t_dec=50, t_inc=55, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=1, vwin=1, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=0, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr, fwin=fwin, base_rtt=base_rtt,duration=duration,config_specs=config_specs,timely_t_high=timely_t_high,timely_t_low=timely_t_low, timely_beta=timely_beta,enable_pfc=enable_pfc,enable_qcn=enable_qcn)
    elif args.cc == "hp":
        ai = 10 * bw / 25;
        # if args.hpai > 0:
        #     ai = args.hpai
        if hpai > 0:
            ai = hpai
        hai = ai # useless
        int_multi = max(bw / 25, 1);
        cc = "%s%d"%(args.cc, args.utgt)
        if (mi > 0):
            cc += "mi%d"%mi
        # if args.hpai > 0:
        #     cc += "ai%d"%ai
        if hpai > 0:
            cc += "ai%d"%ai
        # config_name = "%s/config_%s_%s_%s%s%s.txt"%(root, topo, trace, cc, failure, config_specs)
        print("cc:", cc)
        config = config_template.format(local_dir=local_dir,root=root, bw=bw, trace=trace, topo=topo, trace_track=topo.replace("topo","trace"), cc=args.cc, mode=3, t_alpha=1, t_dec=4, t_inc=300, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=1, vwin=1, us=1, u_tgt=u_tgt, mi=mi, int_multi=int_multi, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=0, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr, fwin=fwin, base_rtt=base_rtt,duration=duration,config_specs=config_specs,timely_t_high=timely_t_high,timely_t_low=timely_t_low, timely_beta=timely_beta,enable_pfc=enable_pfc,enable_qcn=enable_qcn)
    elif args.cc == "dctcp":
        ai = 10 # ai is useless for dctcp
        hai = ai  # also useless
        dctcp_ai=615 # calculated from RTT=13us and MTU=1KB, because DCTCP add 1 MTU per RTT.
        # masking_threshold_K={dctcp_k} KB (e.g., 30KB)
        kmax_map = "2 %d %d %d %d"%(bw*1000000000, dctcp_k*bw/10, bw*4*1000000000, dctcp_k*bw*4/10)
        kmin_map = "2 %d %d %d %d"%(bw*1000000000, dctcp_k*bw/10, bw*4*1000000000, dctcp_k*bw*4/10)
        pmax_map = "2 %d %.2f %d %.2f"%(bw*1000000000, 1.0, bw*4*1000000000, 1.0)
        config = config_template.format(local_dir=local_dir,root=root, bw=bw, trace=trace, topo=topo, trace_track=topo.replace("topo","trace"), cc=args.cc, mode=8, t_alpha=1, t_dec=4, t_inc=300, g=0.0625, ai=ai, hai=hai, dctcp_ai=dctcp_ai, has_win=1, vwin=1, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=1, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr, fwin=fwin, base_rtt=base_rtt,duration=duration,config_specs=config_specs,timely_t_high=timely_t_high,timely_t_low=timely_t_low, timely_beta=timely_beta,enable_pfc=enable_pfc,enable_qcn=enable_qcn)
    elif args.cc == "timely":
        ai = 10 * bw / 10
        hai = 50 * bw / 10
        config = config_template.format(local_dir=local_dir,root=root, bw=bw, trace=trace, topo=topo, trace_track=topo.replace("topo","trace"), cc=args.cc, mode=7, t_alpha=1, t_dec=4, t_inc=300, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=0, vwin=0, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=1, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr, fwin=fwin, base_rtt=base_rtt,duration=duration,config_specs=config_specs,timely_t_high=timely_t_high,timely_t_low=timely_t_low, timely_beta=timely_beta,enable_pfc=enable_pfc,enable_qcn=enable_qcn)
    elif args.cc == "timely_vwin":
        ai = 10 * bw / 10;
        hai = 50 * bw / 10;
        config = config_template.format(local_dir=local_dir,root=root, bw=bw, trace=trace, topo=topo, trace_track=topo.replace("topo","trace"), cc=args.cc, mode=7, t_alpha=1, t_dec=4, t_inc=300, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=1, vwin=1, us=0, u_tgt=u_tgt, mi=mi, int_multi=1, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=1, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr, fwin=fwin, base_rtt=base_rtt,duration=duration,config_specs=config_specs,timely_t_high=timely_t_high,timely_t_low=timely_t_low, timely_beta=timely_beta,enable_pfc=enable_pfc,enable_qcn=enable_qcn)
    elif args.cc == "hpccPint":
        ai = 10 * bw / 25;
        # if args.hpai > 0:
        #     ai = args.hpai
        if hpai > 0:
            ai = hpai
        hai = ai # useless
        int_multi = max(bw / 25, 1);
        cc = "%s%d"%(args.cc, args.utgt)
        if (mi > 0):
            cc += "mi%d"%mi
        # if args.hpai > 0:
        #     cc += "ai%d"%ai
        if hpai > 0:
            cc += "ai%d"%ai
        cc += "log%.3f"%pint_log_base
        cc += "p%.3f"%pint_prob
        # config_name = "%s/config_%s_%s_%s%s%s.txt"%(root, topo, trace, cc, failure, config_specs)
        print("cc:", cc)
        config = config_template.format(local_dir=local_dir, root=root, bw=bw, trace=trace, topo=topo, trace_track=topo.replace("topo","trace"), cc=args.cc, mode=10, t_alpha=1, t_dec=4, t_inc=300, g=0.00390625, ai=ai, hai=hai, dctcp_ai=1000, has_win=1, vwin=1, us=1, u_tgt=u_tgt, mi=mi, int_multi=int_multi, pint_log_base=pint_log_base, pint_prob=pint_prob, ack_prio=0, link_down=args.down, failure=failure, kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz, enable_tr=enable_tr, fwin=fwin, base_rtt=base_rtt,duration=duration,config_specs=config_specs,timely_t_high=timely_t_high,timely_t_low=timely_t_low, timely_beta=timely_beta,enable_pfc=enable_pfc,enable_qcn=enable_qcn)
    else:
        print("unknown cc:", args.cc)
        sys.exit(1)

    with open(config_name, "w") as file:
        file.write(config)
    with open("%s/param_%s%s%s.txt"%(root, topo, failure, config_specs), "w") as file:
        file.write(" ".join(map(str, DEFAULT_PARAM_VEC)) + "\n")
        file.write("0 {} {} {} {} {} {} {} {} {} {} {}\n".format(bfsz, fwin, enable_pfc, cc, dctcp_k, dcqcn_k_min, dcqcn_k_max, u_tgt, hpai, timely_t_low, timely_t_high))
    np.save("%s/param_%s%s%s.npy"%(root, topo, failure, config_specs), DEFAULT_PARAM_VEC)
    os.system("./waf --run 'scratch/third %s'"%(config_name))