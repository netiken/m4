import subprocess
import argparse
import numpy as np
import os
import numpy as np
from os.path import abspath, dirname
from enum import Enum
from collections import deque, defaultdict
import traceback

cur_dir = dirname(abspath(__file__))
os.chdir(cur_dir)


def fix_seed(seed):
    np.random.seed(seed)


def gen_busy_periods(flows, flow_size_threshold, remainsize_list):
    if flow_size_threshold == 100000000:
        flow_size_threshold = np.inf
    active_graphs = {}

    busy_periods = []
    busy_periods_len = []
    busy_periods_time = []
    busy_periods_unique = set()
    remainsizes = []
    remainsizes_num = []
    events = []
    for flow_id, flow in flows.items():
        events.append(
            (flow["start_time"], "start", flow_id, flow["links"], flow["size"])
        )
        events.append((flow["end_time"], "end", flow_id, flow["links"], flow["size"]))
    events.sort()

    link_to_graph = {}  # Map to quickly find which graph a link belongs to
    graph_id_new = 0  # Unique identifier for each graph
    large_flow_to_info = {}
    flow_to_size = {}

    for event_idx, (time, event, flow_id, links, size) in enumerate(events):
        cur_time = time
        # if flow_id % 1000 == 0:
        #     print(f"Processing flow {flow_id}")
        if event == "start":
            flow_to_size[flow_id] = size
            if size > flow_size_threshold:
                large_flow_to_info[flow_id] = (time, links)
                # involved_graph_ids = set()
                # for link in links:
                #     if link in link_to_graph:
                #         involved_graph_ids.add(link_to_graph[link])
                # if involved_graph_ids:
                #     for gid in involved_graph_ids:
                #         graph = active_graphs[gid]
                #         graph["all_links"].add(link)
                #         graph["all_flows"].add(flow_id)
            else:
                new_active_links = defaultdict(set)
                new_all_links = set()
                new_flows = set()
                new_all_flows = set()
                new_event_idxs = set()

                # Find all graphs involved with the new flow's links
                involved_graph_ids = set()
                for link in links:
                    if link in link_to_graph:
                        involved_graph_ids.add(link_to_graph[link])

                if involved_graph_ids:
                    for gid in involved_graph_ids:
                        graph = active_graphs[gid]
                        new_active_links.update(graph["active_links"])
                        new_all_links.update(graph["all_links"])
                        new_flows.update(graph["active_flows"])
                        new_all_flows.update(graph["all_flows"])
                        new_event_idxs.update(graph["event_idxs"])
                        if cur_time > graph["start_time"]:
                            cur_time = graph["start_time"]

                        for link in graph["active_links"]:
                            link_to_graph[link] = graph_id_new
                        del active_graphs[gid]

                for link in links:
                    new_active_links[link].add(flow_id)
                    new_all_links.add(link)
                    link_to_graph[link] = graph_id_new
                new_flows.add(flow_id)
                new_all_flows.add(flow_id)
                new_event_idxs.add(event_idx)
                for large_flow_id in large_flow_to_info:
                    _, links_tmp = large_flow_to_info[large_flow_id]
                    if large_flow_id not in new_all_flows and not links_tmp.isdisjoint(
                        new_all_links
                    ):
                        new_all_flows.add(large_flow_id)
                active_graphs[graph_id_new] = {
                    "active_links": new_active_links,
                    "all_links": new_all_links,
                    "active_flows": new_flows,
                    "all_flows": new_all_flows,
                    "start_time": cur_time,
                    "event_idxs": new_event_idxs,
                }
                graph_id_new += 1

        elif event == "end":
            graph = None
            flow_to_size.pop(flow_id)
            if flow_id in large_flow_to_info:
                large_flow_to_info.pop(flow_id)
                # involved_graph_ids = set()
                # for link in links:
                #     if link in link_to_graph:
                #         involved_graph_ids.add(link_to_graph[link])
                # if involved_graph_ids:
                #     for gid in involved_graph_ids:
                #         graph = active_graphs[gid]
                #         graph["all_links"].add(link)
                #         graph["all_flows"].add(flow_id)
                # continue
            else:
                for link in links:
                    if link in link_to_graph:
                        graph_id = link_to_graph[link]
                        graph = active_graphs[graph_id]
                        break

                if graph:
                    for link in links:
                        if flow_id in graph["active_links"][link]:
                            graph["active_links"][link].remove(flow_id)
                            if not graph["active_links"][link]:
                                del graph["active_links"][link]
                                del link_to_graph[link]
                        else:
                            assert (
                                False
                            ), f"Flow {flow_id} not found in link {link} of graph {graph_id}"
                    if flow_id in graph["active_flows"]:
                        graph["active_flows"].remove(flow_id)
                    else:
                        assert (
                            False
                        ), f"Flow {flow_id} not found in active flows of graph {graph_id}"
                    graph["event_idxs"].add(event_idx)

                    # if not graph['active_flows']:  # If no active flows left in the graph
                    n_small_flows = len(
                        [
                            flow_id
                            for flow_id in graph["active_flows"]
                            if flow_to_size[flow_id] <= flow_size_threshold
                        ]
                    )
                    # n_large_flows=len(graph['active_flows'])-n_small_flows
                    if n_small_flows == 0:  # If no active flows left in the graph
                        assert (
                            len(graph["active_flows"])
                            == len(graph["active_links"])
                            == 0
                        ), f"n_active_flows: {len(graph['active_flows'])}, n_active_links: {len(graph['active_links'])}"
                        # end_time = cur_time
                        # for flow_id in graph['active_flows']:
                        #     if flow_to_end_time[flow_id]>end_time:
                        #         end_time=flow_to_end_time[flow_id]

                        fid_target = sorted(graph["all_flows"])
                        busy_periods.append(tuple(fid_target))
                        busy_periods_len.append(len(graph["all_flows"]))
                        busy_periods_time.append([graph["start_time"], cur_time])
                        busy_periods_unique.update(graph["all_flows"])

                        busy_period_event_idxs = sorted(graph["event_idxs"])
                        remainsize = []
                        for i in busy_period_event_idxs:
                            tmp = remainsize_list[i]
                            if isinstance(tmp, dict):
                                tmp_list = []
                                for j in fid_target:
                                    if j in tmp:
                                        tmp_list.append(tmp[j])
                                if len(tmp_list) > 0:
                                    remainsize.append(tmp_list)
                                else:
                                    remainsize.append([0])
                            else:
                                remainsize.append(tmp)
                        assert (
                            len(remainsize) == len(fid_target) * 2
                        ), f"{len(remainsize)} != {len(fid_target) * 2}"

                        remainsizes_num.append(np.max([len(x) for x in remainsize]))
                        remainsizes.append(tuple(remainsize))

                        del active_graphs[graph_id]
                        # for link in graph["active_links"]:
                        #     del link_to_graph[link]

                        # if n_large_flows>0:
                        #     new_active_links = defaultdict(set)
                        #     new_all_links = set()
                        #     new_flows = graph['active_flows']
                        #     new_all_flows = graph['active_flows']
                        #     start_time=cur_time
                        #     for flow_id in graph['active_flows']:
                        #         for link in large_flow_to_info[flow_id][1]:
                        #             new_active_links[link].add(flow_id)
                        #             new_all_links.add(link)
                        #             link_to_graph[link] = graph_id_new
                        #         # if large_flow_to_info[flow_id][0]<start_time:
                        #         #     start_time=large_flow_to_info[flow_id][0]
                        #     active_graphs[graph_id_new] = {
                        #         'active_links': new_active_links,
                        #         'all_links': new_all_links,
                        #         'active_flows': new_flows,
                        #         'all_flows': new_all_flows,
                        #         'start_time': start_time
                        #     }
                        #     graph_id_new += 1
                else:
                    assert False, f"Flow {flow_id} has no active graph"
    print(
        f"n_flow_event: {len(events)}, {len(busy_periods)} busy periods, flow_size_threshold: {flow_size_threshold}, n_flows_unique: {len(busy_periods_unique)} , n_flows_per_period_est: {np.min(busy_periods_len)}, {np.mean(busy_periods_len)}, {np.max(busy_periods_len)}"
    )
    return busy_periods, busy_periods_time, remainsizes, remainsizes_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-p",
        dest="prefix",
        action="store",
        default="topo4-4_traffic",
        help="Specify the prefix of the fcts file. Usually like fct_<topology>_<trace>",
    )
    parser.add_argument("-s", dest="step", action="store", default="5")
    parser.add_argument(
        "--random_seed", dest="random_seed", type=int, default=0, help="random seed"
    )
    parser.add_argument(
        "--shard_cc", dest="shard_cc", type=int, default=0, help="random seed"
    )
    parser.add_argument(
        "-t",
        dest="type",
        action="store",
        type=int,
        default=0,
        help="0: normal, 1: incast, 2: all",
    )
    parser.add_argument(
        "--enable_tr",
        dest="enable_tr",
        action="store",
        type=int,
        default=0,
        help="enable tracing",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        action="store",
        default="data/input",
        help="the name of the flow file",
    )
    parser.add_argument(
        "--max_inflight_flows",
        dest="max_inflight_flows",
        type=int,
        default=0,
        help="max inflgiht flows for close-loop traffic",
    )
    parser.add_argument(
        "--cc",
        dest="cc",
        action="store",
        default="hp",
        help="hp/dcqcn/timely/dctcp/hpccPint",
    )
    args = parser.parse_args()
    enable_tr = args.enable_tr
    flow_size_threshold_list = [100000000]

    fix_seed(args.random_seed)
    time_limit = int(30000 * 1e9)
    shard_cc = args.shard_cc
    # max_inflight_flows = args.max_inflight_flows
    # config_specs = "_s%d_i%d" % (shard_cc, max_inflight_flows)
    # config_specs = "_%s" % (args.cc)
    config_specs = ""
    output_dir = args.output_dir

    with open("%s/flows.txt" % (output_dir), "r") as f:
        all_lines = f.read().splitlines()
    n_flows = int(all_lines[0].strip())
    data_lines = all_lines[1:]
    assert n_flows == len(data_lines)
    size_list = []
    fat_list = []
    for line in data_lines:
        tmp = line.split(" ")
        size = tmp[-2]
        fat = float(tmp[-1]) * 1e9
        size_list.append(size)
        fat_list.append(fat)
    np.save("%s/fsize.npy" % (output_dir), np.array(size_list).astype("int64"))
    np.save("%s/fat.npy" % (output_dir), np.array(fat_list).astype("int64"))
    print(f"fsize: {len(size_list)}, fat: {len(fat_list)}")

    file = "%s/fct_%s%s.txt" % (output_dir, args.prefix, config_specs)
    if not os.path.exists(file):
        link_info_file = "%s/path_1.txt" % (output_dir)
        link_info_list = []
        with open(link_info_file, "r") as file:
            num_flows, num_path = map(int, file.readline().strip().split(","))
            # assert num_flows == len(fids)
            # print(f"num_flows: {num_flows}, num_path: {num_path}")
            for _ in range(num_flows):
                tmp = file.readline().strip().split(":")
                link_info = tmp[1].split(",")
                link_list = [link_info[i] for i in range(1, len(link_info) - 1)]
                link_info_list.append(link_list)

        np.save(
            "%s/flow_to_path.npy" % (output_dir),
            np.array(link_info_list, dtype=object),
        )
        link_list = list(set().union(*link_info_list))
        link_list = sorted(link_list)
        np.save("%s/flink.npy" % (output_dir), np.array(link_list))
        exit(0)
    # flowId, sip, dip, sport, dport, size (B), start_time, fcts (ns), standalone_fct (ns)
    cmd = (
        "cat %s" % (file)
        + " | awk '{if ($5==0 && $7+$8<"
        + "%d" % time_limit
        + ") {slow=$8/$9;print slow<1?$9:$8, $9, $6, $7, $2, $3, $1}}' | sort -n -k 4,4 -k 7,7"
    )
    output = subprocess.check_output(cmd, shell=True)

    try:
        output = output.decode()
        tmp = output[:-1].split("\n")
        res_np = np.array([x.split() for x in tmp])
        print(res_np.shape)
        fcts = res_np[:, 0].astype("int64")
        i_fcts = res_np[:, 1].astype("int64")
        fsize = res_np[:, 2].astype("int64")
        fats = res_np[:, 3].astype("int64")
        fids = res_np[:, 6].astype("int64")

        np.save(
            "%s/fct_%s%s.npy" % (output_dir, args.prefix, config_specs), fcts
        )  # Byte
        np.save(
            "%s/fct_i_%s%s.npy" % (output_dir, args.prefix, config_specs),
            i_fcts,
        )  # ns
        np.save("%s/fid_%s%s.npy" % (output_dir, args.prefix, config_specs), fids)
        # np.save("%s/fat.npy" % (output_dir), fats)
        # np.save("%s/fsize.npy" % (output_dir), fsize)
        link_info_file = "%s/path_1.txt" % (output_dir)
        flows = {}
        for i in range(len(fids)):
            flows[fids[i]] = {
                "start_time": fats[i],
                "end_time": fats[i] + fcts[i],
                "size": fsize[i],
            }
        link_info_list = []
        with open(link_info_file, "r") as file:
            num_flows, num_path = map(int, file.readline().strip().split(","))
            # assert num_flows == len(fids)
            # print(f"num_flows: {num_flows}, num_path: {num_path}")
            for _ in range(num_flows):
                tmp = file.readline().strip().split(":")
                flow_id = int(tmp[0])
                link_info = tmp[1].split(",")
                link_list = [link_info[i] for i in range(1, len(link_info) - 1)]
                if flow_id in flows:
                    flows[flow_id]["links"] = link_list
                link_info_list.append(link_list)

        np.save(
            "%s/flow_to_path.npy" % (output_dir),
            np.array(link_info_list, dtype=object),
        )
        link_list = list(set().union(*link_info_list))
        link_list = sorted(link_list)
        np.save("%s/flink.npy" % (output_dir), np.array(link_list))
        if enable_tr:
            tr_path = "%s/mix_%s%s.tr" % (output_dir, args.prefix, config_specs)
            log_path = tr_path.replace(".tr", ".log")
            if not os.path.exists(log_path):
                os.system(f"{cur_dir}/../analysis/trace_reader {tr_path} > {log_path}")
            if os.path.exists(tr_path):
                os.system("rm %s" % tr_path)
            if os.path.exists(log_path):
                remainsize_list = []
                queuelen_list = defaultdict(list)
                with open(log_path, "r") as file:
                    for line in file:
                        line = line.strip().rstrip(",").split(",")
                        # Print each line
                        if not line[0].startswith("q"):
                            if len(line[0]) > 1:
                                line_dict = {}
                                for i in range(len(line)):
                                    tmp = line[i].split(":")
                                    line_dict[int(tmp[0])] = int(tmp[1])
                                remainsize_list.append(line_dict)
                            else:
                                remainsize_list.append([0])
                        else:
                            tmp = line[0].split("-")
                            queuelen_list[int(tmp[1])].append(int(tmp[2]))
            queuelen_list = np.array(queuelen_list, dtype=object)
            np.save(
                "%s/qlen_%s%s.npy" % (output_dir, args.prefix, config_specs),
                queuelen_list,
            )
        else:
            # Create empty remainsize_list when no trace files
            remainsize_list = [[0]] * (len(fids) * 2)  # 2 events per flow (start, end)
        
        # Generate busy periods (works with or without trace files)
        for flow_size_threshold in flow_size_threshold_list:
            (
                busy_periods,
                busy_periods_time,
                busy_periods_remainsize,
                remainsizes_num,
            ) = gen_busy_periods(flows, flow_size_threshold, remainsize_list)
            busy_periods = np.array(busy_periods, dtype=object)
            np.save(
                "%s/period_%s%s_t%d.npy"
                % (output_dir, args.prefix, config_specs, flow_size_threshold),
                busy_periods,
            )
            np.save(
                "%s/period_time_%s%s_t%d.npy"
                % (output_dir, args.prefix, config_specs, flow_size_threshold),
                np.array(busy_periods_time),
            )
            if enable_tr:
                # Only save remainsize files when we have real trace data
                busy_periods_remainsize = np.array(
                    busy_periods_remainsize, dtype=object
                )
                np.save(
                    "%s/period_remainsize_%s%s_t%d.npy"
                    % (output_dir, args.prefix, config_specs, flow_size_threshold),
                    np.array(busy_periods_remainsize),
                )
                np.save(
                    "%s/period_remainsize_num_%s%s_t%d.npy"
                    % (output_dir, args.prefix, config_specs, flow_size_threshold),
                    np.array(remainsizes_num),
                )
                # with open("%s/period_%s%s.txt" % (output_dir, args.prefix, config_specs), "w") as file:
                #     for period in flow_id_per_period_est:
                #         file.write(" ".join(map(str, period)) + "\n")

            # if os.path.exists(log_path):
            #     os.system("rm %s" % log_path)

        # os.system("rm %s" % (file))

        # if os.path.exists("%s/flows.txt" % (output_dir)):

        # os.system("rm %s/flows.txt" % (output_dir))

        # os.system(
        #     "rm %s" % ("%s/pfc_%s%s.txt" % (output_dir, args.prefix, config_specs))
        # )

        # os.system(
        #     "rm %s" % ("%s/qlen_%s%s.txt" % (output_dir, args.prefix, config_specs))
        # )

        # os.system(
        #     "rm %s"
        #     % ("%s/pdrop_%s%s.txt" % (output_dir, args.prefix, config_specs))
        # )
    except Exception as e:
        print(output_dir, args.prefix, config_specs, e)
        traceback.print_exc()
        pass
