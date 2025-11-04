import os
import re
import shutil
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def list_subdirs(path):
    try:
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        return []


def list_txt_files(path):
    try:
        return sorted([f for f in os.listdir(path) if f.endswith(".txt") and os.path.isfile(os.path.join(path, f))])
    except FileNotFoundError:
        return []


def filter_complete_experiments(raw_dir, exp_names, client_dirs):
    """Filter experiments to only include those with logs for required clients and server"""
    complete_experiments = []
    
    for exp in exp_names:
        has_all_clients = True
        
        # Check each client
        for client in client_dirs:
            client_exp_path = os.path.join(raw_dir, client, exp)
            if not os.path.exists(client_exp_path):
                has_all_clients = False
                break
            # Check if there's at least one log file
            try:
                files = os.listdir(client_exp_path)
                log_files = [f for f in files if f.endswith('.txt')]
                if not log_files:
                    has_all_clients = False
                    break
            except:
                has_all_clients = False
                break
        
        # Check server logs
        if has_all_clients:
            server_exp_path = os.path.join(raw_dir, 'server', exp)
            if not os.path.exists(server_exp_path):
                has_all_clients = False
            else:
                try:
                    files = os.listdir(server_exp_path)
                    log_files = [f for f in files if f.endswith('.txt')]
                    if not log_files:
                        has_all_clients = False
                except:
                    has_all_clients = False
        
        if has_all_clients:
            complete_experiments.append(exp)
            print(f"Including experiment {exp} (all {len(client_dirs)} clients + server)")
        else:
            print(f"Skipping experiment {exp} (missing clients or server logs)")
    
    return complete_experiments


def concat_txt_files(src_dir, dst_file):
    files = list_txt_files(src_dir)
    if not files:
        return False
    with open(dst_file, "w", encoding="utf-8") as out_f:
        for name in files:
            full = os.path.join(src_dir, name)
            with open(full, "r", encoding="utf-8", errors="ignore") as in_f:
                for line in in_f:
                    out_f.write(line)
    return True


def copy_single_txt(src_dir, dst_file):
    files = list_txt_files(src_dir)
    if not files:
        return False
    if len(files) == 1:
        shutil.copy2(os.path.join(src_dir, files[0]), dst_file)
        return True
    return concat_txt_files(src_dir, dst_file)


def parse_field_int(line, key):
    idx = line.find(key)
    if idx == -1:
        return None
    idx += len(key)
    end = idx
    n = len(line)
    while end < n and line[end].isdigit():
        end += 1
    if end == idx:
        return None
    try:
        return int(line[idx:end])
    except Exception:
        return None


def parse_ts_ns(line):
    return parse_field_int(line, "ts_ns=")


def parse_flow_id(line):
    flow_id = parse_field_int(line, "flow_id=")
    if flow_id is not None:
        return flow_id
    
    # If no explicit flow_id, generate synthetic one from clt, wrkr, slot
    clt = parse_field_int(line, "clt=")
    wrkr = parse_field_int(line, "wrkr=")
    slot = parse_field_int(line, "slot=")
    
    if clt is not None and wrkr is not None and slot is not None:
        # Generate synthetic flow_id: combine clt, wrkr, slot into unique ID
        return (clt << 16) | (wrkr << 8) | slot
    
    return None


def parse_dur_ns(line):
    return parse_field_int(line, "dur_ns=")


def parse_start_ns(line):
    return parse_field_int(line, "start_ns=")


def parse_size_bytes(line):
    return parse_field_int(line, "size=")


def parse_wire_bytes(line):
    return parse_field_int(line, "wire_bytes=")


def parse_client_id(line):
    return parse_field_int(line, "clt=")


def parse_event(line):
    key = "event="
    idx = line.find(key)
    if idx == -1:
        return None
    idx += len(key)
    end = idx
    n = len(line)
    while end < n and line[end] != ' ' and line[end] != '\n' and line[end] != '\t':
        end += 1
    if end == idx:
        return None
    return line[idx:end]


def merge_server_logs_time_sorted(src_dir, dst_file):
    files = list_txt_files(src_dir)
    if not files:
        return False
    entries = []
    for name in files:
        full = os.path.join(src_dir, name)
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ts = parse_ts_ns(line)
                if ts is not None:
                    entries.append((ts, line))
    if not entries:
        return False
    entries.sort(key=lambda x: x[0])
    with open(dst_file, "w", encoding="utf-8") as out_f:
        for _, line in entries:
            if line.endswith("\n"):
                out_f.write(line)
            else:
                out_f.write(line + "\n")
    return True


def plot_metric(values, out_path, title):
    if not HAS_MPL:
        return
    if not values:
        return
    x = list(range(len(values)))
    y = [v if v is not None else float('nan') for v in values]
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='.', linewidth=1)
    plt.xlabel('index')
    plt.ylabel('ns')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_ratio(values, out_path, title):
    if not HAS_MPL:
        return
    if not values:
        return
    x = list(range(len(values)))
    y = [v if v is not None else float('nan') for v in values]
    plt.figure(figsize=(16, 8))
    plt.plot(x, y, marker='.', linewidth=2)
    ax = plt.gca()
    ax.set_ylim(0, 15)
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.tick_params(axis='both', which='both', labelsize=12, width=1.5)
    plt.xlabel('index', fontsize=14, fontweight='bold')
    plt.ylabel('slowdown (actual/ideal)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def ideal_fct_ns_for_size(size_bytes):
    if size_bytes is None:
        return None
    # 10 Gbps -> 0.8 ns per byte, plus 6000 ns RTT (6us)
    return int(round(size_bytes * 0.8 + 6000))


def write_flows_file_for_experiment(out_dir):
    # discover any number of client files: c0.txt, c1.txt, ...
    try:
        client_files = [
            os.path.join(out_dir, name)
            for name in sorted(os.listdir(out_dir))
            if re.fullmatch(r"c\d+\.txt", name)
        ]
    except FileNotFoundError:
        client_files = []
    server_file = os.path.join(out_dir, "server.txt")
    flows = {}
    first_ts = {}

    any_found = False

    # read client files
    for cf in client_files:
        if not os.path.exists(cf):
            continue
        any_found = True
        with open(cf, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ts = parse_ts_ns(line)
                fid = parse_flow_id(line)
                if ts is None or fid is None:
                    continue
                if fid not in flows:
                    flows[fid] = []
                    first_ts[fid] = ts
                if ts < first_ts[fid]:
                    first_ts[fid] = ts
                flows[fid].append((ts, line))

    # read server file
    if os.path.exists(server_file):
        any_found = True
        with open(server_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ts = parse_ts_ns(line)
                fid = parse_flow_id(line)
                if ts is None or fid is None:
                    continue
                if fid not in flows:
                    flows[fid] = []
                    first_ts[fid] = ts
                if ts < first_ts[fid]:
                    first_ts[fid] = ts
                flows[fid].append((ts, line))

    if not any_found:
        return False

    # Sort flows by first seen ts
    ordered_flows = sorted(first_ts.items(), key=lambda kv: kv[1])

    # metric arrays in flow order
    metric_values = {
        "total": [],
        "network_time": [],
        "server_overhead": [],
        "client_to_worker": [],
        "worker_to_client": [],
        "rdma_time": [],
    }

    # rename debug dump file
    out_path = os.path.join(out_dir, "flows_debug.txt")
    with open(out_path, "w", encoding="utf-8") as out_f:
        for fid, _ in ordered_flows:
            out_f.write(f"flow_id={fid}\n")
            lines = sorted(flows[fid], key=lambda x: x[0])
            # write lines for this flow
            for _, line in lines:
                if line.endswith("\n"):
                    out_f.write(line)
                else:
                    out_f.write(line + "\n")
            # compute metrics
            req_send_ts = None
            req_send_size = None
            resp_recv_ud_ts = None
            req_recv_ts = None
            resp_send_ts = None
            resp_send_wire = None
            rdma_dur_max = None
            rdma_start_for_max = None
            rdma_wire_for_max = None
            for _, line in lines:
                ev = parse_event(line)
                if ev == "req_send":
                    if req_send_ts is None:
                        req_send_ts = parse_ts_ns(line)
                        req_send_size = parse_size_bytes(line)
                elif ev == "resp_recv_ud":
                    resp_recv_ud_ts = parse_ts_ns(line)
                elif ev == "req_recv":
                    if req_recv_ts is None:
                        req_recv_ts = parse_ts_ns(line)
                elif ev == "resp_send":
                    resp_send_ts = parse_ts_ns(line)
                    w = parse_wire_bytes(line)
                    if w is not None:
                        resp_send_wire = w
                elif ev == "resp_rdma_read":
                    dur = parse_dur_ns(line)
                    if dur is not None:
                        if rdma_dur_max is None or dur > rdma_dur_max:
                            rdma_dur_max = dur
                            rdma_start_for_max = parse_start_ns(line)
                            rdma_wire_for_max = parse_wire_bytes(line)
            # calculate
            def calc_total():
                if req_send_ts is not None and resp_recv_ud_ts is not None:
                    return resp_recv_ud_ts - req_send_ts
                return None
            def calc_server_overhead():
                if req_recv_ts is not None and resp_send_ts is not None:
                    return resp_send_ts - req_recv_ts
                return None
            def calc_network_time(total, server_overhead):
                if total is not None and server_overhead is not None:
                    return total - server_overhead
                return None
            def calc_client_to_worker():
                if req_recv_ts is not None and req_send_ts is not None:
                    return req_recv_ts - req_send_ts
                return None
            def calc_worker_to_client():
                if resp_recv_ud_ts is not None and resp_send_ts is not None:
                    return resp_recv_ud_ts - resp_send_ts
                return None
            total = calc_total()
            server_overhead = calc_server_overhead()
            network_time = calc_network_time(total, server_overhead)
            client_to_worker = calc_client_to_worker()
            worker_to_client = calc_worker_to_client()
            rdma_time = rdma_dur_max if rdma_dur_max is not None else None

            # append to arrays in flow order
            metric_values["total"].append(total)
            metric_values["network_time"].append(network_time)
            metric_values["server_overhead"].append(server_overhead)
            metric_values["client_to_worker"].append(client_to_worker)
            metric_values["worker_to_client"].append(worker_to_client)
            metric_values["rdma_time"].append(rdma_time)

            # write metrics line
            def fmt(val):
                return str(val) if val is not None else ""
            out_f.write(
                "total=[" + fmt(total) + "] "
                + "network_time=[" + fmt(network_time) + "] "
                + "server_overhead=[" + fmt(server_overhead) + "] "
                + "client_to_worker=[" + fmt(client_to_worker) + "] "
                + "worker_to_client=[" + fmt(worker_to_client) + "] "
                + "rdma_time=[" + fmt(rdma_time) + "]\n"
            )
            out_f.write("\n")

    # create plots for each metric
    plot_metric(metric_values["network_time"], os.path.join(out_dir, "network_time.png"), "network_time (ns)")
    plot_metric(metric_values["total"], os.path.join(out_dir, "total.png"), "total (ns)")
    plot_metric(metric_values["server_overhead"], os.path.join(out_dir, "server_overhead.png"), "server_overhead (ns)")
    plot_metric(metric_values["client_to_worker"], os.path.join(out_dir, "client_to_worker.png"), "client_to_worker (ns)")
    plot_metric(metric_values["worker_to_client"], os.path.join(out_dir, "worker_to_client.png"), "worker_to_client (ns)")
    plot_metric(metric_values["rdma_time"], os.path.join(out_dir, "rdma_time.png"), "rdma_time (ns)")

    # Build ns3 input files under ns3/
    ns3_dir = os.path.join(out_dir, "ns3")
    ensure_dir(ns3_dir)

    # Re-parse per-flow details to emit ns3 files
    # We'll collect all pseudo-flows (control + rdma) into a unified list, then sort by start time and reindex IDs
    all_flows = []  # entries: dict with keys: start_ns, src_node, dst_node, src_port, dst_port, size_b, actual_ns, ideal_ns, src_ip_bin, dst_ip_bin

    # server/source is fixed to node 4
    SRC_NODE = 4
    SRC_PORT = 10000
    DST_PORT = 100
    SRC_IP_BIN = f"0b{SRC_NODE:06d}"

    for fid, _ in ordered_flows:
        lines = sorted(flows[fid], key=lambda x: x[0])
        req_send_ts = None
        req_send_size = None
        resp_recv_ud_ts = None
        req_recv_ts = None
        resp_send_ts = None
        resp_send_wire = None
        rdma_dur_max = None
        rdma_start_for_max = None
        rdma_wire_for_max = None
        client_id = None
        for _, line in lines:
            if client_id is None:
                c = parse_client_id(line)
                if c is not None:
                    client_id = c
            ev = parse_event(line)
            if ev == "req_send":
                if req_send_ts is None:
                    req_send_ts = parse_ts_ns(line)
                    req_send_size = parse_size_bytes(line)
            elif ev == "resp_recv_ud":
                resp_recv_ud_ts = parse_ts_ns(line)
            elif ev == "req_recv":
                if req_recv_ts is None:
                    req_recv_ts = parse_ts_ns(line)
            elif ev == "resp_send":
                resp_send_ts = parse_ts_ns(line)
                w = parse_wire_bytes(line)
                if w is not None:
                    resp_send_wire = w
            elif ev == "resp_rdma_read":
                dur = parse_dur_ns(line)
                if dur is not None:
                    if rdma_dur_max is None or dur > rdma_dur_max:
                        rdma_dur_max = dur
                        rdma_start_for_max = parse_start_ns(line)
                        rdma_wire_for_max = parse_wire_bytes(line)
        # default client if missing
        if client_id is None:
            client_id = 0
        # use actual client id as destination node (no clamping)
        dst_node = client_id
        dst_ip_bin = f"0b{dst_node:06d}"

        # compute derived
        total = None
        if req_send_ts is not None and resp_recv_ud_ts is not None:
            total = resp_recv_ud_ts - req_send_ts
        server_overhead = None
        if req_recv_ts is not None and resp_send_ts is not None:
            server_overhead = resp_send_ts - req_recv_ts
        network_time = None
        if total is not None and server_overhead is not None:
            network_time = total - server_overhead

        # CONTROL pseudo-flow
        control_size = (req_send_size or 0) + (resp_send_wire or 0)
        control_start_ns = req_send_ts
        control_actual_ns = network_time
        control_ideal_ns = ideal_fct_ns_for_size(control_size) if control_size > 0 else None
        if control_actual_ns is None or control_start_ns is None or control_size <= 0:
            pass
        elif control_actual_ns < 0:
            print(f"skip control flow fid={fid}: negative actual_fct_ns={control_actual_ns}")
        else:
            all_flows.append({
                "start_ns": control_start_ns,
                "src_node": SRC_NODE,
                "dst_node": dst_node,
                "src_port": SRC_PORT,
                "dst_port": DST_PORT,
                "size_b": control_size,
                "actual_ns": control_actual_ns,
                "ideal_ns": control_ideal_ns or 0,
                "src_ip_bin": SRC_IP_BIN,
                "dst_ip_bin": dst_ip_bin,
            })

        # RDMA pseudo-flow
        rdma_size = rdma_wire_for_max
        rdma_start_ns = rdma_start_for_max
        rdma_actual_ns = rdma_dur_max
        rdma_ideal_ns = ideal_fct_ns_for_size(rdma_size) if rdma_size is not None and rdma_size > 0 else None
        if rdma_actual_ns is None or rdma_start_ns is None or rdma_size is None or rdma_size <= 0:
            pass
        elif rdma_actual_ns < 0:
            print(f"skip rdma flow fid={fid}: negative actual_fct_ns={rdma_actual_ns}")
        else:
            all_flows.append({
                "start_ns": rdma_start_ns,
                "src_node": SRC_NODE,
                "dst_node": dst_node,
                "src_port": SRC_PORT,
                "dst_port": DST_PORT,
                "size_b": rdma_size,
                "actual_ns": rdma_actual_ns,
                "ideal_ns": rdma_ideal_ns or 0,
                "src_ip_bin": SRC_IP_BIN,
                "dst_ip_bin": dst_ip_bin,
            })

    # If there are no flows, bail
    if not all_flows:
        return True

    # Sort all flows by start time and normalize so min start = 0
    all_flows.sort(key=lambda d: d["start_ns"])
    min_start_ns = all_flows[0]["start_ns"]

    # Write ns3/flows.txt with reindexed flow IDs by time
    flows_txt_path = os.path.join(ns3_dir, "flows.txt")
    with open(flows_txt_path, "w", encoding="utf-8") as f:
        f.write(str(len(all_flows)) + "\n")
        for new_id, d in enumerate(all_flows):
            start_secs_norm = (d["start_ns"] - min_start_ns) / 1e9
            f.write(f"{new_id} {d['src_node']} {d['dst_node']} {d['src_port']} {d['dst_port']} {d['size_b']} {start_secs_norm}\n")

    # Write ns3/fct_topology_flows.txt with same reindexed IDs and normalized start_time_ns
    fct_txt_path = os.path.join(ns3_dir, "fct_topology_flows.txt")
    with open(fct_txt_path, "w", encoding="utf-8") as f:
        for new_id, d in enumerate(all_flows):
            start_ns_norm = d["start_ns"] - min_start_ns
            f.write(f"{new_id} {d['src_ip_bin']} {d['dst_ip_bin']} {d['src_port']} {d['dst_port']} {d['size_b']} {start_ns_norm} {d['actual_ns']} {d['ideal_ns']}\n")

    # Generate ns3/path_1.txt only if client IDs are within fixed topology (0..3)
    max_dst = max((d["dst_node"] for d in all_flows), default=0)
    min_dst = min((d["dst_node"] for d in all_flows), default=0)
    if min_dst >= 0 and max_dst <= 3:
        # Nodes: server(4)->s0(5)->r(8); c0(0)->s1(6)->r(8); c1(1),c2(2)->s2(7)->r(8)
        S0 = 5
        S1 = 6
        S2 = 7
        R  = 8

        def dst_switch_for_client(dst_node):
            if dst_node == 0:
                return S1
            if dst_node in (1, 2):
                return S2
            # fallback
            return S2

        path_lines = []
        unique_links = set()
        for new_id, d in enumerate(all_flows):
            dst_node = d["dst_node"]
            dsw = dst_switch_for_client(dst_node)
            links = [(d["src_node"], S0), (S0, R), (R, dsw), (dsw, dst_node)]
            for a, b in links:
                x, y = (a, b) if a <= b else (b, a)
                unique_links.add((x, y))
            parts = [f"{new_id}:{d['src_node']}-{dst_node}"] + [f"{a}-{b}" for (a, b) in links]
            path_lines.append(",".join(parts) + ",")

        path_file = os.path.join(ns3_dir, "path_1.txt")
        with open(path_file, "w", encoding="utf-8") as f:
            f.write(f"{len(all_flows)},{len(unique_links)}\n")
            for line in path_lines:
                f.write(line + "\n")
    else:
        print(f"  ns3/path_1.txt: skipped (client IDs outside 0..3, found {min_dst}..{max_dst})")

    # read fct file and plot slowdown actual/ideal
    ratios = []
    try:
        with open(fct_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    actual_ns = int(parts[7])
                    ideal_ns = int(parts[8])
                except Exception:
                    continue
                if ideal_ns <= 0 or actual_ns < 0:
                    continue
                ratios.append(actual_ns / float(ideal_ns))
    except Exception:
        pass
    # remove any stale fct_ratio.png
    try:
        stale = os.path.join(ns3_dir, "fct_ratio.png")
        if os.path.exists(stale):
            os.remove(stale)
    except Exception:
        pass
    plot_ratio(ratios, os.path.join(ns3_dir, "fct_slowdown.png"), "FCT Slowdown (actual/ideal)")

    return True


def main(source_dir_name=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use provided source directory or default to test_logs
    if source_dir_name is None:
        source_dir_name = "test_logs"
    
    raw_dir = os.path.join(base_dir, source_dir_name)
    
    # Create output directory based on source directory name
    if source_dir_name == "test_logs":
        out_root = os.path.join(base_dir, "expirements_12")
    else:
        out_root = os.path.join(base_dir, f"expirements_{source_dir_name}")

    # Check if raw_dir exists
    if not os.path.exists(raw_dir):
        print(f"Source directory {raw_dir} does not exist.")
        return

    # discover client layouts under raw_dir
    all_raw = list_subdirs(raw_dir)
    client_logs_dirs = sorted([d for d in all_raw if re.fullmatch(r"c\d+_logs", d)])
    client_flat_dirs = sorted([d for d in all_raw if re.fullmatch(r"c\d+", d)])
    has_logs_layout = len(client_logs_dirs) > 0
    has_flat_layout = not has_logs_layout and len(client_flat_dirs) > 0

    # Determine which client directories we're working with
    if has_logs_layout:
        client_dirs = client_logs_dirs
    elif has_flat_layout:
        client_dirs = client_flat_dirs
    else:
        client_dirs = []

    # For single-client scenarios (like singleton_1, 16_static, 16_v0), create a synthetic experiment
    # Check if we have a single client scenario: only c1 and server directories, no subdirs under c1
    is_single_client_scenario = (
        not has_logs_layout and 
        "c1" in all_raw and 
        "server" in all_raw and 
        len([d for d in all_raw if d.startswith('c') and d != 'c1']) == 0
    )
    
    if is_single_client_scenario:
        print(f"Detected single-client scenario in {source_dir_name}")
        # Create a synthetic experiment name based on the directory
        exp_names = {source_dir_name}
        client_dirs = ["c1"]
        has_single_client = True
    else:
        has_single_client = False
        # Discover experiment names
        exp_names = set()
        if has_logs_layout:
            # union of subfolders across c*_logs
            for d in client_logs_dirs:
                exp_names.update(list_subdirs(os.path.join(raw_dir, d)))
            # also include experiments found under server_logs if present
            if "server_logs" in all_raw:
                exp_names.update(list_subdirs(os.path.join(raw_dir, "server_logs")))
        elif has_flat_layout:
            # flat layout with per-experiment subdirs under each client folder (e.g., c4/100_2/)
            # Discover experiments as the union of subdirectories across all c*/
            exp_names = set()
            for d in client_flat_dirs:
                subdir_path = os.path.join(raw_dir, d)
                exp_names.update(list_subdirs(subdir_path))
            # Fallback to a synthetic name if none found
            if not exp_names:
                exp_names = {"full_server"}

    if not exp_names:
        print(f"No experiments found under {source_dir_name}.")
        return

    # Filter to only include experiments with required clients and server
    if not has_single_client:
        exp_names = filter_complete_experiments(raw_dir, exp_names, client_dirs)
    else:
        # For single client scenarios, just check that the required files exist
        exp_names = list(exp_names)  # Convert set to list
    
    if not exp_names:
        if has_single_client:
            print("No complete single-client scenario found.")
        else:
            print(f"No complete experiments found (need all {len(client_dirs)} clients + server).")
        return

    print(f"\nProcessing {len(exp_names)} complete experiments...")
    ensure_dir(out_root)

    for exp in sorted(exp_names):
        print(f"Processing experiment: {exp}")
        out_dir = os.path.join(out_root, exp)
        ensure_dir(out_dir)

        # cleanup any previous client outputs (c*.txt)
        try:
            for name in os.listdir(out_dir):
                if re.fullmatch(r"c\d+\.txt", name):
                    try:
                        os.remove(os.path.join(out_dir, name))
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

        if has_single_client:
            # Handle single-client scenario - copy c1 directly to c0.txt
            src_dir = os.path.join(raw_dir, "c1")
            dst_file = os.path.join(out_dir, "c0.txt")
            if os.path.isdir(src_dir):
                ok = copy_single_txt(src_dir, dst_file)
                if ok:
                    print(f"  c1 -> c0.txt")
                else:
                    print(f"  c1: no .txt files found")
            else:
                print(f"  c1: missing")
        elif has_logs_layout:
            # cX_logs -> c(X-1).txt for X>0, c0_logs -> c0.txt
            for src_name in client_logs_dirs:
                m = re.fullmatch(r"c(\d+)_logs", src_name)
                if not m:
                    continue
                idx = int(m.group(1))
                out_idx = idx - 1 if idx > 0 else 0
                dst_file = os.path.join(out_dir, f"c{out_idx}.txt")
                src_dir = os.path.join(raw_dir, src_name, exp)
                if os.path.isdir(src_dir):
                    ok = copy_single_txt(src_dir, dst_file)
                    if ok:
                        print(f"  {src_name} -> {os.path.basename(dst_file)}")
                    else:
                        print(f"  {src_name}: no .txt files found")
                else:
                    print(f"  {src_name}: missing")
        elif has_flat_layout:
            # cX/<exp>/ -> c(X-1).txt for X>=1, map c1->c0, c2->c1, ..., c11->c10
            for src_name in client_flat_dirs:
                m = re.fullmatch(r"c(\d+)", src_name)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx < 1 or idx > 11:  # Only process c1-c11
                    continue
                out_idx = idx - 1  # c1->c0, c2->c1, ..., c11->c10
                dst_file = os.path.join(out_dir, f"c{out_idx}.txt")
                src_dir = os.path.join(raw_dir, src_name, exp)
                if os.path.isdir(src_dir):
                    ok = copy_single_txt(src_dir, dst_file)
                    if ok:
                        print(f"  {src_name}/{exp} -> {os.path.basename(dst_file)}")
                    else:
                        print(f"  {src_name}/{exp}: no .txt files found")
                else:
                    print(f"  {src_name}/{exp}: missing")

        # server logs: merge by ts_ns time sort
        srv_dst = os.path.join(out_dir, "server.txt")
        srv_ok = False
        if has_single_client:
            # Handle single-client scenario - use server directory directly
            srv_src = os.path.join(raw_dir, "server")
            if os.path.isdir(srv_src):
                srv_ok = merge_server_logs_time_sorted(srv_src, srv_dst)
                print("  server -> server.txt (time-sorted)" if srv_ok else "  server: no entries with ts_ns found")
            else:
                print("  server: missing")
        elif has_logs_layout:
            # prefer server_logs/<exp>
            srv_src = os.path.join(raw_dir, "server_logs", exp)
            if os.path.isdir(srv_src):
                srv_ok = merge_server_logs_time_sorted(srv_src, srv_dst)
                print("  server_logs -> server.txt (time-sorted)" if srv_ok else "  server_logs: no entries with ts_ns found")
            else:
                # fallback: server/<exp>
                srv_src_alt = os.path.join(raw_dir, "server", exp)
                if os.path.isdir(srv_src_alt):
                    srv_ok = merge_server_logs_time_sorted(srv_src_alt, srv_dst)
                    print("  server/<exp> -> server.txt (time-sorted)" if srv_ok else "  server/<exp>: no entries with ts_ns found")
                else:
                    # fallback: flat server/ directory
                        srv_src_flat = os.path.join(raw_dir, "server")
                        srv_src_flat_logs = os.path.join(raw_dir, "server_logs")
                        if os.path.isdir(srv_src_flat):
                            srv_ok = merge_server_logs_time_sorted(srv_src_flat, srv_dst)
                            print("  server (flat) -> server.txt (time-sorted)" if srv_ok else "  server (flat): no entries with ts_ns found")
                        elif os.path.isdir(srv_src_flat_logs):
                            srv_ok = merge_server_logs_time_sorted(srv_src_flat_logs, srv_dst)
                            print("  server_logs (flat) -> server.txt (time-sorted)" if srv_ok else "  server_logs (flat): no entries with ts_ns found")
                        else:
                            print("  server_logs: missing")
        elif has_flat_layout:
            # server logs may be under server/<exp> or server_logs/<exp>, fallback to flat server/
            srv_src_exp = os.path.join(raw_dir, "server", exp)
            srv_src_logs_exp = os.path.join(raw_dir, "server_logs", exp)
            srv_src_flat = os.path.join(raw_dir, "server")
            srv_src_flat_logs = os.path.join(raw_dir, "server_logs")
            if os.path.isdir(srv_src_exp):
                srv_ok = merge_server_logs_time_sorted(srv_src_exp, srv_dst)
                print("  server/<exp> -> server.txt (time-sorted)" if srv_ok else "  server/<exp>: no entries with ts_ns found")
            elif os.path.isdir(srv_src_logs_exp):
                srv_ok = merge_server_logs_time_sorted(srv_src_logs_exp, srv_dst)
                print("  server_logs/<exp> -> server.txt (time-sorted)" if srv_ok else "  server_logs/<exp>: no entries with ts_ns found")
            elif os.path.isdir(srv_src_flat):
                srv_ok = merge_server_logs_time_sorted(srv_src_flat, srv_dst)
                print("  server (flat) -> server.txt (time-sorted)" if srv_ok else "  server (flat): no entries with ts_ns found")
            elif os.path.isdir(srv_src_flat_logs):
                srv_ok = merge_server_logs_time_sorted(srv_src_flat_logs, srv_dst)
                print("  server_logs (flat) -> server.txt (time-sorted)" if srv_ok else "  server_logs (flat): no entries with ts_ns found")
            else:
                print("  server: missing")

        # flows_debug + ns3 files
        if write_flows_file_for_experiment(out_dir):
            print("  clients+server -> flows_debug.txt (+ metric plots), ns3/flows.txt, ns3/fct_topology_flows.txt, ns3/path_1.txt, ns3/fct_slowdown.png")
        else:
            print("  clients/server: no files found for flows_debug.txt")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
