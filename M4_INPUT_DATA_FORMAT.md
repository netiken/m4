# Input Files for run_m4_post.py

Complete guide showing what files your collaborator needs to provide for `run_m4_post.py`.

## Directory Structure

```
/your/dataset/0/ns3/
├── flows.txt                    # ✅ Required: Flow specifications
├── fct_topology_flows.txt       # ✅ Required: Flow completion times  
├── path_1.txt                   # ✅ Required: Network routing paths
└── mix_topology_flows.tr        # ⚠️  Optional: Packet-level trace (for queue data)
```

**That's it!** Only these 4 files are read by `run_m4_post.py`.

## Required Files with Details

### 1. **flows.txt**
Defines all flows that will be simulated.

**Format:** `flow_id src_node dst_node protocol port size_bytes start_time_seconds`

**Data meaning:**
- `flow_id`: Unique identifier for this flow (0, 1, 2, ...)
- `src_node, dst_node`: Source and destination **node IDs** in the network topology  
- `src_port, dst_port`: Source and destination ports
- `size_bytes`: How many bytes this flow will send
- `start_time_seconds`: When the flow starts (in seconds from simulation start)

**Example:**
```
2000                    # Total number of flows
0 1 0 3 100 86223 1     # Flow 0: from node 1 to node 0, 86KB, starts at t=1s
1 17 19 3 100 70220 1.000003221   # Flow 1: from node 17 to node 19, 70KB, starts at t=1.000003221s
2 15 12 3 100 107287 1.000011388  # Flow 2: from node 15 to node 12, 107KB, starts at t=1.000011388s
```

### 2. **fct_topology_flows.txt**  
Flow completion results from NS-3 simulation.

**Format:** `flowId src_ip_binary dst_ip_binary port1 port2 size_bytes start_time_ns actual_fct_ns ideal_fct_ns`

**Data meaning:**
- `flowId`: Same flow ID as in flows.txt
- `src_ip_binary, dst_ip_binary`: Source/destination IP in binary format (0b000901 = node 9's IP)
- `port1, port2`: Network ports used
- `size_bytes`: Flow size (matches flows.txt)
- `start_time_ns`: Start time in **nanoseconds** (flows.txt * 1e9)
- `actual_fct_ns`: How long the flow actually took to complete (nanoseconds)
- `ideal_fct_ns`: How long it would take with no congestion (nanoseconds)

**Example:**
```
4 0b000901 0b000a01 10000 100 5444 1000032281 7464 7421
# Flow 4: from node 9 to node 10, took 7464ns actual vs 7421ns ideal
1 0b001101 0b001301 10000 100 70220 1000003221 62144 61712  
# Flow 1: from node 17 to node 19, took 62144ns actual vs 61712ns ideal
```

### 3. **path_1.txt**
Network routing paths that each flow takes through the network.

**Format:** First line: `num_flows,num_paths`. Then: `flow_id:src-dst,link1,link2,link3,...,`

**Data meaning:**
- **First pair (src-dst):** Source and destination nodes for this flow
- **Following pairs (links):** Actual network links used in the routing path
- **Links are node pairs:** `A-B` means a direct network connection between node A and node B

**Example:**
```
2000,557                # 2000 flows, 557 unique network links total  
0:1-0,1-32,32-0,        # Flow 0: from node 1 to node 0, route via links 1-32, 32-0
1:17-19,17-36,36-19,    # Flow 1: from node 17 to node 19, route via links 17-36, 36-19  
2:15-12,15-35,35-12,    # Flow 2: from node 15 to node 12, route via links 15-35, 35-12
```

**Path interpretation:** 
- Flow 0: goes from node 1 to node 0, routing path is `1→32→0` using links `1-32` and `32-0`
- Flow 1: goes from node 17 to node 19, routing path is `17→36→19` using links `17-36` and `36-19`

### 4. **mix_topology_flows.tr** (Optional)
Packet-level trace from NS-3 showing detailed flow behavior.

**When needed:** Only if you want queue lengths and remaining flow sizes for training
**Format:** Binary NS-3 trace file (gets converted to .log format by script)
**Contains:** Packet send/receive events, queue occupancy, flow progress

## Usage

```bash
# Basic processing
python run_m4_post.py -p topology_flows --output_dir /path/to/ns3/data

# With trace data (recommended)  
python run_m4_post.py -p topology_flows --output_dir /path/to/ns3/data --enable_tr 1
```

## Key Points

- **File naming:** Use prefix `topology_flows` (or adjust `-p` parameter)
- **Time units:** flows.txt uses seconds, fct_*.txt uses nanoseconds  
- **IP format:** fct_*.txt uses binary format (0b...)
- **Flow IDs:** Must match between all files

**That's it!** Your collaborator provides these 3-4 files, you run the script, and it creates the numpy files for training.