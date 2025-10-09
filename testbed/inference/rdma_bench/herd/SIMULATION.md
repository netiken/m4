## HERD Simulation Implementation Guide

This document specifies a simulator-agnostic blueprint to model the HERD key-value application contained in this repository. It is designed for deterministic, message/flow-level simulation, while remaining compatible with packet-level simulators when needed.

The goals are:
- Predict throughput and latency (flow completion time, FCT) under varying configurations
- Preserve determinism by mirroring HERD’s RNG, request formation, batching, and polling
- Remain independent of any single simulation framework

---

### 1. Architecture Overview (what to simulate)
- Clients issue requests via UC RDMA WRITE to a server-managed request region (RR).
- Server workers poll the RR, translate HERD opcodes to MICA ops, batch process with MICA, then reply via UD SEND.
- Each request/response is single-packet sized (≤ MTU) in this benchmark variant.

Key code references:
- Clients: `herd/client.c` (UC WRITE requests, UD RECV completions)
- Workers: `herd/worker.c` (RR polling, batching, UD SEND responses)
- Data model: `mica/mica.{h,c}` (GET/PUT behavior and response size semantics)
- Constants: `herd/main.h` (NUM_WORKERS, NUM_CLIENTS, WINDOW_SIZE, etc.)

---

### 2. Entities and State (components to implement)
- ClientThread (per client)
  - Configuration: `id`, `WINDOW_SIZE`, `update_percentage`, `NUM_WORKERS`
  - RNG state: 64-bit seed; use HERD’s fastrand: `seed = seed * 1103515245 + 12345; return seed >> 32;`
  - Key selection: permutation of `[0..HERD_NUM_KEYS-1]` per client as in code
  - Outstanding requests: track `WINDOW_SIZE` slots per worker (ring index)
  - CQ (recv) polling cadence: poll every `WINDOW_SIZE` sends
  - Logging hooks: record per-op start (cycle or time), request bytes

- WorkerThread (per worker)
  - Round-robin scan over clients each iteration
  - For each client, check RR slot `OFFSET(worker, client, window_slot)`
  - Collect up to `postlist` ready ops per iteration; batch into `mica_batch_op`
  - For each response, compute response length and enqueue a UD SEND
  - Doorbell batching: responses posted per server-port with linked WR lists

- NetworkFabric (logical)
  - Message delivery with serialization delay and optional queueing
  - Optionally, switches and links (for packet-level implementations)

- MICAKeyValue
  - Deterministic GET/PUT semantics and response sizes (or a calibrated service time model if not simulating full hash-table logic)

---

### 3. Message Types and Sizes (exact bytes)
- Request (Client → Server via UC WRITE):
  - GET: 16B key + 1B opcode = 17 bytes
  - PUT: 16B key + 1B opcode + 1B len + `val_len` bytes = 18 + `val_len`
  - All requests are posted inline; assume single-packet serialization

- Response (Server → Client via UD SEND):
  - GET success: `val_len` bytes (value only)
  - GET miss: 0 bytes
  - PUT: 0 bytes
  - An immediate data field encodes `(worker_id << 16) | window_slot` (used by the client to index start time/size arrays)

Note: In RDMA terms, UD headers exist, but at the simulator’s message level you only need payload bytes to compute serialization time. Add fixed per-message overheads if desired.

---

### 4. Deterministic Workload Generation
- RNG: Implement HERD’s `hrd_fastrand` and seed each client with `0xdeadbeef` (matching code). If reproducing the per-client permutation behavior, advance the RNG `clt_gid * HERD_NUM_KEYS` times before building the permutation.
- Key choice: `key_index = fastrand(seed) % HERD_NUM_KEYS` (or from the permutation array)
- Worker selection: `worker_id = fastrand(seed) % NUM_WORKERS`
- Op type: `is_update = (fastrand(seed) % 100) < update_percentage`
- Key to 16B representation: CityHash128 of the chosen integer index; in simulators where CityHash isn’t available, either import it or stub with a stable 128-bit PRNG while keeping value-length mapping stable (see below).
- Value length for PUTs and GET-hit responses: replicate `herd_val_len_from_key_parts(part0, part1)`:
  - Inputs: the two 64-bit halves of the 128-bit key
  - Computation: `mix = part0 ^ (part1 >> 32) ^ (part1 & 0xffffffffULL)`; `val_len = min_len + (mix % (MICA_MAX_VALUE - min_len + 1))` with `min_len = 8`, `MICA_MAX_VALUE = 46` (per `mica.h` with 64B op)

---

### 5. Event Model (discrete-event workflow)
Implement the following loop behaviors as simulator events:

1) Client send event
   - Compute request size (GET or PUT) as above
   - Enqueue message to fabric destined to `(worker_id, client_id, window_slot)`
   - Record start timestamp and request bytes in `start_cycles[worker_id][slot]` and `req_bytes[worker_id][slot]`
   - Increment the worker’s window index modulo `WINDOW_SIZE`
   - Every `WINDOW_SIZE` sends, schedule a client CQ poll event at current time (or at configured polling interval) to harvest up to `WINDOW_SIZE` responses

2) Network delivery (Client → Worker)
   - Delivery time = send_time + serialization(req_bytes)/link_rate + propagation + queueing (if modeled)
   - On arrival, mark the RR entry `(worker_id, client_id, slot)` as ready with the opcode, key, and (for PUT) value bytes/len

3) Worker polling and batching
   - Each worker iterates clients in round-robin order; on each poll tick, it examines up to `NUM_CLIENTS` RR entries using the current per-client `window_slot`
   - If `opcode >= HERD_OP_GET`, translate to MICA opcode and collect into the current batch
   - Stop when `batch_size == postlist` or a full client pass is done
   - Schedule a MICA service completion event with service time = `mica_service_time(batch_size, op_mix)`
     - Simple model: `svc = base_overhead + sum_i c_get + sum_j c_put + f_batch(batch_size)`
     - Calibrate constants; see Section 7

4) Worker response posting (after MICA)
   - For each op in batch, determine response length: GET-hit → `val_len`, else 0
   - Group responses per server-port if modeling multiple ports; apply doorbell batching overhead per group
   - Enqueue UD SEND messages to fabric with immediate data `(worker_id << 16) | window_slot_used`

5) Network delivery (Worker → Client)
   - Delivery time = send_time + serialization(resp_bytes)/link_rate + propagation + queueing
   - Responses accumulate in the client’s recv CQ

6) Client CQ poll
   - Triggered every `WINDOW_SIZE` sends
   - Dequeue up to `WINDOW_SIZE` completions; for each completion:
     - Parse `(worker_id, slot)` from imm_data
     - Compute FCT = `now - start_cycles[worker_id][slot]`
     - Optionally, compute total flow size = `req_bytes[worker_id][slot] + resp_bytes`
   - Continue the client loop

---

### 6. Concurrency and Timing Knobs
- Client-side
  - `WINDOW_SIZE`: number of outstanding ops per worker
  - CQ poll gating: responses only observed at gating points; this is critical for determinism
  - `UNSIG_BATCH`: affects client send-CQ polling cadence; can be ignored for pure network timing but included for CPU/NIC time modeling

- Server-side
  - `postlist`: maximum batch per worker dispatch
  - `NUM_UD_QPS`: rotate QPs when posting batches if you model per-QP limits
  - Round-robin order over clients: use the same deterministic cycling as code

- Network
  - Link rate, base RTT, optional switch buffers/ECN/PFC for packet-level variants
  - For message-level, a single capacity queue per hop is sufficient

---

### 7. Calibration: minimal measurements to collect
Run a short benchmark on your target hardware (no full traces required) and measure:
- NIC/link parameters
  - Effective payload throughput (Gbps) for small inline sends (estimate per-message overhead if needed)
  - Baseline RTT (us) between client and server under no load

- Client/worker CPU costs
  - Per-op MICA processing cost: measure time per GET and PUT vs. batch size (e.g., microbench or instrument `mica_batch_op`)
  - Doorbell batching overhead per postlist (constant)
  - Per-send posting overhead (constant)
  - Client CQ poll cost per batch of `WINDOW_SIZE`

Suggested quick method:
1) Run HERD with logging enabled (already logs FCT and flow size at clients).
2) Use runs at low load to back out base RTT; use high-load saturation to fit CPU constants that reconcile simulated and observed throughput/latency.

Record the following calibrated constants for the simulator:
- `t_nic_post_send`, `t_doorbell`, `t_cq_poll`, `t_mica_get`, `t_mica_put`, `f_batch(postlist)` (could be linear or small lookup table)
- `link_rate_gbps`, `propagation_us`

---

### 8. Simulator Interfaces (what to implement in any framework)
- Timers/events: schedule/send/recv/poll/service completion
- Message API: enqueue(dest, bytes, metadata), dequeue()
- RNG: deterministic per-client generator and per-client key permutation
- Metrics API: record per-op FCT, throughput, queue occupancy (optional)

Suggested class/module boundaries:
- `ClientThread` (workload gen, send gating, recv processing)
- `WorkerThread` (RR polling, batching, response scheduling)
- `NetworkFabric` (message timing; optional packetization)
- `MICAService` (service-time model; optional exact hash-table for validation)
- `Stats` (histograms: FCT, throughput, response size distribution)

---

### 9. Output Metrics
- Throughput: ops/s overall and per worker
- Latency: FCT distribution (p50/p90/p99), mean
- Optional: queue sizes at client recv, server send; batch size distribution; GET-hit rate

---

### 10. Step-by-Step Implementation Workflow
1) Parse configuration: `NUM_WORKERS`, `NUM_CLIENTS`, `WINDOW_SIZE`, `postlist`, `update_percentage`, `HERD_NUM_KEYS`, link/RTT constants
2) Initialize entities:
   - Create `ClientThread[i]` with seed `0xdeadbeef`, build permutation of keys as in code
   - Create `WorkerThread[w]`, initialize per-client window indices
   - Initialize `NetworkFabric` with link rate/latency parameters
   - Initialize `MICAService` model with calibrated constants (or full logic)
3) Bootstrap event loop:
   - For each client, schedule initial sends up to `WINDOW_SIZE` per worker (or start the send loop immediately)
4) On each client send:
   - Draw op type, key, worker; compute request bytes; record start time; enqueue message
   - If `(nb_tx % WINDOW_SIZE == 0)`, schedule a client CQ poll event shortly after
5) On request arrival at a worker:
   - Mark RR entry ready; it will be picked up on the next worker poll
6) On worker poll event:
   - Round-robin over clients; collect up to `postlist` ready ops; schedule `MICAService` completion
7) On MICA completion:
   - For each op, compute response bytes; group per-port (optional); schedule UD SENDs (apply doorbell overhead once per group)
8) On response arrival at client:
   - Buffer completions and wait for the next CQ poll gate
9) On client CQ poll:
   - Process up to `WINDOW_SIZE` completions; compute FCT; emit stats
10) Continue until steady-state samples are collected; export metrics

---

### 11. Validation Checklist
- Sanity: distribution of request and response sizes matches expectations (GET-hit vlen in [8, 46], mean ~ consistent with RNG)
- Throughput vs. HERD run within ±10% under matched configuration
- FCT percentiles trend correctly as you vary `WINDOW_SIZE`, `postlist`, and `update_percentage`

---

### 12. Optional Packet-Level Enhancements
- Model each message as one UDP packet with headers (RoCE/UDP/IP/Ethernet) and payload; add header bytes to serialization
- Add ECN marking and simple DCQCN if you study congestion control
- Add PFC if you study lossless behavior and head-of-line blocking

---

### 13. Minimal Pseudocode for Critical Functions

RNG and value length:
```c
uint32_t hrd_fastrand(uint64_t* seed) {
  *seed = *seed * 1103515245 + 12345;
  return (uint32_t)(*seed >> 32);
}

uint8_t herd_val_len(uint64_t part0, uint64_t part1) {
  const uint8_t min_len = 8; const uint8_t max_len = 46;
  uint32_t range = (uint32_t)(max_len - min_len + 1);
  uint64_t mix = part0 ^ (part1 >> 32) ^ (part1 & 0xffffffffULL);
  return (uint8_t)(min_len + (mix % range));
}
```

Client request sizing:
```c
size_t request_bytes(bool is_update, uint8_t vlen) {
  return is_update ? (16 + 1 + 1 + vlen) : (16 + 1);
}

size_t response_bytes(bool is_get, bool hit, uint8_t vlen) {
  if (!is_get) return 0; // PUT
  return hit ? vlen : 0; // GET
}
```

---

Implement the above modules and event flow in your simulator of choice. Start with message-level timing and calibrated CPU constants; validate; then add packet-level details only if your study requires them.


