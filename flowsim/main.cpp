/*
File: main.cpp

Overview
This file drives two modes:
1) HERD-like client/server message simulation (default when argc < 6)
2) Original trace-driven flow simulation (when all CLI args are provided)

Chronological flow (HERD-like mode)
The HERD-like mode models a two-stage exchange per GET using a sliding window of size WINDOW_SIZE.
Stage A: Client sends GET, server replies with small UD-like 41B message. Client logs resp_recv_ud,
then immediately sends a 10B handshake. Stage B: Server replies with a large 100074B message.
Client logs resp_rdma_read, finalizes the flow, and immediately issues another GET to keep the
window full. The network is modeled by Topology (per-hop latency, bandwidth serialization). A
"Chunk" represents an entire message; multi-packet effects are abstracted as serialization delay.

Setup (main)
- Decide herd_mode based on argc.
- Create global EventQueue and set it on Topology.
- Build the topology and per-client routes.
- Initialize clients (RNG seeds, key permutations, per-worker window indices) and open logs.
- Schedule client_start_batch for each client at t0.

Client send path (initial window and per-completion refill)
1) client_start_batch
   - Prime the window by scheduling WINDOW_SIZE initial GETs.
   - Apply STARTUP_DELAY_NS before the second send to model startup delay; subsequent sends are spaced by SEND_SPACING_NS.

2) add_flow / add_flow_for_client
   - Pick worker (HERD-style RNG), op type (GET/PUT), key, and compute value/request sizes.
   - Fill FlowCtx (client, worker, slot, sizes, routes, start_time), is_handshake=false, advance slot.
   - Log req_send and send a request Chunk via topology->send with on_request_arrival as callback.
   - Topology transports the Chunk over the route applying link latency and bandwidth.

Server receive/compute/send path
3) on_request_arrival
   - Called when the request or handshake reaches the worker. Record c2s_get or c2s_handshake.
   - Schedule worker_recv immediately (network propagation already accounted for by Topology).

4) worker_recv
   - Log request reception at the worker.
   - Schedule worker_send at now + SERVER_OVERHEAD_NS.
   - Additionally logs handshake_recv when a client handshake arrives.

5) worker_send
   - Log response send and set server_send_time in ctx.
   - For GET phase: respond with RESP_UD_BYTES; for handshake phase: respond with RESP_RDMA_BYTES.
   - Send a response Chunk via topology->send with on_response_arrival as callback.

Client completion and CQ processing
6) on_response_arrival
   - Called when a response reaches the client. Dispatch to client_recv_ud (small) or
     client_recv_rdma_finalize (large) based on ctx->is_handshake.

7) client_recv_ud
   - Log resp_recv_ud, record s2c_ud timing, then immediately send a 10B handshake for the same ctx
     (is_handshake=true) using the same routes and slot.

8) client_recv_rdma_finalize
   - Log resp_rdma_read with dur_ns equal to time since handshake_send and record s2c_rdma timing.
     Free ctx, decrement inflight, and if more sends remain for the client, immediately send another
     GET to maintain the window (add_flow_for_client).

9) client_recv_finalize
   - Legacy helper used by trace mode; HERD-like path uses client_recv_ud and client_recv_rdma_finalize.

Handling multiple packets / sizes
- This file does not split messages into packets explicitly. A Chunk’s size determines its
  serialization time on each link (size / bandwidth). Per-packet behavior is abstracted away.
- Windowing/pipelining is modeled via per-client batches (WINDOW_SIZE) and per-worker ring slots,
  allowing multiple in-flight requests per client without per-packet simulation.

Chronological flow (Trace-driven mode)
1) main
   - Load arrival times and flow sizes from .npy files, construct topology, and read routes.
   - Schedule add_flow_trace for the first arrival time.
2) add_flow_trace
   - Log the arrival, create a Chunk for the flow size, and send via topology->send with
     record_fct_trace as callback. Schedule the next arrival (schedule_next_arrival).
3) record_fct_trace
   - Record FCT as (now - arrival_time) and print it.

Outputs
- HERD-like mode writes per-client logs (client_*.log), a server log (server.log), and a consolidated
  per-stage flow timeline to flows.txt.
- Trace-driven mode writes FCTs back to a .npy file specified on the CLI.
*/

// NOTE: This file now supports two modes:
// 1) Original trace-driven flow mode (if enough CLI args are provided)
// 2) HERD-like client/worker message simulation (default fallback; 1 client, 1 worker)

#include "npy.hpp"
#include "Topology.h"
#include "TopologyBuilder.h"
#include "Type.h"
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <cassert>
#include <cstring>
#include <memory>
#include <iostream>
#include <fstream>
#include <deque>

// CityHash for 128-bit key generation (exactly as in HERD)
#include "rdma_bench/mica/mica.h" // Brings in city.h and MICA_MAX_VALUE

std::shared_ptr<EventQueue> event_queue;
std::shared_ptr<Topology> topology;
std::vector<Route> routing; // forward routes per op (used by trace mode)
std::vector<int64_t> fat;   // arrival times
std::vector<int64_t> fsize; // flow sizes (unused in HERD mode)
std::unordered_map<int, int64_t> fct; // op index -> FCT (ns)
uint64_t limit;

// ======================== HERD-like simulation bits ========================
// Minimal constants mirroring herd/main.h
static constexpr int HERD_NUM_KEYS = (8 * 1024 * 1024);
static constexpr int NUM_WORKERS = 12;
static int NUM_CLIENTS = 1; // configurable per-topology mode
static constexpr int WINDOW_SIZE = 16; // per-worker ring slots per client

// New protocol constants for staged exchange
static constexpr uint64_t RESP_UD_BYTES = 41;       // Server's first small response
static constexpr uint64_t HANDSHAKE_BYTES = 10;     // Client's handshake payload size
static constexpr uint64_t RESP_RDMA_BYTES = 1024008; // Server's large variable response

// Callback when a request arrives at the server. Immediately send a response.
// Tunables for extra timing (ns)
// Disable explicit propagation in main; rely on Topology link latency instead
static constexpr uint64_t SERVER_OVERHEAD_NS = 24267; // 400 us between recv and send
static constexpr uint64_t SEND_SPACING_NS = 2500;     // Inter-send spacing within a batch
static constexpr uint64_t STARTUP_DELAY_NS = 0;       // Extra delay between first and second initial sends
static constexpr uint64_t HANDSHAKE_DELAY_NS = 8647;  // Delay between resp_recv_ud and handshake_send


// RNG exactly as in libhrd/hrd.h
static inline uint32_t hrd_fastrand(uint64_t* seed) {
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t)(*seed >> 32);
}

// Value length derivation exactly matching herd/client.c & worker.c
static inline uint8_t herd_val_len_from_key_parts(uint64_t part0, uint64_t part1) {
    const uint8_t min_len = 8;
    const uint8_t max_len = MICA_MAX_VALUE; // 46 bytes for 64B mica_op
    const uint32_t range = (uint32_t)(max_len - min_len + 1);
    uint64_t mix = part0 ^ (part1 >> 32) ^ (part1 & 0xffffffffULL);
    return (uint8_t)(min_len + (mix % range));
}

// Generate random permutation of [0, n-1] as in herd/client.c
static int* herd_get_random_permutation(int n, int clt_gid, uint64_t* seed) {
    assert(n > 0);
    for (int i = 0; i < clt_gid * HERD_NUM_KEYS; i++) {
        hrd_fastrand(seed);
    }
    int* log = (int*)malloc(n * sizeof(int));
    assert(log != nullptr);
    for (int i = 0; i < n; i++) log[i] = i;
    for (int i = n - 1; i >= 1; i--) {
        int j = (int)(hrd_fastrand(seed) % (uint32_t)(i + 1));
        int temp = log[i];
        log[i] = log[j];
        log[j] = temp;
    }
    return log;
}

struct ClientState {
    int id;
    uint64_t seed;
    int* key_perm; // permutation array of size HERD_NUM_KEYS
    int ws[NUM_WORKERS]; // per-worker window slot ring index

    ClientState() : id(0), seed(0xdeadbeef), key_perm(nullptr) {
        for (int i = 0; i < NUM_WORKERS; i++) ws[i] = 0;
    }
};

struct FlowCtx {
    int op_index;    // global op index used for FCT output
    int client_id;   // 0
    int worker_id;   // 0
    int slot;        // window slot used (0..WINDOW_SIZE-1)
    bool is_update;  // PUT vs GET
    uint8_t vlen;    // value length for PUT or GET-hit
    uint64_t req_bytes;
    uint64_t resp_bytes;
    bool is_handshake; // false: GET phase; true: handshake phase
    EventTime start_time;
    EventTime server_send_time; // for s->c stage FCT
    EventTime handshake_send_time; // when client sent the handshake
    Route route_fwd; // client -> worker
    Route route_rev; // worker -> client
};

struct FlowRecord {
    int op_index;
    int client_id;
    int worker_id;
    int slot;
    uint64_t req_bytes;
    uint64_t resp_bytes;
    EventTime start_ns;
    EventTime end_ns;
    uint64_t fct_ns;
    std::string stage; // "c2s" or "s2c"
};

static std::vector<ClientState> g_clients;
static std::vector<Route> g_routes_c2s; // per-client route to server
static std::vector<Route> g_routes_s2c; // per-client route back
static std::vector<FlowRecord> g_flow_records;
static std::vector<std::deque<FlowCtx*>> g_pending_completions; // per-client CQ buffers
static std::vector<bool> g_poll_scheduled; // per-client poll scheduled flag
static std::vector<uint64_t> g_total_sent_per_client; // per-client total sends
static std::vector<uint64_t> g_client_limit; // per-client send limit
static std::vector<uint32_t> g_batch_inflight_per_client; // per-client window in flight (legacy; unused)
static std::vector<uint32_t> g_inflight_per_client; // per-client active flows (sliding window)
static uint64_t g_next_op_index = 0; // global unique op index
static std::vector<std::ofstream> g_client_logs; // one per client
static std::ofstream g_server_log;

// Forward decls
static void add_flow(void* index_ptr_v);
static void on_request_arrival(void* arg);
static void on_response_arrival(void* arg);
static void worker_recv(void* arg);
static void worker_send(void* arg);
static void client_recv_finalize(void* arg);
static void client_recv_ud(void* arg);
static void client_recv_rdma_finalize(void* arg);
static void client_send_handshake(void* arg);
static void client_cq_poll(void* arg);
static void client_completion_ready(void* arg);
static void client_start_batch(void* arg);
static void add_flow_for_client(void* client_id_ptr);

static void on_request_arrival(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Determine response size for current phase
    // Phase 1 (GET): server responds with small UD-sized message
    // Phase 2 (handshake): server responds with large RDMA-sized message
    if (ctx->is_handshake) {
        ctx->resp_bytes = RESP_RDMA_BYTES;
    } else {
        ctx->resp_bytes = RESP_UD_BYTES;
    }
    // Record client->server FCT (c2s stage)
    {
        FlowRecord rec;
        rec.op_index = ctx->op_index;
        rec.client_id = ctx->client_id;
        rec.worker_id = ctx->worker_id;
        rec.slot = ctx->slot;
        rec.req_bytes = ctx->req_bytes;
        rec.resp_bytes = 0;
        rec.start_ns = ctx->start_time;
        rec.end_ns = event_queue->get_current_time();
        rec.fct_ns = (uint64_t)(rec.end_ns - rec.start_ns);
        rec.stage = ctx->is_handshake ? "c2s_handshake" : "c2s_get";
        g_flow_records.push_back(rec);
    }
    // No extra propagation here; Topology models per-hop latency
    EventTime when = event_queue->get_current_time();
    event_queue->schedule_event(when, (void (*)(void*)) &worker_recv, ctx);
}

static void worker_recv(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log request receive at worker (after propagation)
    EventTime when;
    if (!ctx->is_handshake) {
        g_server_log << "event=reqq_recv ts_ns=" << event_queue->get_current_time()
                     << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id
                     << " wrkr=" << ctx->worker_id
                     << " slot=" << ctx->slot
                     << " size=" << ctx->req_bytes
                     << " src=client:" << ctx->client_id
                     << " dst=worker:" << ctx->worker_id
                     << "\n";
        when = event_queue->get_current_time() + SERVER_OVERHEAD_NS;

    } else {
        g_server_log << "event=hand_recv ts_ns=" << event_queue->get_current_time()
                     << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id
                     << " wrkr=" << ctx->worker_id
                     << " slot=" << ctx->slot
                     << " size=" << ctx->req_bytes
                     << " src=client:" << ctx->client_id
                     << " dst=worker:" << ctx->worker_id
                     << "\n";
        when = event_queue->get_current_time(); //no overhead for handshake (RDMA DMA technique)
    }
    // Schedule worker send after overhead
    event_queue->schedule_event(when, (void (*)(void*)) &worker_send, ctx);
}

static void worker_send(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log response send from worker
    // If responding to a handshake, also log a handshake_send event on the server
    if (ctx->is_handshake) {
        g_server_log << "event=hand_conf ts_ns=" << event_queue->get_current_time()
                     << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id
                     << " wrkr=" << ctx->worker_id
                     << " slot=" << ctx->slot
                     << " size=" << ctx->resp_bytes
                     << " src=worker:" << ctx->worker_id
                     << " dst=client:" << ctx->client_id
                     << "\n";
    } else {
        g_server_log << "event=resp_send ts_ns=" << event_queue->get_current_time()
        << " id=" << ctx->op_index
        << " clt=" << ctx->client_id
        << " wrkr=" << ctx->worker_id
        << " slot=" << ctx->slot
        << " size=" << ctx->resp_bytes
        << " src=worker:" << ctx->worker_id
        << " dst=client:" << ctx->client_id
        << "\n";
    }
    ctx->server_send_time = event_queue->get_current_time();
    auto resp_chunk = std::make_unique<Chunk>(ctx->resp_bytes, ctx->route_rev,
                                              (void (*)(void*)) &on_response_arrival,
                                              ctx);
    topology->send(std::move(resp_chunk));
}

static void on_response_arrival(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    EventTime when = event_queue->get_current_time();
    // Route to the appropriate client receive handler based on phase
    if (ctx->is_handshake) {
        event_queue->schedule_event(when, (void (*)(void*)) &client_recv_rdma_finalize, ctx);
    } else {
        event_queue->schedule_event(when, (void (*)(void*)) &client_recv_ud, ctx);
    }
}

static void client_recv_finalize(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log generic finalize (legacy path)
    g_client_logs[ctx->client_id] << "event=resp_recv ts_ns=" << event_queue->get_current_time()
                 << " id=" << ctx->op_index
                 << " clt=" << ctx->client_id
                 << " wrkr=" << ctx->worker_id
                 << " slot=" << ctx->slot
                 << " size=" << ctx->resp_bytes
                 << " src=worker:" << ctx->worker_id
                 << " dst=client:" << ctx->client_id
                 << "\n";
    // Record flow completion at client read time (CQ processed)
    {
        int64_t fct_value = (int64_t)(event_queue->get_current_time() - ctx->server_send_time);
        fct[ctx->op_index] = fct_value;
        FlowRecord rec;
        rec.op_index = ctx->op_index;
        rec.client_id = ctx->client_id;
        rec.worker_id = ctx->worker_id;
        rec.slot = ctx->slot;
        rec.req_bytes = 0;
        rec.resp_bytes = ctx->resp_bytes;
        rec.start_ns = ctx->server_send_time;
        rec.end_ns = event_queue->get_current_time();
        rec.fct_ns = (uint64_t)fct_value;
        rec.stage = "s2c";
        g_flow_records.push_back(rec);
    }
    delete ctx;
}

// Client receives the small UD-style response and immediately sends handshake
static void client_recv_ud(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log: resp_recv_ud
    g_client_logs[ctx->client_id] << "event=resp_recv_ud ts_ns=" << event_queue->get_current_time()
                 << " id=" << ctx->op_index
                 << " clt=" << ctx->client_id
                 << " wrkr=" << ctx->worker_id
                 << " slot=" << ctx->slot
                 << " size=" << ctx->resp_bytes
                 << " src=worker:" << ctx->worker_id
                 << " dst=client:" << ctx->client_id
                 << "\n";
    // Record s2c stage for UD response
    {
        FlowRecord rec;
        rec.op_index = ctx->op_index;
        rec.client_id = ctx->client_id;
        rec.worker_id = ctx->worker_id;
        rec.slot = ctx->slot;
        rec.req_bytes = 0;
        rec.resp_bytes = ctx->resp_bytes;
        rec.start_ns = ctx->server_send_time;
        rec.end_ns = event_queue->get_current_time();
        rec.fct_ns = (uint64_t)(rec.end_ns - rec.start_ns);
        rec.stage = "s2c_ud";
        g_flow_records.push_back(rec);
    }
    // Schedule handshake after HANDSHAKE_DELAY_NS
    ctx->is_handshake = true;
    ctx->req_bytes = HANDSHAKE_BYTES;
    EventTime when = event_queue->get_current_time() + HANDSHAKE_DELAY_NS;
    event_queue->schedule_event(when, (void (*)(void*)) &client_send_handshake, ctx);
}

// Client receives the large RDMA-style response, finalizes, and slides the window
static void client_recv_rdma_finalize(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log: resp_rdma_read
    uint64_t now_ns = (uint64_t)event_queue->get_current_time();
    uint64_t dur_ns = (ctx->handshake_send_time <= now_ns) ? (now_ns - (uint64_t)ctx->handshake_send_time) : 0;
    g_client_logs[ctx->client_id] << "event=resp_rdma_read ts_ns=" << now_ns
                 << " id=" << ctx->op_index
                 << " start_ns=" << ctx->handshake_send_time
                 << " dur_ns=" << dur_ns
                 << " clt=" << ctx->client_id
                 << " wrkr=" << ctx->worker_id
                 << " slot=" << ctx->slot
                 << " size=" << ctx->resp_bytes
                 << " src=worker:" << ctx->worker_id
                 << " dst=client:" << ctx->client_id
                 << "\n";
    // Record s2c stage for RDMA-sized response and finalize FCT
    {
        int64_t fct_value = (int64_t)(event_queue->get_current_time() - ctx->server_send_time);
        fct[ctx->op_index] = fct_value;
        FlowRecord rec;
        rec.op_index = ctx->op_index;
        rec.client_id = ctx->client_id;
        rec.worker_id = ctx->worker_id;
        rec.slot = ctx->slot;
        rec.req_bytes = 0;
        rec.resp_bytes = ctx->resp_bytes;
        rec.start_ns = ctx->server_send_time;
        rec.end_ns = event_queue->get_current_time();
        rec.fct_ns = (uint64_t)fct_value;
        rec.stage = "s2c_rdma";
        g_flow_records.push_back(rec);
    }
    int client_id = ctx->client_id;
    // Completion reduces inflight window by one
    if (g_inflight_per_client[client_id] > 0) g_inflight_per_client[client_id]--;
    delete ctx;
    // Slide the window: issue a new GET if under limit
    if (g_total_sent_per_client[client_id] < g_client_limit[client_id]) {
        int* cid = (int*)malloc(sizeof(int)); *cid = client_id;
        event_queue->schedule_event(event_queue->get_current_time(), (void (*)(void*)) &add_flow_for_client, cid);
    }
}

// Client sends the handshake after the configured delay
static void client_send_handshake(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    ctx->start_time = event_queue->get_current_time();
    ctx->handshake_send_time = ctx->start_time;
    g_client_logs[ctx->client_id] << "event=hand_send ts_ns=" << ctx->start_time
                 << " id=" << ctx->op_index
                 << " clt=" << ctx->client_id
                 << " wrkr=" << ctx->worker_id
                 << " slot=" << ctx->slot
                 << " size=" << ctx->req_bytes
                 << " src=client:" << ctx->client_id
                 << " dst=worker:" << ctx->worker_id
                 << "\n";
    auto hs_chunk = std::make_unique<Chunk>(ctx->req_bytes, ctx->route_fwd,
                                            (void (*)(void*)) &on_request_arrival,
                                            ctx);
    topology->send(std::move(hs_chunk));
}

static void client_completion_ready(void* arg) {
    // Legacy path unused in sliding-window mode
    auto* ctx = static_cast<FlowCtx*>(arg);
    event_queue->schedule_event(event_queue->get_current_time(), (void (*)(void*)) &client_recv_rdma_finalize, ctx);
}

static void client_cq_poll(void* arg) {
    // Legacy path unused in sliding-window mode
    free(arg);
}

static void schedule_next_arrival(int index) {
    int* index_ptr = (int*)malloc(sizeof(int));
    *index_ptr = index;
    event_queue->schedule_event((EventTime)fat.at(index), (void (*)(void*)) &add_flow, index_ptr);
}

static void client_start_batch(void* arg) {
    // Prime sliding window: issue WINDOW_SIZE GETs per client at start
    int client_id = arg ? *(int*)arg : 0; if (arg) free(arg);
    EventTime base = event_queue->get_current_time();
    uint32_t to_send = WINDOW_SIZE;
    for (uint32_t i = 0; i < to_send && g_total_sent_per_client[client_id] < g_client_limit[client_id]; i++) {
        int* cid = (int*)malloc(sizeof(int)); *cid = client_id;
        // Apply a one-time startup delay between the first and second send
        EventTime extra = (i > 0 ? STARTUP_DELAY_NS : 0);
        EventTime send_time = base + extra + (EventTime)(i * SEND_SPACING_NS);
        event_queue->schedule_event(send_time, (void (*)(void*)) &add_flow_for_client, cid);
    }
}

// Schedules and sends one HERD request (client -> worker)
static void add_flow(void* index_ptr_v) {
    int* index_ptr = static_cast<int*>(index_ptr_v);
    const int op_index = *index_ptr;
    free(index_ptr);

    // Choose worker randomly (HERD-style) using per-client RNG
    int curr_client = (int)(op_index % NUM_CLIENTS);
    int wn = (int)(hrd_fastrand(&g_clients[curr_client].seed) % (uint32_t)NUM_WORKERS);
    // Choose op type and key
    int is_update = 0; // PUT ratio 0% by default; adjust if needed
    int key_i = (int)(hrd_fastrand(&g_clients[curr_client].seed) % (uint32_t)HERD_NUM_KEYS);
    uint128 key128 = CityHash128((char*)&g_clients[curr_client].key_perm[key_i], 4);
    uint64_t part0 = key128.first;
    uint64_t part1 = key128.second;
    uint8_t vlen = herd_val_len_from_key_parts(part0, part1);

    uint64_t req_bytes = is_update ? (uint64_t)(16 + 1 + 1 + vlen) : (uint64_t)(16 + 1);

    // Prepare flow context
    auto* ctx = new FlowCtx();
    ctx->op_index = op_index;
    // Round-robin assign logical client id per op across NUM_CLIENTS
    ctx->client_id = (int)(op_index % NUM_CLIENTS);
    ctx->worker_id = wn;
    ctx->slot = g_clients[ctx->client_id].ws[wn];
    ctx->is_update = (is_update != 0);
    ctx->vlen = vlen;
    ctx->req_bytes = req_bytes;
    ctx->is_handshake = false;
    ctx->route_fwd = g_routes_c2s[ctx->client_id];
    ctx->route_rev = g_routes_s2c[ctx->client_id];
    ctx->start_time = event_queue->get_current_time();

    // Advance the window slot as in HERD
    g_clients[ctx->client_id].ws[wn] = (g_clients[ctx->client_id].ws[wn] + 1) % WINDOW_SIZE;

    // Send request
    g_client_logs[ctx->client_id] << "event=reqq_send ts_ns=" << event_queue->get_current_time()
                 << " id=" << ctx->op_index
                 << " clt=" << ctx->client_id
                 << " wrkr=" << ctx->worker_id
                 << " slot=" << ctx->slot
                 << " size=" << ctx->req_bytes
                 << " src=client:" << ctx->client_id
                 << " dst=worker:" << ctx->worker_id
                 << "\n";
    auto req_chunk = std::make_unique<Chunk>(ctx->req_bytes, ctx->route_fwd,
                                             (void (*)(void*)) &on_request_arrival,
                                             ctx);
    topology->send(std::move(req_chunk));

    // No auto-scheduling here; batches control when sends occur
}

// Issue one GET for a specific client to maintain sliding window
static void add_flow_for_client(void* client_id_ptr) {
    int client_id = *(int*)client_id_ptr; free(client_id_ptr);
    int op_index = (int)g_next_op_index; g_next_op_index++;

    int wn = (int)(hrd_fastrand(&g_clients[client_id].seed) % (uint32_t)NUM_WORKERS);
    int is_update = 0;
    int key_i = (int)(hrd_fastrand(&g_clients[client_id].seed) % (uint32_t)HERD_NUM_KEYS);
    uint128 key128 = CityHash128((char*)&g_clients[client_id].key_perm[key_i], 4);
    uint64_t part0 = key128.first;
    uint64_t part1 = key128.second;
    uint8_t vlen = herd_val_len_from_key_parts(part0, part1);
    uint64_t req_bytes = is_update ? (uint64_t)(16 + 1 + 1 + vlen) : (uint64_t)(16 + 1);

    auto* ctx = new FlowCtx();
    ctx->op_index = op_index;
    ctx->client_id = client_id;
    ctx->worker_id = wn;
    ctx->slot = g_clients[ctx->client_id].ws[wn];
    ctx->is_update = (is_update != 0);
    ctx->vlen = vlen;
    ctx->req_bytes = req_bytes;
    ctx->is_handshake = false;
    ctx->route_fwd = g_routes_c2s[ctx->client_id];
    ctx->route_rev = g_routes_s2c[ctx->client_id];
    ctx->start_time = event_queue->get_current_time();

    g_clients[ctx->client_id].ws[wn] = (g_clients[ctx->client_id].ws[wn] + 1) % WINDOW_SIZE;

    g_client_logs[ctx->client_id] << "event=req_send ts_ns=" << event_queue->get_current_time()
                 << " id=" << ctx->op_index
                 << " clt=" << ctx->client_id
                 << " wrkr=" << ctx->worker_id
                 << " slot=" << ctx->slot
                 << " size=" << ctx->req_bytes
                 << " src=client:" << ctx->client_id
                 << " dst=worker:" << ctx->worker_id
                 << "\n";
    auto req_chunk = std::make_unique<Chunk>(ctx->req_bytes, ctx->route_fwd,
                                             (void (*)(void*)) &on_request_arrival,
                                             ctx);
    topology->send(std::move(req_chunk));

    g_total_sent_per_client[client_id]++;
    g_inflight_per_client[client_id]++;
}

// ======================= Trace-driven original mode ========================
// Original record_fct for trace mode kept for compatibility (unused in HERD)
static void record_fct_trace(void* index_v) {
    int* index = static_cast<int*>(index_v);
    int64_t fct_value = (int64_t)(event_queue->get_current_time() - fat.at(*index));
    fct[*index] = fct_value;
    std::cout << "fct " << *index << " " << fct_value << "\n";
    free(index);
}

static void add_flow_trace(void* index_ptr_v) {
    int* index_ptr = static_cast<int*>(index_ptr_v);
    std::cout << "flow arrival " << *index_ptr << " " << fsize.at(*index_ptr) << "\n";
    Route route = routing.at(*index_ptr);
    int64_t flow_size = fsize.at(*index_ptr);
    auto chunk = std::make_unique<Chunk>((uint64_t)flow_size, route, (void (*)(void*)) &record_fct_trace, index_ptr);
    topology->send(std::move(chunk));
    if (*index_ptr + 1 < (int)limit) {
        schedule_next_arrival(*index_ptr + 1);
    }
}

int main(int argc, char* argv[]) {
    bool herd_mode = false;

    // If not enough args for trace mode, switch to HERD single-link mode
    if (argc < 6) herd_mode = true;

    event_queue = std::make_shared<EventQueue>();
    Topology::set_event_queue(event_queue);

    if (!herd_mode) {
        // =============== Original trace-driven flow completion path ===============
    const std::string fat_path = argv[1];
    const std::string fsize_path = argv[2];
    const std::string topo_path = argv[3];
    const std::string routing_path = argv[4];
    const std::string write_path = argv[5];
 
    npy::npy_data d_fat = npy::read_npy<int64_t>(fat_path);
    std::vector<int64_t> arrival_times = d_fat.data;

    npy::npy_data d_fsize = npy::read_npy<int64_t>(fsize_path);
    std::vector<int64_t> flow_sizes = d_fsize.data;

    topology = construct_fat_tree_topology(topo_path);

    std::filesystem::path cwd = std::filesystem::current_path() / routing_path;
    std::ifstream infile(cwd);
    int num_hops;

    limit = arrival_times.size();

    while (infile >> num_hops) {
        auto route = Route();
        for (int i = 0; i < num_hops; i++) {
            int64_t device_id;
            infile >> device_id;
                route.push_back(topology->get_device((int)device_id));
        }
        routing.push_back(route);
    }

        for (int i = 0; i < (int)arrival_times.size() && (uint64_t)i < limit; i++) {
        fat.push_back(arrival_times.at(i));
            fsize.push_back(flow_sizes.at(i));
        }

        int* index_ptr = (int*)malloc(sizeof(int));
    *index_ptr = 0;
        event_queue->schedule_event((EventTime)fat.at(0), (void (*)(void*)) &add_flow_trace, index_ptr);

    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    std::vector<int64_t> fct_vector;
        for (int i = 0; i < (int)arrival_times.size() && (uint64_t)i < limit; i++) {
        fct_vector.push_back(fct[i]);
    }

    npy::npy_data<int64_t> d;
    d.data = fct_vector;
    d.shape = {limit};
    d.fortran_order = false;
    npy::write_npy(write_path, d);
        return 0;
    }

    // ====================== HERD single client/worker mode =======================
    // Topology switch: single client/server or multi-client tree
    bool multi_client_topo = false; // flip to false for single-link topology
    double bw_bpns = 10.0;
    if (!multi_client_topo) {
        // Single client (0) <-> server (1)
        topology = std::make_shared<Topology>(2, 2);
        //setting prop delay to 1 instead of 3500
        topology->connect(0, 1, bw_bpns, 3500 , true);
        NUM_CLIENTS = 1;
        g_clients.assign(NUM_CLIENTS, ClientState());
        g_client_logs.resize(NUM_CLIENTS);
        for (int i = 0; i < NUM_CLIENTS; i++) {
            g_clients[i].id = i;
            g_clients[i].key_perm = herd_get_random_permutation(HERD_NUM_KEYS, i, &g_clients[i].seed);
        }
        g_routes_c2s.resize(NUM_CLIENTS);
        g_routes_s2c.resize(NUM_CLIENTS);
        // Build routes for client 0
        Route c2s; c2s.push_back(topology->get_device(0)); c2s.push_back(topology->get_device(1));
        Route s2c; s2c.push_back(topology->get_device(1)); s2c.push_back(topology->get_device(0));
        g_routes_c2s[0] = c2s; g_routes_s2c[0] = s2c;
        g_client_logs[0].open("client_0.log");
    } else {
        // Multi-client: Server A (0) -> S (1) -> Root R (2)
        // Client B (3) -> S1 (7) -> R (2)
        // Client C (4) -> S2 (5) -> R (2)
        // Client D (6) -> S2 (5) -> R (2)
        topology = std::make_shared<Topology>(8, 4); // 8 devices total
        // Links with 2us latency and 10B/ns bandwidth
        topology->connect(0, 1, bw_bpns, 800.0 , true); // A<->S
        topology->connect(1, 2, bw_bpns, 800.0 , true); // S<->R
        topology->connect(3, 7, bw_bpns, 800.0 , true); // B<->S1
        topology->connect(4, 5, bw_bpns, 800.0 , true); // C<->S2
        topology->connect(5, 2, bw_bpns, 800.0 , true); // S2<->R
        topology->connect(6, 5, bw_bpns, 800.0 , true); // D<->S2
        topology->connect(7, 2, bw_bpns, 800.0 , true); // S1<->R
        NUM_CLIENTS = 3;
        g_clients.assign(NUM_CLIENTS, ClientState());
        g_client_logs.resize(NUM_CLIENTS);
        for (int i = 0; i < NUM_CLIENTS; i++) {
            g_clients[i].id = i;
            g_clients[i].key_perm = herd_get_random_permutation(HERD_NUM_KEYS, i, &g_clients[i].seed);
        }
        g_routes_c2s.resize(NUM_CLIENTS);
        g_routes_s2c.resize(NUM_CLIENTS);
        // Map client indices with full hop-by-hop routes:
        // 0: B(3)->S1(7)->R(2)->S(1)->A(0)
        // 1: C(4)->S2(5)->R(2)->S(1)->A(0)
        // 2: D(6)->S2(5)->R(2)->S(1)->A(0)
        auto build_route = [&](std::vector<int> path){ Route r; for(int id: path) r.push_back(topology->get_device(id)); return r; };
        g_routes_c2s[0] = build_route({3,7,2,1,0}); g_routes_s2c[0] = build_route({0,1,2,7,3});
        g_routes_c2s[1] = build_route({4,5,2,1,0}); g_routes_s2c[1] = build_route({0,1,2,5,4});
        g_routes_c2s[2] = build_route({6,5,2,1,0}); g_routes_s2c[2] = build_route({0,1,2,5,6});
        g_client_logs[0].open("client_0.log");
        g_client_logs[1].open("client_1.log");
        g_client_logs[2].open("client_2.log");
    }

    // Open server log
    g_server_log.open("server.log");

    // Generate simple arrival sequence: N ops spaced by 1000 ns (start at 1ns to satisfy strict > current_time)
    const int default_ops = 650; // total ops (will be divided across clients)
    limit = (uint64_t)default_ops;
    fat.clear(); fat.reserve(default_ops);
    //for (int i = 0; i < default_ops; i++) fat.push_back((int64_t)i * 2500 + 1);
    
    // Initialize per-client limits and state
    g_pending_completions.assign(NUM_CLIENTS, std::deque<FlowCtx*>());
    g_poll_scheduled.assign(NUM_CLIENTS, false);
    g_total_sent_per_client.assign(NUM_CLIENTS, 0);
    g_client_limit.assign(NUM_CLIENTS, default_ops / NUM_CLIENTS);
    g_batch_inflight_per_client.assign(NUM_CLIENTS, 0);
    g_inflight_per_client.assign(NUM_CLIENTS, 0);

    // Kick off: start initial sliding window for each client at t0
    EventTime start_time = event_queue->get_current_time() + 1;
    for (int cid = 0; cid < NUM_CLIENTS; cid++) {
        int* arg_c = (int*)malloc(sizeof(int)); *arg_c = cid;
        event_queue->schedule_event(start_time, (void (*)(void*)) &client_start_batch, arg_c);
    }

    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    // Cleanup
    // Cleanup client key perms
    for (int i = 0; i < (int)g_clients.size(); i++) {
        if (g_clients[i].key_perm != nullptr) {
            free(g_clients[i].key_perm);
            g_clients[i].key_perm = nullptr;
        }
    }

    // Close logs and write flows to flows.txt
    {
        for (auto& cl : g_client_logs) if (cl.is_open()) cl.close();
        if (g_server_log.is_open()) g_server_log.close();
        std::ofstream ofs("flows.txt");
        for (const auto& r : g_flow_records) {
            ofs << r.op_index << " "
                << r.client_id << " "
                << r.worker_id << " "
                << r.slot << " "
                << r.req_bytes << " "
                << r.resp_bytes << " "
                << r.start_ns << " "
                << r.end_ns << " "
                << r.fct_ns << " "
                << r.stage << "\n";
        }
    }
    std::cout << "HERD-sim completed ops: " << fct.size() << ", wrote flows.txt\n";
    return 0;
}

