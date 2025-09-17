#include "npy.hpp"
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>
#include <ryml_std.hpp>
#include <ryml.hpp>
#include "Topology.h"
#include "TopologyBuilder.h"
#include "Type.h"

#include <iomanip>
#include <queue>

// CityHash for 128-bit key generation (exactly as in HERD)
#include "rdma_bench/mica/mica.h" // Brings in city.h and MICA_MAX_VALUE

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
static constexpr uint64_t SERVER_OVERHEAD_NS = 24267; // flowsim control-path server delay
static constexpr uint64_t SEND_SPACING_NS = 2500;     // Inter-send spacing within a batch
static constexpr uint64_t STARTUP_DELAY_NS = 60346 - SEND_SPACING_NS;       // Extra delay between first and second initial sends
static constexpr uint64_t HANDSHAKE_DELAY_NS = 8647;  // flowsim control-path client delay

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
    // Advance RNG based on client ID to avoid correlation
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
// HERD record aggregation (flows.txt)
static std::vector<std::string> herd_flow_records;
// Shared pointers to use inside callbacks
static std::shared_ptr<EventQueue> g_event_queue;
static std::shared_ptr<Topology> g_topology;
// ML+HERD dynamic arrival control
static int g_ops_limit = 0;
static int g_next_to_schedule = 0;

// Emit flowsim-style staged records based on ML-predicted FCT
static void generate_herd_messages_m4(int flow_id, uint64_t flow_start_ns, uint64_t ml_fct_ns, uint64_t large_size_bytes) {
    // Stage ordering and simple timing offsets to mirror HERD logs
    uint64_t req_send_time = flow_start_ns;                    // client sends 17B GET
    uint64_t req_recv_time = req_send_time + 3513;             // network c2s
    uint64_t resp_send_time = req_recv_time + SERVER_OVERHEAD_NS; // server compute
    uint64_t resp_recv_time = resp_send_time + 3540;           // network s2c small
    uint64_t handshake_send_time = resp_recv_time + HANDSHAKE_DELAY_NS; // client handshake emit
    uint64_t large_completion_time = flow_start_ns + ml_fct_ns; // overall completion driven by ML

    // Client/server logs (single client 0, worker 0)
    if (g_client_logs.empty()) g_client_logs.resize(1);
    if (!g_client_logs[0].is_open()) g_client_logs[0].open("client_0.log");
    if (!g_server_log.is_open()) g_server_log.open("server.log");

    g_client_logs[0] << "event=req_send ts_ns=" << req_send_time << " id=" << flow_id
                     << " clt=0 wrkr=0 slot=0 size=17 src=client:0 dst=worker:0\n";
    g_server_log     << "event=reqq_recv ts_ns=" << req_recv_time << " id=" << flow_id
                     << " clt=0 wrkr=0 slot=0 size=17 src=client:0 dst=worker:0\n";
    g_server_log     << "event=resp_send ts_ns=" << resp_send_time << " id=" << flow_id
                     << " clt=0 wrkr=0 slot=0 size=" << RESP_UD_BYTES << " src=worker:0 dst=client:0\n";
    g_client_logs[0] << "event=resp_recv_ud ts_ns=" << resp_recv_time << " id=" << flow_id
                     << " clt=0 wrkr=0 slot=0 size=" << RESP_UD_BYTES << " src=worker:0 dst=client:0\n";
    g_client_logs[0] << "event=hand_send ts_ns=" << handshake_send_time << " id=" << flow_id
                     << " clt=0 wrkr=0 slot=0 size=10 src=client:0 dst=worker:0\n";
    g_server_log     << "event=hand_recv ts_ns=" << handshake_send_time + 3508 << " id=" << flow_id
                     << " clt=0 wrkr=0 slot=0 size=10 src=client:0 dst=worker:0\n";
    g_server_log     << "event=hand_conf ts_ns=" << handshake_send_time + 3508 << " id=" << flow_id
                     << " clt=0 wrkr=0 slot=0 size=" << large_size_bytes << " src=worker:0 dst=client:0\n";
    g_client_logs[0] << "event=resp_rdma_read ts_ns=" << large_completion_time << " id=" << flow_id
                     << " start_ns=" << handshake_send_time
                     << " dur_ns=" << (large_completion_time - handshake_send_time)
                     << " clt=0 wrkr=0 slot=0 size=" << large_size_bytes << " src=worker:0 dst=client:0\n";

    // flows.txt staged records (4 lines per op)
    herd_flow_records.push_back(std::to_string(flow_id) + " 0 0 0 17 0 " + std::to_string(req_send_time) + " " + std::to_string(req_recv_time) + " " + std::to_string(req_recv_time - req_send_time) + " c2s_get");
    herd_flow_records.push_back(std::to_string(flow_id) + " 0 0 0 0 " + std::to_string(RESP_UD_BYTES) + " " + std::to_string(resp_send_time) + " " + std::to_string(resp_recv_time) + " " + std::to_string(resp_recv_time - resp_send_time) + " s2c_ud");
    herd_flow_records.push_back(std::to_string(flow_id) + " 0 0 0 10 0 " + std::to_string(handshake_send_time) + " " + std::to_string(handshake_send_time + 3508) + " 3508 c2s_handshake");
    herd_flow_records.push_back(std::to_string(flow_id) + " 0 0 0 0 " + std::to_string(large_size_bytes) + " " + std::to_string(handshake_send_time + 3508) + " " + std::to_string(large_completion_time) + " " + std::to_string(large_completion_time - (handshake_send_time + 3508)) + " s2c_rdma");
}

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

// HERD callbacks (adapting flowsim semantics to inference Topology/EventQueue)
static void on_request_arrival(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Decide response size by phase
    ctx->resp_bytes = ctx->is_handshake ? RESP_RDMA_BYTES : RESP_UD_BYTES;
    // Record c2s in flows.txt buffer
    FlowRecord rec{ctx->op_index, ctx->client_id, ctx->worker_id, ctx->slot, ctx->req_bytes, 0,
                   ctx->start_time, g_event_queue->get_current_time(), (uint64_t)(g_event_queue->get_current_time() - ctx->start_time),
                   ctx->is_handshake ? std::string("c2s_handshake") : std::string("c2s_get")};
    g_flow_records.push_back(rec);
    // Schedule worker receive after optional server overhead
    EventTime when = g_event_queue->get_current_time();
    g_event_queue->schedule_completion(when, (void (*)(void*)) &worker_recv, ctx);
}

static void worker_recv(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log server recv
    if (!ctx->is_handshake) {
        g_server_log << "event=reqq_recv ts_ns=" << g_event_queue->get_current_time() << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot
                     << " size=" << ctx->req_bytes << " src=client:" << ctx->client_id << " dst=worker:" << ctx->worker_id << "\n";
    } else {
        g_server_log << "event=hand_recv ts_ns=" << g_event_queue->get_current_time() << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot
                     << " size=" << ctx->req_bytes << " src=client:" << ctx->client_id << " dst=worker:" << ctx->worker_id << "\n";
    }
    EventTime when = g_event_queue->get_current_time() + (ctx->is_handshake ? 0 : SERVER_OVERHEAD_NS);
    g_event_queue->schedule_completion(when, (void (*)(void*)) &worker_send, ctx);
}

static void worker_send(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log send
    if (ctx->is_handshake) {
        g_server_log << "event=hand_conf ts_ns=" << g_event_queue->get_current_time() << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot
                     << " size=" << ctx->resp_bytes << " src=worker:" << ctx->worker_id << " dst=client:" << ctx->client_id << "\n";
    } else {
        g_server_log << "event=resp_send ts_ns=" << g_event_queue->get_current_time() << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot
                     << " size=" << ctx->resp_bytes << " src=worker:" << ctx->worker_id << " dst=client:" << ctx->client_id << "\n";
    }
    ctx->server_send_time = g_event_queue->get_current_time();
    auto resp_chunk = std::make_unique<Chunk>(ctx->op_index, ctx->resp_bytes, ctx->route_rev, (void (*)(void*)) &on_response_arrival, ctx);
    g_topology->send(std::move(resp_chunk));
}

static void on_response_arrival(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    EventTime when = g_event_queue->get_current_time();
    if (ctx->is_handshake) g_event_queue->schedule_completion(when, (void (*)(void*)) &client_recv_rdma_finalize, ctx);
    else g_event_queue->schedule_completion(when, (void (*)(void*)) &client_recv_ud, ctx);
}

static void client_recv_ud(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    g_client_logs[ctx->client_id] << "event=resp_recv_ud ts_ns=" << g_event_queue->get_current_time() << " id=" << ctx->op_index
                                  << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot
                                  << " size=" << ctx->resp_bytes << " src=worker:" << ctx->worker_id << " dst=client:" << ctx->client_id << "\n";
    FlowRecord rec{ctx->op_index, ctx->client_id, ctx->worker_id, ctx->slot, 0, ctx->resp_bytes, ctx->server_send_time,
                   g_event_queue->get_current_time(), (uint64_t)(g_event_queue->get_current_time() - ctx->server_send_time), "s2c_ud"};
    g_flow_records.push_back(rec);
    // Send handshake
    ctx->is_handshake = true; ctx->req_bytes = HANDSHAKE_BYTES; ctx->start_time = g_event_queue->get_current_time() + HANDSHAKE_DELAY_NS;
    // Schedule the actual handshake send at start_time to avoid negative FCT
    g_event_queue->schedule_completion(ctx->start_time, (void (*)(void*)) &client_send_handshake, ctx);
}

static void client_send_handshake(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    ctx->handshake_send_time = g_event_queue->get_current_time();
    // Log client handshake send
    g_client_logs[ctx->client_id] << "event=hand_send ts_ns=" << ctx->handshake_send_time << " id=" << ctx->op_index
                                  << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot
                                  << " size=" << ctx->req_bytes << " src=client:" << ctx->client_id << " dst=worker:" << ctx->worker_id << "\n";
    // Send handshake to server; server path continues in on_request_arrival
    auto req_chunk = std::make_unique<Chunk>(ctx->op_index, ctx->req_bytes, ctx->route_fwd, (void (*)(void*)) &on_request_arrival, ctx);
    g_topology->send(std::move(req_chunk));
}

static void client_recv_rdma_finalize(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log and finalize RDMA stage
    g_client_logs[ctx->client_id] << "event=resp_rdma_read ts_ns=" << g_event_queue->get_current_time() << " id=" << ctx->op_index
                                  << " start_ns=" << ctx->handshake_send_time << " dur_ns=" << (g_event_queue->get_current_time() - ctx->handshake_send_time)
                                  << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot
                                  << " size=" << ctx->resp_bytes << " src=worker:" << ctx->worker_id << " dst=client:" << ctx->client_id << "\n";
    FlowRecord rec{ctx->op_index, ctx->client_id, ctx->worker_id, ctx->slot, 0, ctx->resp_bytes, ctx->server_send_time,
                   g_event_queue->get_current_time(), (uint64_t)(g_event_queue->get_current_time() - ctx->server_send_time), "s2c_rdma"};
    g_flow_records.push_back(rec);
    // Refill window if under limit
    g_inflight_per_client[ctx->client_id]--;
    if (g_total_sent_per_client[ctx->client_id] < g_client_limit[ctx->client_id]) {
        int* arg_c = (int*)malloc(sizeof(int)); *arg_c = ctx->client_id;
        g_event_queue->schedule_completion(g_event_queue->get_current_time(), (void (*)(void*)) &add_flow_for_client, arg_c);
    }
    delete ctx;
}

static void client_start_batch(void* arg) {
    int client_id = *(int*)arg; free(arg);
    EventTime send_time = g_event_queue->get_current_time();
    for (int w = 0; w < WINDOW_SIZE && g_total_sent_per_client[client_id] < g_client_limit[client_id]; w++) {
        int* cid = (int*)malloc(sizeof(int)); *cid = client_id;
        g_event_queue->schedule_completion(send_time, (void (*)(void*)) &add_flow_for_client, cid);
        send_time += (w == 0 ? STARTUP_DELAY_NS : SEND_SPACING_NS);
    }
}

static void add_flow_for_client(void* client_id_ptr) {
    int client_id = *(int*)client_id_ptr; free(client_id_ptr);
    int op_index = (int)g_next_op_index; g_next_op_index++;
    // HERD-style randomized worker and key selection
    int wn = (int)(hrd_fastrand(&g_clients[client_id].seed) % (uint32_t)NUM_WORKERS);
    int is_update = 0;
    int key_i = (int)(hrd_fastrand(&g_clients[client_id].seed) % (uint32_t)HERD_NUM_KEYS);
    uint32_t key_idx = g_clients[client_id].key_perm ? (uint32_t)g_clients[client_id].key_perm[key_i] : (uint32_t)key_i;
    uint128 key128 = CityHash128((char*)&key_idx, 4);
    uint8_t vlen = herd_val_len_from_key_parts(key128.first, key128.second);
    auto* ctx = new FlowCtx();
    ctx->op_index = op_index; ctx->client_id = client_id; ctx->worker_id = wn;
    ctx->slot = g_clients[client_id].ws[wn]; ctx->is_update = (is_update != 0);
    ctx->vlen = vlen; ctx->req_bytes = 17; ctx->is_handshake = false;
    ctx->route_fwd = g_routes_c2s[client_id]; ctx->route_rev = g_routes_s2c[client_id];
    ctx->start_time = g_event_queue->get_current_time(); ctx->handshake_send_time = 0;
    g_clients[client_id].ws[wn] = (g_clients[client_id].ws[wn] + 1) % WINDOW_SIZE;
    g_client_logs[client_id] << "event=req_send ts_ns=" << g_event_queue->get_current_time() << " id=" << ctx->op_index
                             << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot
                             << " size=" << ctx->req_bytes << " src=client:" << ctx->client_id << " dst=worker:" << ctx->worker_id << "\n";
    auto req_chunk = std::make_unique<Chunk>(ctx->op_index, ctx->req_bytes, ctx->route_fwd, (void (*)(void*)) &on_request_arrival, ctx);
    g_topology->send(std::move(req_chunk));
    g_total_sent_per_client[client_id]++;
    g_inflight_per_client[client_id]++;
}

// flowsim parameters
std::vector<int64_t> fat;
std::vector<int64_t> fsize;
std::vector<int64_t> fct_i;
std::vector<double> params;
std::unordered_map<int, int64_t> fct_map;
//std::vector<int64_t> fid;
//torch::Tensor fid_tensor;
uint64_t limit;

uint32_t num_tors = 70;
uint32_t num_per_tor = 16;

std::vector<int> host_ids;
uint32_t flow_limit;
std::unordered_map<uint32_t, uint32_t> flow_counts;
std::queue<uint32_t> flow_queue;
std::unordered_map<uint32_t, std::queue<uint32_t>> tor_queue;

int32_t n_flows;

std::vector<int32_t> flowid_to_linkid_flat;
std::vector<int32_t> flowid_to_linkid_offsets;
std::vector<int32_t> edges_flow_ids;
std::vector<int32_t> edges_link_ids;

std::vector<float> res_fct;
std::vector<float> res_sldn;

// m4 options
auto options_int64 = torch::TensorOptions().dtype(torch::kInt64);


// m4 model
int gpu_id = 0;
torch::Device device(torch::kCUDA, gpu_id);

static torch::jit::script::Module lstmcell_time;
static torch::jit::script::Module lstmcell_rate;
static torch::jit::script::Module lstmcell_time_link;
static torch::jit::script::Module lstmcell_rate_link;
static torch::jit::script::Module output_layer;
static torch::jit::script::Module gnn_layer_0;
static torch::jit::script::Module gnn_layer_1;
static torch::jit::script::Module gnn_layer_2;

// m4 tensors
torch::Tensor size_tensor;
torch::Tensor fat_tensor;
torch::Tensor i_fct_tensor;
torch::Tensor params_tensor;

torch::Tensor release_time_tensor;
bool queued;

torch::Tensor flowid_to_linkid_flat_tensor;
torch::Tensor flowid_to_linkid_offsets_tensor;
torch::Tensor flowid_to_nlinks_tensor;

torch::Tensor edges_flow_ids_tensor;
torch::Tensor edges_link_ids_tensor;

torch::Tensor edge_index;

torch::Tensor h_vec;
torch::Tensor z_t_link;

torch::Tensor link_to_graph_id;
torch::Tensor link_to_nflows;
torch::Tensor flow_to_graph_id;

torch::Tensor time_last;
torch::Tensor flowid_active_mask;

torch::Tensor res_fct_tensor;
torch::Tensor res_sldn_tensor;

torch::Tensor sldn_est;

static torch::Tensor ones_cache;

int graph_id_counter;
int graph_id_cur;

int flow_id_in_prop;
int current_flow;
int n_flows_active;
int n_flows_arrived;
int n_flows_completed;
float time_clock;
int completed_flow_id;
int min_idx;

float flow_arrival_time;
float flow_completion_time;

int get_tor(int flow_id) {
    return host_ids.at(flow_id) / num_per_tor;
}

void setup_m4(torch::Device device) {
    if (!torch::cuda::is_available()) {
        std::cerr << "[ERROR] CUDA is not available!" << std::endl;
        return;
    }

    // Disable gradient calculations
    torch::NoGradGuard no_grad;

    // Load models
    static bool models_loaded = false;
    if (!models_loaded) {
        const std::string model_dir = "models_old";
        try {
            lstmcell_time = torch::jit::load(model_dir + "/lstmcell_time.pt", device);
            lstmcell_rate = torch::jit::load(model_dir + "/lstmcell_rate.pt", device);
            lstmcell_rate_link = torch::jit::load(model_dir + "/lstmcell_rate_link.pt", device);
            lstmcell_time_link = torch::jit::load(model_dir + "/lstmcell_time_link.pt", device);
            output_layer = torch::jit::load(model_dir + "/output_layer.pt", device);
            gnn_layer_0 = torch::jit::load(model_dir + "/gnn_layer_0.pt", device);
            gnn_layer_1 = torch::jit::load(model_dir + "/gnn_layer_1.pt", device);
            gnn_layer_2 = torch::jit::load(model_dir + "/gnn_layer_2.pt", device);
        }
        catch (const c10::Error& e) {
            std::cerr << "[ERROR] Failed to load one or more models: " << e.what() << std::endl;
            return;
        }

        // Set models to evaluation mode
        lstmcell_time.eval();
        lstmcell_rate.eval();
        lstmcell_rate_link.eval();
        lstmcell_time_link.eval();
        output_layer.eval();
        gnn_layer_0.eval();
        gnn_layer_1.eval();
        gnn_layer_2.eval();

        // Optimize models for inference
        lstmcell_time = torch::jit::optimize_for_inference(lstmcell_time);
        lstmcell_rate = torch::jit::optimize_for_inference(lstmcell_rate);
        lstmcell_time_link = torch::jit::optimize_for_inference(lstmcell_time_link);
        lstmcell_rate_link = torch::jit::optimize_for_inference(lstmcell_rate_link);
        output_layer = torch::jit::optimize_for_inference(output_layer);
        gnn_layer_0 = torch::jit::optimize_for_inference(gnn_layer_0);
        gnn_layer_1 = torch::jit::optimize_for_inference(gnn_layer_1);
        gnn_layer_2 = torch::jit::optimize_for_inference(gnn_layer_2);
    }
}

void setup_m4_tensors(torch::Device device, int32_t n_edges, int32_t n_links, int32_t h_vec_dim) {
    // Define tensor options
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool);
    auto options_double = torch::TensorOptions().dtype(torch::kFloat64);

    // Clone tensors to ensure ownership
    size_tensor = torch::from_blob(fsize.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    size_tensor = torch::log2(size_tensor /1000.0f + 1.0f);

    fat_tensor = torch::from_blob(fat.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    i_fct_tensor = torch::from_blob(fct_i.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    params_tensor = torch::from_blob(params.data(), {13}, options_double).to(torch::kFloat32).to(device);

    //fid_tensor = torch::from_blob(fid.data(), {n_flows}, options_int64).to(device);

    // Convert flowid_to_linkid to tensors
    flowid_to_linkid_flat_tensor = torch::from_blob(flowid_to_linkid_flat.data(), {n_edges}, options_int32).to(device);
    flowid_to_linkid_offsets_tensor = torch::from_blob(flowid_to_linkid_offsets.data(), {n_flows + 1}, options_int32).to(device);
    flowid_to_nlinks_tensor = flowid_to_linkid_offsets_tensor.slice(0, 1, n_flows+1) - flowid_to_linkid_offsets_tensor.slice(0, 0, n_flows);
    
    // Convert edges_flow_ids and edges_link_ids to tensors
    edges_flow_ids_tensor = torch::from_blob(edges_flow_ids.data(), {n_edges}, options_int32).to(device);
    edges_link_ids_tensor = torch::from_blob(edges_link_ids.data(), {n_edges}, options_int32).to(device);

    // Construct edge_index tensor [2, 2 * n_edges] for bidirectional edges
    edge_index = torch::stack({edges_flow_ids_tensor, edges_link_ids_tensor}, 0); // [2, n_edges]

    // Initialize tensors for active flows
    h_vec = torch::zeros({n_flows, h_vec_dim}, options_float).to(device);
    h_vec.index_put_({torch::arange(n_flows, device=device), 0}, 1.0f);
    h_vec.index_put_({torch::arange(n_flows, device=device), 2}, size_tensor);
    h_vec.index_put_({torch::arange(n_flows, device=device), 3}, flowid_to_nlinks_tensor.to(options_float));

    // Initialize z_t_link as in Python
    z_t_link = torch::zeros({n_links, h_vec_dim}, options_float).to(device); // [n_links, h_vec_dim]
    z_t_link.index_put_({torch::arange(n_links, device=device), 1}, 1.0f);
    z_t_link.index_put_({torch::arange(n_links, device=device), 2}, 1.0f);

    // Initialize graph management tensors
    link_to_graph_id = -torch::ones({n_links}, options_int32).to(device);
    link_to_nflows = torch::zeros({n_links}, options_int32).to(device);
    flow_to_graph_id = -torch::ones({n_flows}, options_int32).to(device);

    graph_id_counter = 0;
    graph_id_cur = 0;

    // Initialize time_last and flowid_active_mask
    time_last = torch::zeros({n_flows}, options_float).to(device);
    flowid_active_mask = torch::zeros({n_flows}, options_bool).to(device);

    release_time_tensor = torch::zeros({n_flows}, options_float).to(device);

    // Initialize result tensors
    res_fct_tensor = torch::zeros({n_flows, 2}, options_float).to(device);
    res_sldn_tensor = torch::zeros({n_flows, 2}, options_float).to(device);

    // Initialize counters
    flow_id_in_prop = 0;
    current_flow = 0;
    n_flows_active = 0;
    n_flows_arrived = 0;
    n_flows_completed = 0;
    time_clock = 0.0f;
    completed_flow_id = -1; // Initialize with invalid ID
    min_idx = -1;

    ones_cache = torch::ones({n_links}, options_int32).to(device);
}

void update_times_m4() {
    torch::NoGradGuard no_grad;
    // Determine next flow arrival and completion times
    if (flow_limit == 0) {
        if (current_flow < n_flows) {
            flow_arrival_time = fat_tensor[current_flow].item<float>();
            flow_id_in_prop = current_flow;
        } else {
            flow_arrival_time = std::numeric_limits<float>::infinity();
            flow_id_in_prop = -1;
        }
    } else {
        flow_id_in_prop = -1;
        flow_arrival_time = std::numeric_limits<float>::infinity();
        int queue_size = 0;
        for (int i = 0; i < num_tors; i++) {
            queue_size += tor_queue[i].size();
        }
        std::cout << "waiting in queue " << queue_size << " " << flow_queue.size() << "\n";
        if (!flow_queue.empty()) {
            std::cout << "flow queue " << flow_queue.front() << "\n";
            flow_id_in_prop = flow_queue.front();
            flow_arrival_time = fat_tensor[flow_id_in_prop].item<float>() < time_clock ? time_clock : fat_tensor[flow_id_in_prop].item<float>();
            queued = true;
        }
        else {
            std::cout << "checking flow\n";
            while (current_flow < n_flows) {
                int tor = get_tor(current_flow);
                if (flow_counts[tor] < flow_limit) {
                    std::cout << "taking flow " << current_flow << "\n";
                    flow_id_in_prop = current_flow;
                    flow_arrival_time = fat_tensor[current_flow].item<float>();
                    break;
                } else {
                    std::cout << "pushing flow " << current_flow << " " << " " << host_ids.at(current_flow) << " " << get_tor(current_flow) << " " << tor_queue[get_tor(current_flow)].size() << "\n";
                    tor_queue[get_tor(current_flow)].push(current_flow);
                    current_flow++;
                }
            }
            queued = false;
        }
    }
    flow_completion_time = std::numeric_limits<float>::infinity();

    if (n_flows_active > 0) {
        // Get indices of active flows
        auto flowid_active_indices = torch::nonzero(flowid_active_mask).flatten();
        auto h_vec_active = h_vec.index_select(0, flowid_active_indices);
        auto nlinks_cur = flowid_to_nlinks_tensor.index_select(0, flowid_active_indices).unsqueeze(1); // [n_active,1]
        auto params_data_cur = params_tensor.repeat({n_flows_active, 1});
        std::cout << nlinks_cur.size(0) << " " << params_data_cur.size(0) << " " << h_vec_active.size(0) << "\n";
        auto input_tensor = torch::cat({nlinks_cur, params_data_cur, h_vec_active}, 1);

        // Perform inference
        sldn_est = output_layer.forward({ input_tensor }).toTensor().view(-1);; // [n_active]
        sldn_est = torch::clamp(sldn_est, 1.0f, std::numeric_limits<float>::infinity());

        auto fct_stamp_est = release_time_tensor.index_select(0, flowid_active_indices) + sldn_est * i_fct_tensor.index_select(0, flowid_active_indices);

        // Find the flow with the minimum estimated completion time
        min_idx = torch::argmin(fct_stamp_est).item<int>();
        flow_completion_time = fct_stamp_est[min_idx].item<float>();
        completed_flow_id = flowid_active_indices[min_idx].item<int>();
    }
}


void step_m4() {
    torch::NoGradGuard no_grad;

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    
    // Decide whether the next event is a flow arrival or completion
    if (flow_arrival_time < flow_completion_time) {
        // New flow arrives before the next completion

        std::cout << flow_id_in_prop << " arrived\n";

        if (queued) {
            flow_queue.pop();
        } else {
            current_flow++;
        }

        time_clock = flow_arrival_time;

        flowid_active_mask[flow_id_in_prop] = true;
        
        time_last[flow_id_in_prop] = time_clock;
        release_time_tensor.index_put_({flow_id_in_prop}, flow_arrival_time);
        n_flows_arrived++;

        // Assign graph IDs
        int start_idx = flowid_to_linkid_offsets[flow_id_in_prop];
        int end_idx = flowid_to_linkid_offsets[flow_id_in_prop + 1];
        auto links_tensor = flowid_to_linkid_flat_tensor.slice(0, start_idx, end_idx);

        link_to_nflows.index_add_(0, links_tensor, ones_cache.slice(0, 0, links_tensor.size(0)));

        // Extract graph IDs for valid links
        auto graph_ids_tensor = link_to_graph_id.index({links_tensor});
        auto graph_mask = graph_ids_tensor != -1;
        auto valid_graph_ids_tensor = graph_ids_tensor.masked_select(graph_mask);

        // Convert unique graph IDs to a CPU vector for iteration
        auto unique_graph_ids_tensor = std::get<0>(torch::_unique(valid_graph_ids_tensor, false, false));
        int64_t num_unique_ids = unique_graph_ids_tensor.size(0);

        // Define `graph_id_cur` to use for assigning IDs
        if (num_unique_ids == 0) {
            // Case: No existing graph ID, assign a new one
            graph_id_cur = graph_id_counter;
            flow_to_graph_id.index_put_({flow_id_in_prop}, graph_id_cur);
            link_to_graph_id.index_put_({links_tensor}, graph_id_cur);
            graph_id_counter += 1;
        } else if (num_unique_ids == 1) {
            // Case: One unique graph ID exists, reuse it
            graph_id_cur = unique_graph_ids_tensor.item<int64_t>();
            flow_to_graph_id.index_put_({flow_id_in_prop}, graph_id_cur);
            link_to_graph_id.index_put_({links_tensor}, graph_id_cur);
        } else {
            // Case: Multiple graph IDs need to be merged into a new one
            graph_id_cur = graph_id_counter;

            // Update all flows and links with old graph IDs to the new ID
            auto old_graph_ids = unique_graph_ids_tensor;

            // Create masks for flows and links with old graph IDs
            auto flows_with_old_ids_mask = torch::isin(flow_to_graph_id, old_graph_ids);
            auto links_with_old_ids_mask = torch::isin(link_to_graph_id, old_graph_ids);

            // Update graph IDs in a single operation
            flow_to_graph_id.masked_fill_(flows_with_old_ids_mask, graph_id_cur);
            link_to_graph_id.masked_fill_(links_with_old_ids_mask, graph_id_cur);

            // Assign the new graph ID to the current flow and its links
            flow_to_graph_id.index_put_({flow_id_in_prop}, graph_id_cur);
            link_to_graph_id.index_put_({links_tensor}, graph_id_cur);

            // Increment the graph ID counter for the next assignment
            graph_id_counter += 1;
        }
        flow_counts[get_tor(flow_id_in_prop)] += 1;
        n_flows_active += 1;
    }
    else {
        // Flow completes before the next arrival
        time_clock = flow_completion_time;
        // Actual FCT and SLDN
        res_fct_tensor[completed_flow_id][0] = flow_completion_time - release_time_tensor[completed_flow_id].item<float>();
        res_sldn_tensor[completed_flow_id][0] = sldn_est[min_idx];
        // Emit HERD-style staged messages for this completed flow (ML-driven timing)
        {
            uint64_t start_ns = (uint64_t)release_time_tensor[completed_flow_id].item<float>();
            uint64_t ml_fct_ns = (uint64_t)res_fct_tensor[completed_flow_id][0].item<float>();
            uint64_t large_size = (uint64_t)fsize[completed_flow_id];
            generate_herd_messages_m4(completed_flow_id, start_ns, ml_fct_ns, large_size);
        }
        // Update active flow mask to mark the flow as completed
        flowid_active_mask[completed_flow_id] = false;

        // Decrement the count of active flows and increment completed flows
        n_flows_active--;
        n_flows_completed++;
        flow_counts[get_tor(completed_flow_id)] -= 1;
        // Refill window by scheduling the next flow's arrival at current time
        if (g_next_to_schedule < g_ops_limit) {
            fat_tensor.index_put_({g_next_to_schedule}, time_clock);
            g_next_to_schedule++;
        } else if (!tor_queue[get_tor(completed_flow_id)].empty()) {
            std::cout << "tor push " << completed_flow_id << " " << get_tor(completed_flow_id) << " " << tor_queue[get_tor(completed_flow_id)].front() << "\n";
            flow_queue.push(tor_queue[get_tor(completed_flow_id)].front());
            tor_queue[get_tor(completed_flow_id)].pop();
        }
        std::cout << "m4: flow completed " << completed_flow_id <<  " " << get_tor(completed_flow_id) << "\n";

        // Get graph ID of the completed flow
        graph_id_cur = flow_to_graph_id[completed_flow_id].item<int64_t>();
        // Get links for this flow
        int start_idx = flowid_to_linkid_offsets[completed_flow_id];
        int end_idx = flowid_to_linkid_offsets[completed_flow_id + 1];
        auto links_tensor = flowid_to_linkid_flat_tensor.slice(0, start_idx, end_idx);

        link_to_nflows.index_add_(0, links_tensor, -ones_cache.slice(0, 0, links_tensor.size(0)));
        flow_to_graph_id.index_put_({completed_flow_id}, -1);

        // Find links with no active flows using tensor operations
        auto no_flow_mask = (link_to_nflows.index({links_tensor}) == 0);
        auto no_flow_links_tensor = links_tensor.masked_select(no_flow_mask);

        // Update link_to_graph_id and reset z_t_link for links with no active flows in bulk

        // Assign -1 to 'link_to_graph_id' for all 'no_flow_links' at once
        link_to_graph_id.index_put_({no_flow_links_tensor}, -1);

        // // Create tensors with desired reset values for 'z_t_link'
        auto reset_values = torch::zeros({no_flow_links_tensor.size(0), z_t_link.size(1)}, options_float).to(device);
        auto ones = torch::ones({no_flow_links_tensor.size(0)}, options_float).to(device);

        auto slice = z_t_link.index({no_flow_links_tensor, torch::indexing::Slice()});
        z_t_link.index_put_({no_flow_links_tensor, torch::indexing::Slice()}, reset_values);
        z_t_link.index_put_({no_flow_links_tensor, 1}, ones);
        z_t_link.index_put_({no_flow_links_tensor, 2}, ones);
    }
    // Update h_vec for active flows
    auto flowid_active_mask_cur = torch::logical_and(flowid_active_mask, flow_to_graph_id == graph_id_cur);
    auto flowid_active_list_cur = torch::nonzero(flowid_active_mask_cur).flatten();
    std::cout << "actual var: " << n_flows_active << ", n_active_flows: "<<flowid_active_list_cur.numel()<< ", graph_id_cur: " << graph_id_cur<< ", fat: " << flow_arrival_time/1000.0 << ", fct: " << flow_completion_time/1000.0 << std::endl;
    std::cout <<"m4: " << flow_to_graph_id[0].item<int32_t>() << "\n";
    if (flowid_active_list_cur.numel() > 0 && flow_arrival_time < flow_completion_time) {
        
        // Calculate time deltas for active flows
        auto time_deltas = (time_clock - time_last.index_select(0, flowid_active_list_cur).squeeze()).view({-1, 1});

        // Create a mask for the edges corresponding to the active flows
        auto edge_mask = torch::isin(edge_index[0], flowid_active_list_cur);
        auto selected_indices = edge_mask.nonzero().squeeze();
        auto edge_index_cur = edge_index.index_select(1, selected_indices);

        // Determine the number of active flows
        auto n_flows_active_cur = flowid_active_list_cur.size(0);
        auto new_flow_indices=torch::searchsorted(flowid_active_list_cur,edge_index_cur[0]);

        // Extract return_inverse from the tuple (index 1 of the tuple)
        auto unique_result_tuple = torch::_unique(edge_index_cur[1], true, true);
        auto active_link_idx = std::get<0>(unique_result_tuple); // Unique link IDs
        auto new_link_indices = std::get<1>(unique_result_tuple); // Inverse indices for reindexing

        new_link_indices += n_flows_active_cur;
        auto edges_list_active=torch::cat({ torch::stack({new_flow_indices, new_link_indices}, 0), torch::stack({new_link_indices, new_flow_indices}, 0)}, 1);

        // Check if any time delta is greater than zero
        auto h_vec_time_updated = h_vec.index_select(0, flowid_active_list_cur);
        auto h_vec_time_link_updated = z_t_link.index_select(0, active_link_idx);
        auto max_time_delta = torch::max(time_deltas).item<float>();
        if (max_time_delta>0.0f) {
            // Update time using lstmcell_time
            time_deltas.fill_(max_time_delta/1000.0f);
            h_vec_time_updated = lstmcell_time.forward({ time_deltas, h_vec_time_updated}).toTensor();

            auto time_deltas_link = torch::zeros({active_link_idx.size(0), 1}, options_float).to(device);
            time_deltas_link.fill_(max_time_delta / 1000.0f);
            h_vec_time_link_updated = lstmcell_time_link.forward({ time_deltas_link, h_vec_time_link_updated }).toTensor();
        }

        // Forward pass through the GNN layers
        auto z_t_link_cur=z_t_link.index_select(0,active_link_idx);
        auto x_combined=torch::cat({h_vec_time_updated, h_vec_time_link_updated}, 0);

        auto gnn_output_0 = gnn_layer_0.forward({x_combined, edges_list_active}).toTensor();
        auto gnn_output_1 = gnn_layer_1.forward({gnn_output_0, edges_list_active}).toTensor();
        auto gnn_output_2 = gnn_layer_2.forward({gnn_output_1, edges_list_active}).toTensor();

        // Update rate using lstmcell_rate
        auto h_vec_rate_updated = gnn_output_2.slice(0,0,n_flows_active_cur);
        auto h_vec_rate_link = gnn_output_2.slice(0, n_flows_active_cur, gnn_output_2.size(0));

        auto params_data = params_tensor.repeat({n_flows_active_cur, 1});
        h_vec_rate_updated = torch::cat({h_vec_rate_updated, params_data}, 1);

        h_vec_rate_updated = lstmcell_rate.forward({ h_vec_rate_updated, h_vec_time_updated }).toTensor();
        h_vec_rate_link = lstmcell_rate_link.forward({ h_vec_rate_link, h_vec_time_link_updated }).toTensor();

        // Update h_vec with the new hidden states
        h_vec.index_copy_(0, flowid_active_list_cur, h_vec_rate_updated);

        //auto z_t_link_updated = h_vec_rate_link.slice(0, n_flows_active_cur, n_flows_active_cur + active_link_idx.size(0));
        new_link_indices -= n_flows_active_cur;
        auto long_indices = active_link_idx.to(torch::kInt64);
        z_t_link.index_copy_(0, long_indices, h_vec_rate_link);

        // Update time_last to the current time for active flows
        time_last.index_put_({flowid_active_list_cur}, time_clock);
    }
}


int main(int argc, char *argv[]) {
    // Mirror flowsim: run HERD-like mode when not enough args; else run ML+HERD
    if (argc < 6) {
        g_event_queue = std::make_shared<EventQueue>();
        Topology::set_event_queue(g_event_queue);

        // Single client (0) <-> server (1) link
        g_topology = std::make_shared<Topology>(2, 2);
        // Match flowsim: bandwidth ~ 10 GB/s / 8 = 1.25 B/ns, latency 3500 ns
        g_topology->connect(0, 1, 10.0 / 8.0, 3500.0f, true);

        // Build routes 0 -> 1 -> 0
        Route c2s; c2s.push_back(g_topology->get_device(0)); c2s.push_back(g_topology->get_device(1));
        Route s2c; s2c.push_back(g_topology->get_device(1)); s2c.push_back(g_topology->get_device(0));

        NUM_CLIENTS = 1;
        g_clients.assign(NUM_CLIENTS, ClientState());
        for (int i = 0; i < NUM_CLIENTS; i++) {
            g_clients[i].id = i;
            g_clients[i].key_perm = herd_get_random_permutation(HERD_NUM_KEYS, i, &g_clients[i].seed);
        }
        g_routes_c2s = {c2s};
        g_routes_s2c = {s2c};
        g_client_logs.resize(NUM_CLIENTS); g_client_logs[0].open("client_0.log");
        g_server_log.open("server.log");

        g_total_sent_per_client.assign(NUM_CLIENTS, 0);
        g_client_limit.assign(NUM_CLIENTS, 650);
        g_inflight_per_client.assign(NUM_CLIENTS, 0);

        // Schedule initial window
        EventTime t0 = g_event_queue->get_current_time() + 1;
        for (int cid = 0; cid < NUM_CLIENTS; cid++) {
            int* arg = (int*)malloc(sizeof(int)); *arg = cid;
            g_event_queue->schedule_completion(t0, (void (*)(void*)) &client_start_batch, arg);
        }
        while (!g_event_queue->finished()) g_event_queue->proceed();

        // Write flows.txt compatible with flowsim
        std::ofstream ofs("flows.txt");
        for (const auto& r : g_flow_records) {
            ofs << r.op_index << " " << r.client_id << " " << r.worker_id << " " << r.slot << " "
                << r.req_bytes << " " << r.resp_bytes << " " << r.start_ns << " " << r.end_ns << " "
                << r.fct_ns << " " << r.stage << "\n";
        }
        if (g_client_logs[0].is_open()) g_client_logs[0].close();
        if (g_server_log.is_open()) g_server_log.close();
        std::cout << "HERD-sim completed ops: " << g_flow_records.size() << ", wrote flows.txt\n";
        return 0;
    }
    
    // =============== Original ML inference path ===============
    const std::string scenario_path = argv[1];
    const std::string fat_path = scenario_path + "/fat.npy";
    const std::string fsize_path = scenario_path + "/fsize.npy";
    const std::string topo_path = scenario_path + "/topology.txt";
    const std::string routing_path = scenario_path + "/flow_to_path.txt";
    const std::string fct_i_path = scenario_path + "/fct_i_topology_flows.npy";
    const std::string flow_link_path = scenario_path + "/flow_to_links.txt";
    //const std::string fid_path = scenario_path + "/fid_topology_flows.npy";
    const std::string config_path = argv[2];
    const std::string param_path = scenario_path + "/param_topology_flows.npy";
    const std::string write_path = argv[3];
    flow_limit = std::stoi(argv[4]);
    std::string release_path;
    if (argc == 6) {
        release_path = argv[5];
    }

    for (uint32_t i = 0; i < num_tors; i++) {
        flow_counts[i] = 0;
    }

    std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();
 
    npy::npy_data d_fat = npy::read_npy<int64_t>(fat_path);
    std::vector<int64_t> arrival_times = d_fat.data;

    npy::npy_data d_fsize = npy::read_npy<int64_t>(fsize_path);
    std::vector<int64_t> flow_sizes = d_fsize.data;

    //npy::npy_data d_fid = npy::read_npy<int64_t>(fid_path);
    //fid = d_fid.data;

    limit = arrival_times.size();
    n_flows = arrival_times.size();

    for (int i = 0; i < arrival_times.size() & i < limit; i++) {
        int64_t flow_size = flow_sizes.at(i);
        fat.push_back(arrival_times.at(i));
        fsize.push_back(flow_size);
    }

    const double BYTES_PER_HEADER = 48;
    const double MTU = 1000;
    std::shared_ptr<Topology> topology = construct_fat_tree_topology(topo_path);

    float latency = topology->get_latency();
    float bandwidth = topology->get_bandwidth();

    std::vector<Route> routing;
    std::filesystem::path routing_fs = std::filesystem::current_path() / routing_path;
    std::ifstream infile_routing(routing_fs);
    int num_hops;
    // Also capture raw device IDs per flow to synthesize link mapping if needed
    std::vector<std::vector<int>> route_device_ids;
    while (infile_routing >> num_hops) {
        int host_id;
        auto route = Route();
        std::vector<int> flow_devs;
        infile_routing >> host_id;
        host_ids.push_back(host_id);
        route.push_back(topology->get_device(host_id));
        flow_devs.push_back(host_id);
        for (int i = 1; i < num_hops; i++) {
            infile_routing >> host_id;
            route.push_back(topology->get_device(host_id));
            flow_devs.push_back(host_id);
        }
        routing.push_back(route);
        route_device_ids.push_back(std::move(flow_devs));
    }

    // Ideal FCT: fixed 3000ns base + transmission delay only (no propagation/first-packet adders)
    const double FIXED_IDEAL_NS = 3000.0;
    for (int i = 0; i < (int)fat.size(); i++) {
        double trans_delay = (((fsize.at(i) + std::ceil((double)fsize.at(i) / MTU) * BYTES_PER_HEADER)) / bandwidth);
        fct_i.push_back(FIXED_IDEAL_NS + trans_delay);
    }

    // Load params; if missing, use zeros for parity with flowsim
    try {
    npy::npy_data d_param = npy::read_npy<double>(param_path);
    params = d_param.data;
    } catch (...) {
        params.assign(13, 0.0);
    }

    std::filesystem::path cwd = std::filesystem::current_path() / flow_link_path;
    std::ifstream infile(cwd);
    int32_t offset = 0;
    int32_t flow_id = 0;
    if (infile.good()) {
        int num_links;
    while (infile >> num_links) {
        flowid_to_linkid_offsets.push_back(offset);
        for (int i = 0; i < num_links; i++) {
            int32_t link;
            infile >> link;
            flowid_to_linkid_flat.push_back(link);
            offset++;
            edges_flow_ids.push_back(flow_id);
            edges_link_ids.push_back(link);
        }
        flow_id++;
    }
    flowid_to_linkid_offsets.push_back(offset);
    } else {
        // Fallback: synthesize link mapping from route_device_ids
        // Assign a unique incremental link id to each directed hop (u->v)
        std::unordered_map<long long, int32_t> link_id_map; // key = ((long long)u<<32)|v
        auto key_of = [](int u, int v) -> long long { return ( (long long)u << 32 ) | (unsigned int)v; };
        for (size_t f = 0; f < route_device_ids.size(); ++f) {
            const auto &devs = route_device_ids[f];
            flowid_to_linkid_offsets.push_back(offset);
            for (size_t i = 0; i + 1 < devs.size(); ++i) {
                int u = devs[i], v = devs[i+1];
                long long key = key_of(u, v);
                auto it = link_id_map.find(key);
                int32_t lid;
                if (it == link_id_map.end()) {
                    lid = (int32_t)link_id_map.size();
                    link_id_map.emplace(key, lid);
                } else {
                    lid = it->second;
                }
                flowid_to_linkid_flat.push_back(lid);
                offset++;
                edges_flow_ids.push_back((int32_t)f);
                edges_link_ids.push_back(lid);
            }
        }
        flowid_to_linkid_offsets.push_back(offset);
    }

    uint32_t n_edges = (uint32_t)flowid_to_linkid_flat.size();

    infile.close();
    // Open config; if not provided or missing, fall back to test_config_testbed.yaml
    std::ifstream infile_cfg(config_path);
    if (!infile_cfg.good()) {
        std::string fallback_cfg = "config/test_config_testbed.yaml";
        infile_cfg.open(fallback_cfg);
    }
    std::ostringstream contents;
    contents << infile_cfg.rdbuf();
    std::string config_contents = contents.str();
    ryml::Tree config = ryml::parse_in_place(ryml::to_substr(config_contents));
    ryml::NodeRef hidden_size_node = config["model"]["hidden_size"];
    int32_t hidden_size;
    hidden_size_node >> hidden_size;
    ryml::NodeRef n_links_node = config["dataset"]["n_links_max"];
    int32_t n_links;
    n_links_node >> n_links;

    std::cout << "setting up m4\n";

    setup_m4(device);
    setup_m4_tensors(device, n_edges, n_links, hidden_size);

    // Integrate HERD app pipeline into ML path: sliding window and dynamic arrivals
    const int OPS_LIMIT = std::min((int)n_flows, 650);
    int next_to_schedule = 0;
    // Seed initial window
    float t0 = time_clock; // 0
    for (int w = 0; w < WINDOW_SIZE && next_to_schedule < OPS_LIMIT; ++w) {
        float dt = (w == 0 ? 0.0f : (float)SEND_SPACING_NS);
        fat_tensor.index_put_({next_to_schedule}, t0 + dt);
        next_to_schedule++;
    }

    int flow_index = 0;
    int flows_completed = 0;
    // Expose for step_m4() to refill dynamically on completion
    g_ops_limit = OPS_LIMIT;
    g_next_to_schedule = next_to_schedule;

    while (n_flows_arrived < OPS_LIMIT || n_flows_completed < OPS_LIMIT) {
        std::cout << "provoking " << n_flows_arrived << " " << n_flows_completed << "\n";
        update_times_m4();
        step_m4();
    }

    std::vector<float> fct_vector;
    fct_vector.reserve(res_fct_tensor.sizes()[0]);
    double sum_slowdown = 0.0;
    int count_ops = 0;
    for (int i = 0; i < res_fct_tensor.sizes()[0]; i++) {
        float fct = res_fct_tensor[i][0].item<float>();
        fct_vector.push_back(fct);
        float ideal = (i < (int)fct_i.size() ? (float)fct_i[i] : 1.0f);
        if (ideal > 0.0f) { sum_slowdown += (double)fct / (double)ideal; count_ops++; }
    }

    // Print concise slowdown summary
    if (count_ops > 0) {
        double mean_slowdown = sum_slowdown / (double)count_ops;
        std::cout << "M4 mean slowdown=" << mean_slowdown << " over " << count_ops << " ops\n";
    }

    npy::npy_data<float> d;
    d.data = fct_vector;
    d.shape = {limit};
    d.fortran_order = false;
    npy::write_npy(write_path, d);

    if (argc == 6) {
        std::vector<float> release_times;
        for (int i = 0; i < limit; i++) {
            release_times.push_back(release_time_tensor[i].item<float>());
        }

        npy::npy_data<float> d_release;
        d_release.data = release_times;
        d_release.shape = {limit};
        d_release.fortran_order = false;
        npy::write_npy(release_path, d_release);
    }

    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start).count() << " seconds\n";

    //torch::cuda::synchronize();
    //torch::cuda::emptyCache();
}

