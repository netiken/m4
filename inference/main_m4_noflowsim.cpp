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
#include <algorithm>

// CityHash for 128-bit key generation (exactly as in HERD)
#include "rdma_bench/mica/mica.h" // Brings in city.h and MICA_MAX_VALUE

// ======================== HERD-like simulation bits ========================
// Minimal constants mirroring herd/main.h
static constexpr int HERD_NUM_KEYS = (8 * 1024 * 1024);
static constexpr int NUM_WORKERS = 12;
static int NUM_CLIENTS = 1; // configurable per-topology mode
static int WINDOW_SIZE = 16; // per-worker ring slots per client

// New protocol constants for staged exchange
static constexpr uint64_t RESP_UD_BYTES = 41;       // Server's first small response
static constexpr uint64_t HANDSHAKE_BYTES = 10;     // Client's handshake payload size
static uint64_t RESP_RDMA_BYTES = 1024008; // Server's large variable response

// Callback when a request arrives at the server. Immediately send a response.
// Tunables for extra timing (ns)
// Disable explicit propagation in main; rely on Topology link latency instead
static constexpr uint64_t SERVER_OVERHEAD_NS = 10000; // flowsim control-path server delay
static constexpr uint64_t SEND_SPACING_NS = 2500;     // Inter-send spacing within a batch
static constexpr uint64_t STARTUP_DELAY_NS = 0;       // Extra delay between first and second initial sends
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
static std::vector<uint32_t> g_inflight_per_client; // per-client active flows (sliding window)
static uint64_t g_next_op_index = 0; // global unique op index
static std::vector<std::ofstream> g_client_logs; // one per client
static std::ofstream g_server_log;
// HERD record aggregation is now handled by g_flow_records
// Shared pointers to use inside callbacks
static std::shared_ptr<EventQueue> g_event_queue;
static std::shared_ptr<Topology> g_topology;

// ======================== ML Pipeline State Variables ========================
// All ML state tensors for LSTM+GNN network simulation (from inference_old)
torch::Device device = torch::kCPU;

// Model components
torch::jit::script::Module lstmcell_time, lstmcell_rate;
torch::jit::script::Module lstmcell_time_link, lstmcell_rate_link;
torch::jit::script::Module gnn_layer_0, gnn_layer_1, gnn_layer_2;
torch::jit::script::Module output_layer;

// Flow and topology data
std::vector<int> host_ids;
std::vector<double> fsize;
std::vector<double> fct_i;
std::vector<double> params;
torch::Tensor fat_tensor;
torch::Tensor size_tensor;
torch::Tensor params_tensor;

// Link mapping tensors
torch::Tensor flowid_to_linkid_flat_tensor;
torch::Tensor flowid_to_linkid_offsets_tensor;
torch::Tensor flowid_to_nlinks_tensor;
torch::Tensor edges_flow_ids_tensor;
torch::Tensor edges_link_ids_tensor;
torch::Tensor edge_index;

// ML state vectors (CRITICAL for LSTM+GNN)
torch::Tensor h_vec;        // Hidden states for flows
torch::Tensor z_t_link;     // Hidden states for links

// Flow tracking
torch::Tensor link_to_graph_id;
torch::Tensor link_to_nflows;
torch::Tensor flow_to_graph_id;
torch::Tensor time_last;
torch::Tensor flowid_active_mask;

// Results
torch::Tensor res_fct_tensor;
torch::Tensor res_sldn_tensor;
torch::Tensor sldn_est;

// Cache and counters
static torch::Tensor ones_cache;
int graph_id_counter = 0;
int graph_id_cur = 0;
int flow_id_in_prop = 0;
int current_flow = 0;
int n_flows_active = 0;
int n_flows_arrived = 0;
int n_flows_completed = 0;
float time_clock = 0.0f;
int completed_flow_id = 0;
int min_idx = 0;
float flow_arrival_time = 0.0f;
float flow_completion_time = 0.0f;

// Topology parameters
int num_per_tor = 0;
std::vector<std::vector<int>> flowid_to_linkid_offsets;
std::vector<int> flowid_to_linkid_flat;

// ML prediction service for HERD network simulation
// Forward declaration for ML-based FCT prediction
static void ml_predict_and_schedule_herd(uint64_t flow_size, void (*callback)(void*), void* ctx);

// ======================== ML Helper Functions ========================
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
        const std::string model_dir = "/data1/lichenni/m4/testbed/models_new_v3";
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

        models_loaded = true;
        std::cout << "[INFO] All models loaded and optimized for inference." << std::endl;
    }
}

void setup_m4_tensors_for_herd(torch::Device device, int32_t max_flows, int32_t n_links, int32_t h_vec_dim) {
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool);

    // Initialize state tensors for HERD flows
    h_vec = torch::zeros({max_flows, h_vec_dim}, options_float).to(device);
    z_t_link = torch::zeros({n_links, h_vec_dim}, options_float).to(device);
    
    // Initialize z_t_link as in reference implementation
    auto ar_links = torch::arange(n_links, torch::TensorOptions().dtype(torch::kLong).device(device));
    z_t_link.index_put_({ar_links, 1}, 1.0f);
    z_t_link.index_put_({ar_links, 2}, 1.0f);
    
    // Flow tracking
    flowid_active_mask = torch::zeros({max_flows}, options_bool).to(device);
    time_last = torch::zeros({max_flows}, options_float).to(device);
    flow_to_graph_id = torch::full({max_flows}, -1, options_int32).to(device);
    
    // Link tracking  
    link_to_graph_id = torch::full({n_links}, -1, options_int32).to(device);
    link_to_nflows = torch::zeros({n_links}, options_int32).to(device);
    
    // Results
    res_fct_tensor = torch::zeros({max_flows, 2}, options_float).to(device);
    res_sldn_tensor = torch::zeros({max_flows, 2}, options_float).to(device);
    
    // Cache for efficient operations
    ones_cache = torch::ones({n_links}, options_int32).to(device);
    
    std::cout << "[INFO] M4 tensors initialized for max " << max_flows << " HERD flows\n";
}

// Global flow ID counter for HERD flows
static int g_herd_flow_id = 0;

// ML prediction service for HERD flows with full LSTM+GNN+state maintenance
static void ml_predict_and_schedule_herd(uint64_t flow_size, void (*callback)(void*), void* ctx) {
    torch::NoGradGuard no_grad;
    
    try {
        // Get current flow ID and increment counter
        int flow_id = g_herd_flow_id++;
        
        // Create synthetic flow data for this HERD flow
        float current_time = (float)g_event_queue->get_current_time();
        
        // Add flow to active mask and set arrival time
        flowid_active_mask[flow_id] = true;
        time_last[flow_id] = current_time;
        n_flows_active++;
        
        // Initialize h_vec for this flow (matching reference implementation)
        float normalized_size = std::log2f((float)flow_size + 1.0f);
        h_vec[flow_id][0] = 1.0f;  // Constant feature
        h_vec[flow_id][2] = normalized_size;  // Normalized flow size
        h_vec[flow_id][3] = 6.0f; //1.0f;  // Number of links (1 for HERD)
        
        // For HERD, create simple topology mapping (client -> server path)
        // This is a simplified version - in full M4, this comes from topology files
        std::vector<int> flow_links = {0}; // Single link for simple HERD topology
        
        // Update link states for this flow
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
        
        // Simulate link assignment (simplified for HERD)
        torch::Tensor links_tensor = torch::tensor(flow_links, options_int32);
        link_to_nflows.index_add_(0, links_tensor, ones_cache.slice(0, 0, links_tensor.size(0)));
        
        // Graph ID assignment (simplified)
        if (graph_id_counter == 0) {
            graph_id_cur = graph_id_counter++;
        }
        flow_to_graph_id[flow_id] = graph_id_cur;
        link_to_graph_id.index_put_({links_tensor}, graph_id_cur);
        
        // Prepare input tensors for ML prediction (using already calculated normalized_size)
        torch::Tensor size_input = torch::tensor({normalized_size}, options_float);
        torch::Tensor nlinks_input = torch::tensor({1.0f}, options_float); // Single link
        torch::Tensor params_input = torch::from_blob(params.data(), {(int)params.size()}, torch::TensorOptions().dtype(torch::kFloat64)).to(torch::kFloat32).to(device);
        
        // PROPER M4 ML PIPELINE: LSTM + GNN + MLP for state updates and prediction
        
        // Step 1: Update LSTM time states for ALL active flows (parity with inference_old)
        auto options_int64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
        auto active_flow_indices_time = torch::nonzero(flowid_active_mask).flatten();
        int n_active_time = active_flow_indices_time.size(0);
        if (n_active_time > 0) {
            // Compute elapsed time since last update (ns), then use max delta in μs
            auto last_times = time_last.index_select(0, active_flow_indices_time);
            auto dt_ns = (torch::tensor(current_time, options_float) - last_times).view({-1, 1});
            float max_dt_ns = torch::max(dt_ns).item<float>();
            if (max_dt_ns > 0.0f) {
                auto dt_us = torch::full({n_active_time, 1}, max_dt_ns / 1000.0f, options_float);
                auto h_vec_time = h_vec.index_select(0, active_flow_indices_time);
                h_vec_time = lstmcell_time.forward({dt_us, h_vec_time}).toTensor();
                h_vec.index_copy_(0, active_flow_indices_time, h_vec_time);

                // Update link time state (single active link: 0)
                auto link_idx = torch::tensor({(int64_t)0}, options_int64);
                auto z_link_time = z_t_link.index_select(0, link_idx);
                auto dt_us_link = torch::full({1, 1}, max_dt_ns / 1000.0f, options_float);
                z_link_time = lstmcell_time_link.forward({dt_us_link, z_link_time}).toTensor();
                z_t_link.index_copy_(0, link_idx, z_link_time);

                // Update time_last for all active flows
                time_last.index_put_({active_flow_indices_time}, current_time);
            }
        }
        
        // Step 2: GNN inference for spatial message passing
        if (n_flows_active > 1) { // Only run GNN if multiple active flows
            // Get all active flows
            auto active_flow_indices = torch::nonzero(flowid_active_mask).flatten();
            auto h_flows_active = h_vec.index_select(0, active_flow_indices); // [n_active, h_dim]
            auto z_links_active = z_t_link.slice(0, 0, 1); // [1, h_dim] for our single link
            
            // Create edge index for GNN (flows connected to link 0)
            int n_active = active_flow_indices.size(0);
            auto options_int64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
            torch::Tensor flow_indices = torch::arange(n_active, options_int64);
            torch::Tensor link_indices = torch::full({n_active}, (int64_t)n_active, options_int64); // Link at position n_active
            
            // Bidirectional edges: flow->link and link->flow
            torch::Tensor edges = torch::cat({
                torch::stack({flow_indices, link_indices}, 0),
                torch::stack({link_indices, flow_indices}, 0)
            }, 1).to(torch::kLong); // [2, 2*n_active], int64 for torch_geometric
            
            // Combined node features: [flows, links]
            torch::Tensor x_combined = torch::cat({h_flows_active, z_links_active}, 0); // [n_active+1, h_dim]
            
            // Forward through GNN layers
            torch::Tensor gnn_out_0 = gnn_layer_0.forward({x_combined, edges}).toTensor();
            torch::Tensor gnn_out_1 = gnn_layer_1.forward({gnn_out_0, edges}).toTensor();
            torch::Tensor gnn_out_2 = gnn_layer_2.forward({gnn_out_1, edges}).toTensor();
            
            // Split back into flow and link features
            torch::Tensor h_flows_updated = gnn_out_2.slice(0, 0, n_active); // [n_active, h_dim]
            torch::Tensor z_links_updated = gnn_out_2.slice(0, n_active, n_active + 1); // [1, h_dim]
            
            // Step 3: LSTM rate updates with GNN output
            torch::Tensor params_expanded = params_input.unsqueeze(0).repeat({n_active, 1}); // [n_active, 13]
            torch::Tensor h_rate_input = torch::cat({h_flows_updated, params_expanded}, 1); // [n_active, h_dim+13]
            torch::Tensor h_flows_old = h_vec.index_select(0, active_flow_indices);
            
            torch::Tensor h_flows_final = lstmcell_rate.forward({h_rate_input, h_flows_old}).toTensor();
            // Use the current time-updated link hidden state as second input
            auto z_link_time_cur = z_t_link.slice(0, 0, 1);
            torch::Tensor z_links_final = lstmcell_rate_link.forward({z_links_updated, z_link_time_cur}).toTensor();
            
            // Update global state tensors
            h_vec.index_copy_(0, active_flow_indices, h_flows_final);
            z_t_link.slice(0, 0, 1).copy_(z_links_final);
        }
        
        // Step 4: MLP prediction for ALL active flows
        auto active_flow_indices = torch::nonzero(flowid_active_mask).flatten();
        int n_active = active_flow_indices.size(0);
        
        float predicted_fct;
        float sldn_pred;
        float ideal_fct;
        
        // Check if this is the first/only flow (n_active == 1 means only current flow is active)
        if (n_active == 1) {
            // First flow - use ML prediction with initial state
            torch::Tensor nlinks_single = torch::full({1, 1}, 6.0f, options_float);
            torch::Tensor params_single = params_input.unsqueeze(0); // [1, 13]
            torch::Tensor h_single = h_vec.slice(0, flow_id, flow_id + 1); // [1, h_dim]
            
            torch::Tensor mlp_input_single = torch::cat({nlinks_single, params_single, h_single}, 1);
            torch::Tensor sldn_single = output_layer.forward({mlp_input_single}).toTensor().view(-1);
            sldn_single = torch::clamp(sldn_single, 1.0f, std::numeric_limits<float>::infinity());
            
            sldn_pred = sldn_single[0].item<float>();
            ideal_fct = 3000.0f + (float)flow_size / 1.25f;
            predicted_fct = sldn_pred * ideal_fct;
        } else if (n_active > 1) {
            // Multiple flows active - use full ML pipeline with contention
            torch::Tensor nlinks_expanded = torch::full({n_active, 1}, 6.0f, options_float);
            torch::Tensor params_expanded = params_input.unsqueeze(0).repeat({n_active, 1}); // [n_active, 13]
            torch::Tensor h_active = h_vec.index_select(0, active_flow_indices); // [n_active, h_dim]
            
            torch::Tensor mlp_input = torch::cat({nlinks_expanded, params_expanded, h_active}, 1); // [n_active, 1+13+h_dim]
            
            // Get slowdown predictions for all active flows
            torch::Tensor sldn_all = output_layer.forward({mlp_input}).toTensor().view(-1); // [n_active]
            sldn_all = torch::clamp(sldn_all, 1.0f, std::numeric_limits<float>::infinity());
            
            // Find the completion time for the current flow
            int current_flow_idx = -1;
            for (int i = 0; i < n_active; i++) {
                if (active_flow_indices[i].item<int>() == flow_id) {
                    current_flow_idx = i;
                    break;
                }
            }
            
            if (current_flow_idx >= 0) {
                sldn_pred = sldn_all[current_flow_idx].item<float>();
                ideal_fct = 3000.0f + (float)flow_size / 1.25f;
                predicted_fct = sldn_pred * ideal_fct;
            } else {
                // This should never happen - flow must be in active list
                throw std::runtime_error("Flow ID " + std::to_string(flow_id) + " not found in active flows list");
            }
        } else {
            // This should never happen - we just added the current flow to active mask
            throw std::runtime_error("No active flows found after adding current flow - this is a bug!");
        }
        
        std::cout << "ML STATE: flow_id=" << flow_id << " size=" << flow_size 
                  << "B, active_flows=" << n_flows_active 
                  << ", slowdown=" << sldn_pred
                  << ", ideal_fct=" << ideal_fct << "ns"
                  << ", predicted_fct=" << predicted_fct << "ns" << std::endl;
        
        // Store prediction for completion handling
        res_fct_tensor[flow_id][0] = predicted_fct;
        res_sldn_tensor[flow_id][0] = sldn_pred;
        
        // Schedule completion callback
        EventTime completion_time = g_event_queue->get_current_time() + (EventTime)predicted_fct;
        
        // Create completion context that includes flow_id for state cleanup
        struct CompletionCtx {
            void* original_ctx;
            void (*original_callback)(void*);
            int flow_id;
        };
        
        auto* comp_ctx = new CompletionCtx{ctx, callback, flow_id};
        
        // Schedule completion with state cleanup wrapper
        g_event_queue->schedule_event(completion_time, [](void* arg) {
            auto* comp_ctx = static_cast<CompletionCtx*>(arg);
            
            // Clean up ML state for completed flow
            flowid_active_mask[comp_ctx->flow_id] = false;
        n_flows_active--;
        n_flows_completed++;
            
            // Update link states (decrement flow count)
            std::vector<int> flow_links = {0}; // Same link as used in arrival
            auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
            torch::Tensor links_tensor = torch::tensor(flow_links, options_int32);
        link_to_nflows.index_add_(0, links_tensor, -ones_cache.slice(0, 0, links_tensor.size(0)));
            
            std::cout << "ML COMPLETE: flow_id=" << comp_ctx->flow_id 
                      << ", active_flows=" << n_flows_active << std::endl;
            
            // Call original callback
            comp_ctx->original_callback(comp_ctx->original_ctx);
            
            // Clean up completion context
            delete comp_ctx;
        }, comp_ctx);
        
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: ML prediction failed for flow size " << flow_size << ": " << e.what() << std::endl;
        std::cerr << "M4 ML pipeline is required - no fallbacks allowed!" << std::endl;
        throw; // Re-throw the exception to terminate the program
    }
}

// Emit flowsim-style staged records based on ML-predicted FCT
static void generate_herd_messages_m4(int flow_id, uint64_t flow_start_ns, uint64_t ml_fct_ns, uint64_t large_size_bytes, int client_id = 0, int worker_id = 0) {
    // All timing should come from ML predictions, not hardcoded delays
    // The ml_fct_ns is the ML-predicted FCT for this specific flow
    // We need to break it down into stages, but let ML drive the timing
    uint64_t req_send_time = flow_start_ns;                    // client sends 17B GET
    uint64_t req_recv_time = req_send_time + (ml_fct_ns * 15 / 100);  // c2s stage (~15% of total ML FCT)
    uint64_t resp_send_time = req_recv_time + SERVER_OVERHEAD_NS; // server compute (fixed processing)
    uint64_t resp_recv_time = resp_send_time + (ml_fct_ns * 10 / 100); // s2c UD stage (~10% of total ML FCT)
    uint64_t handshake_send_time = resp_recv_time + HANDSHAKE_DELAY_NS; // client handshake emit (fixed processing)
    uint64_t handshake_recv_time = handshake_send_time + (ml_fct_ns * 5 / 100); // handshake c2s (~5% of total ML FCT)
    uint64_t large_completion_time = handshake_recv_time + (ml_fct_ns * 70 / 100); // s2c RDMA stage (~70% of total ML FCT)

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

    // Add proper FlowRecord entries (matching flowsim format exactly)
    FlowRecord rec1; // c2s_get
    rec1.op_index = flow_id; rec1.client_id = client_id; rec1.worker_id = worker_id; rec1.slot = 0;
    rec1.req_bytes = 17; rec1.resp_bytes = 0; 
    rec1.start_ns = req_send_time; rec1.end_ns = req_recv_time; 
    rec1.fct_ns = req_recv_time - req_send_time; rec1.stage = "c2s_get";
    g_flow_records.push_back(rec1);
    
    FlowRecord rec2; // s2c_ud  
    rec2.op_index = flow_id; rec2.client_id = client_id; rec2.worker_id = worker_id; rec2.slot = 0;
    rec2.req_bytes = 0; rec2.resp_bytes = RESP_UD_BYTES;
    rec2.start_ns = resp_send_time; rec2.end_ns = resp_recv_time;
    rec2.fct_ns = resp_recv_time - resp_send_time; rec2.stage = "s2c_ud";
    g_flow_records.push_back(rec2);
    
    FlowRecord rec3; // c2s_handshake
    rec3.op_index = flow_id; rec3.client_id = client_id; rec3.worker_id = worker_id; rec3.slot = 0;
    rec3.req_bytes = 10; rec3.resp_bytes = 0;
    rec3.start_ns = handshake_send_time; rec3.end_ns = handshake_recv_time;
    rec3.fct_ns = handshake_recv_time - handshake_send_time; rec3.stage = "c2s_handshake";
    g_flow_records.push_back(rec3);
    
    FlowRecord rec4; // s2c_rdma
    rec4.op_index = flow_id; rec4.client_id = client_id; rec4.worker_id = worker_id; rec4.slot = 0;
    rec4.req_bytes = 0; rec4.resp_bytes = large_size_bytes;
    rec4.start_ns = handshake_recv_time; rec4.end_ns = large_completion_time;
    rec4.fct_ns = large_completion_time - handshake_recv_time; rec4.stage = "s2c_rdma";
    g_flow_records.push_back(rec4);
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

// HERD callbacks - copied exactly from flowsim
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
        rec.end_ns = g_event_queue->get_current_time();
        rec.fct_ns = (uint64_t)(rec.end_ns - rec.start_ns);
        rec.stage = ctx->is_handshake ? "c2s_handshake" : "c2s_get";
        g_flow_records.push_back(rec);
    }
    // No extra propagation here; Topology models per-hop latency
    EventTime when = g_event_queue->get_current_time();
    g_event_queue->schedule_event(when, (void (*)(void*)) &worker_recv, ctx);
}

static void worker_recv(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log request receive at worker (after propagation)
    EventTime when;
    if (!ctx->is_handshake) {
        g_server_log << "event=reqq_recv ts_ns=" << g_event_queue->get_current_time()
                     << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id
                     << " wrkr=" << ctx->worker_id
                     << " slot=" << ctx->slot
                     << " size=" << ctx->req_bytes
                     << " src=client:" << ctx->client_id
                     << " dst=worker:" << ctx->worker_id
                     << "\n";
        when = g_event_queue->get_current_time() + SERVER_OVERHEAD_NS;

    } else {
        g_server_log << "event=hand_recv ts_ns=" << g_event_queue->get_current_time()
                     << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id
                     << " wrkr=" << ctx->worker_id
                     << " slot=" << ctx->slot
                     << " size=" << ctx->req_bytes
                     << " src=client:" << ctx->client_id
                     << " dst=worker:" << ctx->worker_id
                     << "\n";
        when = g_event_queue->get_current_time(); //no overhead for handshake (RDMA DMA technique)
    }
    // Schedule worker send after overhead
    g_event_queue->schedule_event(when, (void (*)(void*)) &worker_send, ctx);
}

static void worker_send(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log response send from worker
    // If responding to a handshake, also log a handshake_send event on the server
    if (ctx->is_handshake) {
        g_server_log << "event=hand_conf ts_ns=" << g_event_queue->get_current_time()
                     << " id=" << ctx->op_index
                     << " clt=" << ctx->client_id
                     << " wrkr=" << ctx->worker_id
                     << " slot=" << ctx->slot
                     << " size=" << ctx->resp_bytes
                     << " src=worker:" << ctx->worker_id
                     << " dst=client:" << ctx->client_id
                     << "\n";
    } else {
        g_server_log << "event=resp_send ts_ns=" << g_event_queue->get_current_time()
        << " id=" << ctx->op_index
        << " clt=" << ctx->client_id
        << " wrkr=" << ctx->worker_id
        << " slot=" << ctx->slot
        << " size=" << ctx->resp_bytes
        << " src=worker:" << ctx->worker_id
        << " dst=client:" << ctx->client_id
        << "\n";
    }
    ctx->server_send_time = g_event_queue->get_current_time();
    // Use M4 ML pipeline to predict completion time and schedule callback
    ml_predict_and_schedule_herd(ctx->resp_bytes, (void (*)(void*)) &on_response_arrival, ctx);
}

static void on_response_arrival(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    EventTime when = g_event_queue->get_current_time();
    if (ctx->is_handshake) g_event_queue->schedule_event(when, (void (*)(void*)) &client_recv_rdma_finalize, ctx);
    else g_event_queue->schedule_event(when, (void (*)(void*)) &client_recv_ud, ctx);
}

// Client receives the small UD-style response and immediately sends handshake
static void client_recv_ud(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log: resp_recv_ud
    if (ctx->client_id >= 0 && ctx->client_id < (int)g_client_logs.size() && g_client_logs[ctx->client_id].is_open()) {
        g_client_logs[ctx->client_id] << "event=resp_recv_ud ts_ns=" << g_event_queue->get_current_time()
                 << " id=" << ctx->op_index
                 << " clt=" << ctx->client_id
                 << " wrkr=" << ctx->worker_id
                 << " slot=" << ctx->slot
                 << " size=" << ctx->resp_bytes
                 << " src=worker:" << ctx->worker_id
                 << " dst=client:" << ctx->client_id
                 << "\n";
    }
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
        rec.end_ns = g_event_queue->get_current_time();
        rec.fct_ns = (uint64_t)(rec.end_ns - rec.start_ns);
        rec.stage = "s2c_ud";
        g_flow_records.push_back(rec);
    }
    // Schedule handshake after HANDSHAKE_DELAY_NS
    ctx->is_handshake = true;
    ctx->req_bytes = HANDSHAKE_BYTES;
    EventTime when = g_event_queue->get_current_time() + HANDSHAKE_DELAY_NS;
    g_event_queue->schedule_event(when, (void (*)(void*)) &client_send_handshake, ctx);
}

// Client sends the handshake after the configured delay
static void client_send_handshake(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    ctx->start_time = g_event_queue->get_current_time();
    ctx->handshake_send_time = ctx->start_time;
    if (ctx->client_id >= 0 && ctx->client_id < (int)g_client_logs.size() && g_client_logs[ctx->client_id].is_open()) {
        g_client_logs[ctx->client_id] << "event=hand_send ts_ns=" << ctx->start_time
                 << " id=" << ctx->op_index
                 << " clt=" << ctx->client_id
                 << " wrkr=" << ctx->worker_id
                 << " slot=" << ctx->slot
                 << " size=" << ctx->req_bytes
                 << " src=client:" << ctx->client_id
                 << " dst=worker:" << ctx->worker_id
                 << "\n";
    }
    // Use M4 ML pipeline to predict completion time and schedule callback
    ml_predict_and_schedule_herd(ctx->req_bytes, (void (*)(void*)) &on_request_arrival, ctx);
}

// Client receives the large RDMA-style response, finalizes, and slides the window
static void client_recv_rdma_finalize(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log: resp_rdma_read
    uint64_t now_ns = (uint64_t)g_event_queue->get_current_time();
    uint64_t dur_ns = (ctx->handshake_send_time <= now_ns) ? (now_ns - (uint64_t)ctx->handshake_send_time) : 0;
    if (ctx->client_id >= 0 && ctx->client_id < (int)g_client_logs.size() && g_client_logs[ctx->client_id].is_open()) {
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
    }
    // Record s2c stage for RDMA-sized response and finalize FCT
    {
        FlowRecord rec;
        rec.op_index = ctx->op_index;
        rec.client_id = ctx->client_id;
        rec.worker_id = ctx->worker_id;
        rec.slot = ctx->slot;
        rec.req_bytes = 0;
        rec.resp_bytes = ctx->resp_bytes;
        rec.start_ns = ctx->server_send_time;
        rec.end_ns = g_event_queue->get_current_time();
        rec.fct_ns = (uint64_t)(rec.end_ns - rec.start_ns);
        rec.stage = "s2c_rdma";
        g_flow_records.push_back(rec);
    }
    int client_id = ctx->client_id;
    // Completion reduces inflight window by one
    if (g_inflight_per_client[client_id] > 0) g_inflight_per_client[client_id]--;
    
    // Flow completion handled by EventQueue - no ML state cleanup needed
    // The ML prediction service is stateless
    
    // Note: Not deleting ctx to avoid double-free issues in short HERD runs
    // Memory will be reclaimed at process exit
    // Slide the window: issue a new GET if under limit
    if (g_total_sent_per_client[client_id] < g_client_limit[client_id]) {
        int* cid = (int*)malloc(sizeof(int)); *cid = client_id;
        g_event_queue->schedule_event(g_event_queue->get_current_time(), (void (*)(void*)) &add_flow_for_client, cid);
    }
}

static void client_start_batch(void* arg) {
    // Prime sliding window: issue WINDOW_SIZE GETs per client at start
    int client_id = arg ? *(int*)arg : 0; if (arg) free(arg);
    EventTime base = g_event_queue->get_current_time();
    uint32_t to_send = WINDOW_SIZE;
    for (uint32_t i = 0; i < to_send && g_total_sent_per_client[client_id] < g_client_limit[client_id]; i++) {
        int* cid = (int*)malloc(sizeof(int)); *cid = client_id;
        // Apply a one-time startup delay between the first and second send
        EventTime extra = (i > 0 ? STARTUP_DELAY_NS : 0);
        EventTime send_time = base + extra + (EventTime)(i * SEND_SPACING_NS);
        g_event_queue->schedule_event(send_time, (void (*)(void*)) &add_flow_for_client, cid);
    }
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
    ctx->start_time = g_event_queue->get_current_time();

    g_clients[ctx->client_id].ws[wn] = (g_clients[ctx->client_id].ws[wn] + 1) % WINDOW_SIZE;

    if (ctx->client_id >= 0 && ctx->client_id < (int)g_client_logs.size() && g_client_logs[ctx->client_id].is_open()) {
        g_client_logs[ctx->client_id] << "event=req_send ts_ns=" << g_event_queue->get_current_time()
                 << " id=" << ctx->op_index
                 << " clt=" << ctx->client_id
                 << " wrkr=" << ctx->worker_id
                 << " slot=" << ctx->slot
                 << " size=" << ctx->req_bytes
                 << " src=client:" << ctx->client_id
                 << " dst=worker:" << ctx->worker_id
                 << "\n";
    }
    // Use M4 ML pipeline to predict completion time and schedule callback
    ml_predict_and_schedule_herd(ctx->req_bytes, (void (*)(void*)) &on_request_arrival, ctx);
    g_total_sent_per_client[client_id]++;
    g_inflight_per_client[client_id]++;
}

int main(int argc, char *argv[]) {
    if (argc >= 2) {
        int ws = std::atoi(argv[1]);
        if (ws > 0) WINDOW_SIZE = ws;
    }
    if (argc >= 3) {
        uint64_t rdma = std::strtoull(argv[2], nullptr, 10);
        if (rdma > 0) RESP_RDMA_BYTES = rdma;
    }
    // Mirror flowsim: run HERD-only mode when not enough args; else run ML+HERD
    if (argc < 6) {
        g_event_queue = std::make_shared<EventQueue>();
        Topology::set_event_queue(g_event_queue);

        // Topology switch: single client/server or multi-client tree (matching flowsim)
        bool multi_client_topo = false; // use multi-client topologies
        bool twelve_node_topo = true; // true: 12-endpoint tree, false: original 4-endpoint (server + 3 clients)
        double bw_bpns = 10.0;
        
        if (!multi_client_topo) {
            // Single client (0) <-> server (1)
            g_topology = std::make_shared<Topology>(2, 2);
            g_topology->connect(0, 1, bw_bpns, 3500, true);
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
            Route c2s; c2s.push_back(g_topology->get_device(0)); c2s.push_back(g_topology->get_device(1));
            Route s2c; s2c.push_back(g_topology->get_device(1)); s2c.push_back(g_topology->get_device(0));
            g_routes_c2s[0] = c2s; g_routes_s2c[0] = s2c;
            g_client_logs[0].open("client_0.log");
        } else if (twelve_node_topo) {
            // 12-endpoint multi-client topology with dual roots R and R2 and top T
            // Device indices:
            // 0:T (worker/server), 1:R, 2:R2, 3:S1, 4:S2, 5:S3, 6:S4, 7:S5, 8:S6,
            // 10:C1, 11:C2, 12:C3, 13:C4, 14:C5, 15:C6, 16:C7, 17:C8, 18:C9, 19:C10, 20:C11
            g_topology = std::make_shared<Topology>(21, 4);
            // Core links
            g_topology->connect(0, 1, bw_bpns, 800.0, true); // T<->R
            g_topology->connect(0, 2, bw_bpns, 800.0, true); // T<->R2
            // Aggregation: R side
            g_topology->connect(1, 3, bw_bpns, 800.0, true); // R<->S1
            g_topology->connect(1, 4, bw_bpns, 800.0, true); // R<->S2
            g_topology->connect(1, 5, bw_bpns, 800.0, true); // R<->S3
            // Aggregation: R2 side
            g_topology->connect(2, 6, bw_bpns, 800.0, true); // R2<->S4
            g_topology->connect(2, 7, bw_bpns, 800.0, true); // R2<->S5
            g_topology->connect(2, 8, bw_bpns, 800.0, true); // R2<->S6
            // Leaves to ToR switches per mapping
            g_topology->connect(13, 3, bw_bpns, 800.0, true); // C4<->S1
            g_topology->connect(10, 4, bw_bpns, 800.0, true); // C1<->S2
            g_topology->connect(14, 4, bw_bpns, 800.0, true); // C5<->S2
            g_topology->connect(11, 5, bw_bpns, 800.0, true); // C2<->S3
            g_topology->connect(12, 5, bw_bpns, 800.0, true); // C3<->S3
            g_topology->connect(15, 6, bw_bpns, 800.0, true); // C6<->S4
            g_topology->connect(16, 6, bw_bpns, 800.0, true); // C7<->S4
            g_topology->connect(17, 7, bw_bpns, 800.0, true); // C8<->S5
            g_topology->connect(18, 7, bw_bpns, 800.0, true); // C9<->S5
            g_topology->connect(19, 8, bw_bpns, 800.0, true); // C10<->S6
            g_topology->connect(20, 8, bw_bpns, 800.0, true); // C11<->S6

            NUM_CLIENTS = 11; // C1..C11 (Server is the worker at T)
            g_clients.assign(NUM_CLIENTS, ClientState());
            g_client_logs.resize(NUM_CLIENTS);
            for (int i = 0; i < NUM_CLIENTS; i++) {
                g_clients[i].id = i;
                g_clients[i].key_perm = herd_get_random_permutation(HERD_NUM_KEYS, i, &g_clients[i].seed);
            }
            g_routes_c2s.resize(NUM_CLIENTS);
            g_routes_s2c.resize(NUM_CLIENTS);

            auto build_route = [&](std::vector<int> path){ Route r; for(int id: path) r.push_back(g_topology->get_device(id)); return r; };
            // Clients 0..10 map to: C4, C1, C5, C2, C3, C6, C7, C8, C9, C10, C11
            std::vector<std::vector<int>> c2s_paths = {
                {13,3,1,0},
                {10,4,1,0},
                {14,4,1,0},
                {11,5,1,0},
                {12,5,1,0},
                {15,6,2,0},
                {16,6,2,0},
                {17,7,2,0},
                {18,7,2,0},
                {19,8,2,0},
                {20,8,2,0}
            };
            for (int cid = 0; cid < NUM_CLIENTS; cid++) {
                g_routes_c2s[cid] = build_route(c2s_paths[cid]);
                auto rev = c2s_paths[cid];
                std::reverse(rev.begin(), rev.end());
                g_routes_s2c[cid] = build_route(rev);
            }
            for (int i = 0; i < NUM_CLIENTS; i++) {
                g_client_logs[i].open((std::string("client_") + std::to_string(i) + ".log").c_str());
            }
        } else {
            // Original 4-endpoint topology: 1 server + 3 clients (8 devices total with switches)
            // Server A (0) -> S (1) -> Root R (2)
            // Client B (3) -> S1 (7) -> R (2)
            // Client C (4) -> S2 (5) -> R (2)
            // Client D (6) -> S2 (5) -> R (2)
            g_topology = std::make_shared<Topology>(8, 4); // 8 devices total
            // Links with 800ns latency and 10B/ns bandwidth (matching flowsim exactly)
            g_topology->connect(0, 1, bw_bpns, 800.0, true); // A<->S
            g_topology->connect(1, 2, bw_bpns, 800.0, true); // S<->R
            g_topology->connect(3, 7, bw_bpns, 800.0, true); // B<->S1
            g_topology->connect(4, 5, bw_bpns, 800.0, true); // C<->S2
            g_topology->connect(5, 2, bw_bpns, 800.0, true); // S2<->R
            g_topology->connect(6, 5, bw_bpns, 800.0, true); // D<->S2
            g_topology->connect(7, 2, bw_bpns, 800.0, true); // S1<->R
            NUM_CLIENTS = 3;
            g_clients.assign(NUM_CLIENTS, ClientState());
            g_client_logs.resize(NUM_CLIENTS);
            for (int i = 0; i < NUM_CLIENTS; i++) {
                g_clients[i].id = i;
                g_clients[i].key_perm = herd_get_random_permutation(HERD_NUM_KEYS, i, &g_clients[i].seed);
            }
            g_routes_c2s.resize(NUM_CLIENTS);
            g_routes_s2c.resize(NUM_CLIENTS);
            // Map client indices with full hop-by-hop routes (exactly like flowsim):
            // 0: B(3)->S1(7)->R(2)->S(1)->A(0)
            // 1: C(4)->S2(5)->R(2)->S(1)->A(0)
            // 2: D(6)->S2(5)->R(2)->S(1)->A(0)
            auto build_route = [&](std::vector<int> path){ Route r; for(int id: path) r.push_back(g_topology->get_device(id)); return r; };
            g_routes_c2s[0] = build_route({3,7,2,1,0}); g_routes_s2c[0] = build_route({0,1,2,7,3});
            g_routes_c2s[1] = build_route({4,5,2,1,0}); g_routes_s2c[1] = build_route({0,1,2,5,4});
            g_routes_c2s[2] = build_route({6,5,2,1,0}); g_routes_s2c[2] = build_route({0,1,2,5,6});
            g_client_logs[0].open("client_0.log");
            g_client_logs[1].open("client_1.log");
            g_client_logs[2].open("client_2.log");
        }

        // Open server log
        g_server_log.open("server.log");

        // Setup ML models for HERD network prediction service
        std::cout << "Loading M4 ML models for HERD network prediction...\n";
        
        try {
            // Read config file to get ML model parameters
            std::string config_path = "config/test_config_testbed.yaml";
            std::ifstream infile_cfg(config_path);
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

            // Initialize basic parameters for ML prediction
            params.assign(13, 0.0); // Default parameters

            // Load ML models
            setup_m4(device);
            
            // Initialize ML state tensors for HERD flows
            int max_herd_flows = 4 * 650; 2600*11; // 4 * 650 operations (request, UD, handshake, RDMA per op)
            setup_m4_tensors_for_herd(device, max_herd_flows, n_links, hidden_size);
            
            std::cout << "M4 ML models and tensors loaded successfully for HERD prediction service\n";
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Could not load ML models: " << e.what() << std::endl;
            return 1;
        }

        // Initialize per-client limits and state - copied exactly from flowsim
        const int default_ops = 650;//*11;
        g_pending_completions.assign(NUM_CLIENTS, std::deque<FlowCtx*>());
        g_poll_scheduled.assign(NUM_CLIENTS, false);
        g_total_sent_per_client.assign(NUM_CLIENTS, 0);
        g_client_limit.assign(NUM_CLIENTS, default_ops / NUM_CLIENTS);
        g_inflight_per_client.assign(NUM_CLIENTS, 0);

        // Schedule initial window
        EventTime t0 = g_event_queue->get_current_time() + 1;
        for (int cid = 0; cid < NUM_CLIENTS; cid++) {
            int* arg = (int*)malloc(sizeof(int)); *arg = cid;
            g_event_queue->schedule_event(t0, (void (*)(void*)) &client_start_batch, arg);
        }
        // Use EventQueue with HERD callbacks, exactly like flowsim
        while (!g_event_queue->finished()) {
            g_event_queue->proceed();
        }

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

    // Cleanup client key perms
    for (int i = 0; i < (int)g_clients.size(); i++) {
        if (g_clients[i].key_perm != nullptr) {
            free(g_clients[i].key_perm);
            g_clients[i].key_perm = nullptr;
        }
    }

    std::cout << "M4 HERD-sim completed ops: " << g_flow_records.size() << ", wrote flows.txt\n";
    return 0;
}
