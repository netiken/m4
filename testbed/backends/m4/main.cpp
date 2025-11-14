#include <string>
#include <chrono>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>
#include "Topology.h"
#include "TopologyBuilder.h"
#include "Type.h"

#include <iomanip>
#include <queue>
#include <algorithm>

// CityHash for 128-bit key generation (exactly as in HERD)
#include "rdma_bench/mica/mica.h" // Brings in city.h and MICA_MAX_VALUE

// HERD Protocol Constants
static constexpr int HERD_NUM_KEYS = (8 * 1024 * 1024);
static constexpr int NUM_WORKERS = 12;
static int NUM_CLIENTS = 1; // configurable per-topology mode
static int WINDOW_SIZE = 16; // per-worker ring slots per client

// Message Sizes
static constexpr uint64_t RESP_UD_BYTES = 41;
static constexpr uint64_t HANDSHAKE_BYTES = 10;
static uint64_t RESP_RDMA_BYTES = 1024008;

// Timing Parameters

static constexpr uint64_t SERVER_OVERHEAD_NS = 87000;
static constexpr uint64_t SEND_SPACING_NS = 2500;
static constexpr uint64_t STARTUP_DELAY_NS = 0;
static constexpr uint64_t HANDSHAKE_DELAY_NS = 8647;
// Network Parameters
static constexpr float MTU_BYTES = 1000.0f;
static constexpr float HEADER_SIZE_BYTES = 48.0f;
static constexpr float BYTES_TO_NS = 0.8f;
static constexpr float PROPAGATION_PER_LINK_NS = 1000.0f;

// RNG (HERD fastrand)
static inline uint32_t hrd_fastrand(uint64_t* seed) {
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t)(*seed >> 32);
}

// HERD value length derivation
static inline uint8_t herd_val_len_from_key_parts(uint64_t part0, uint64_t part1) {
    const uint8_t min_len = 8;
    const uint8_t max_len = MICA_MAX_VALUE; // 46 bytes for 64B mica_op
    const uint32_t range = (uint32_t)(max_len - min_len + 1);
    uint64_t mix = part0 ^ (part1 >> 32) ^ (part1 & 0xffffffffULL);
    return (uint8_t)(min_len + (mix % range));
}

// Generate random key permutation per client
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
    int* key_perm;
    int ws[NUM_WORKERS];

    ClientState() : id(0), seed(0xdeadbeef), key_perm(nullptr) {
        for (int i = 0; i < NUM_WORKERS; i++) ws[i] = 0;
    }
};

struct FlowCtx {
    int op_index;
    int client_id;
    int worker_id;
    int slot;
    bool is_update;
    uint8_t vlen;
    uint64_t req_bytes;
    uint64_t resp_bytes;
    bool is_handshake;
    EventTime start_time;
    EventTime req_send_time;
    EventTime server_send_time;
    EventTime handshake_send_time;
    Route route_fwd;
    Route route_rev;
    uint64_t ud_predicted_fct_ns;
    uint64_t rdma_predicted_fct_ns;
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
    std::string stage;
};

static std::vector<ClientState> g_clients;
static std::vector<Route> g_routes_c2s;
static std::vector<Route> g_routes_s2c;
static std::vector<FlowRecord> g_flow_records;
static std::vector<std::deque<FlowCtx*>> g_pending_completions;
static std::vector<bool> g_poll_scheduled;
static std::vector<uint64_t> g_total_sent_per_client;
static std::vector<uint64_t> g_client_limit;
static std::vector<uint32_t> g_inflight_per_client;
static uint64_t g_next_op_index = 0;
static std::vector<std::ofstream> g_client_logs;
static std::ofstream g_server_log;
static std::shared_ptr<EventQueue> g_event_queue;
static std::shared_ptr<Topology> g_topology;

// ML Pipeline State Variables
torch::Device device(torch::kCUDA, 0);

// Model components
torch::jit::script::Module lstmcell_time, lstmcell_rate;
torch::jit::script::Module lstmcell_time_link, lstmcell_rate_link;
torch::jit::script::Module gnn_layer_0, gnn_layer_1, gnn_layer_2;
torch::jit::script::Module output_layer;

// Flow and topology data
torch::Tensor flow_params_tensor;
#include <unordered_map>
#include <unordered_set>
static std::unordered_map<long long, int> g_link_index_map;
static std::vector<std::vector<int>> g_c2s_link_indices;
static std::vector<std::vector<int>> g_s2c_link_indices;
static std::vector<std::vector<int>> g_flow_links;
static int g_n_links_override = -1;

// ML state vectors
torch::Tensor h_vec;
torch::Tensor z_t_link;

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
int n_flows_active = 0;
int n_flows_completed = 0;

// Forward declaration
static void ml_predict_and_schedule_herd(uint64_t flow_size, void (*callback)(void*), void* ctx);

// ML Helper Functions

// Compute ideal FCT (network time only, no server overhead)
// Matches training formula: propagation + transmission + pipeline_fill
static inline float compute_ideal_fct(uint64_t flow_size, int n_links) {
    float num_packets = std::ceil((float)flow_size / MTU_BYTES);
    float total_bytes = (float)flow_size + num_packets * HEADER_SIZE_BYTES;
    float transmission_ns = total_bytes * BYTES_TO_NS;
    float propagation_ns = PROPAGATION_PER_LINK_NS * (float)(n_links - 1);
    float first_packet_bytes = std::min(MTU_BYTES, (float)flow_size) + HEADER_SIZE_BYTES;
    float pipeline_ns = first_packet_bytes * BYTES_TO_NS * (float)(n_links - 2);
    return propagation_ns + transmission_ns + pipeline_ns;
}

void setup_m4(torch::Device device, const std::string& model_dir = "checkpoints") {
    if (!torch::cuda::is_available()) {
        std::cerr << "[ERROR] CUDA is not available! M4 requires GPU for ML inference." << std::endl;
        std::cerr << "Please ensure:" << std::endl;
        std::cerr << "  1. PyTorch was installed with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" << std::endl;
        std::cerr << "  2. CUDA toolkit is installed: nvidia-smi should work" << std::endl;
        std::cerr << "  3. CMake built with CUDA enabled (Torch_USE_CUDA=ON)" << std::endl;
        std::exit(1);  // Exit instead of silent failure
    }
    
    std::cout << "[INFO] CUDA available! Using " << torch::cuda::device_count() << " GPU(s)" << std::endl;

    // Disable gradient calculations
    torch::NoGradGuard no_grad;

    // Load models (model_dir passed as parameter)
    static bool models_loaded = false;
    if (!models_loaded) {
        std::cout << "[INFO] Loading models from " << model_dir << std::endl;
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
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);

    // Initialize state tensors for HERD flows
    h_vec = torch::zeros({max_flows, h_vec_dim}, options_float);
    z_t_link = torch::zeros({n_links, h_vec_dim}, options_float);
    
    // Initialize z_t_link as in reference implementation
    auto ar_links = torch::arange(n_links, torch::TensorOptions().dtype(torch::kLong).device(device));
    z_t_link.index_put_({ar_links, torch::tensor(1)}, 1.0f);
    z_t_link.index_put_({ar_links, torch::tensor(2)}, 1.0f);
    
    // Flow tracking
    flowid_active_mask = torch::zeros({max_flows}, options_bool);
    time_last = torch::zeros({max_flows}, options_float);
    flow_to_graph_id = torch::full({max_flows}, -1, options_int32);
    
    // Link tracking  
    link_to_graph_id = torch::full({n_links}, -1, options_int32);
    link_to_nflows = torch::zeros({n_links}, options_int32);
    
    // Results
    res_fct_tensor = torch::zeros({max_flows, 2}, options_float);
    res_sldn_tensor = torch::zeros({max_flows, 2}, options_float);
    
    // Per-flow parameters (13-dimensional parameter vectors)
    flow_params_tensor = torch::zeros({max_flows, 13}, options_float);
    
    // Cache for efficient operations
    ones_cache = torch::ones({n_links}, options_int32);
    
    std::cout << "[INFO] M4 tensors initialized for max " << max_flows << " HERD flows with per-flow parameters\n";
    // Prepare per-flow link storage
    g_flow_links.clear();
    g_flow_links.resize(max_flows);
}

static int g_herd_flow_id = 0;
static void on_request_arrival(void* arg);
static void on_response_arrival(void* arg);
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
        
        // Determine link path for this flow (direction-dependent)
        auto* flow_ctx = static_cast<FlowCtx*>(ctx);
        int cid = flow_ctx->client_id;
        bool to_server = (callback == (void (*)(void*)) &on_request_arrival);
        const std::vector<int>& flow_links_vec = to_server ? g_c2s_link_indices[cid] : g_s2c_link_indices[cid];
        g_flow_links[flow_id] = flow_links_vec;
        
        // Initialize h_vec for this flow (matching reference implementation)
        // Use testbed size transformation (matches dataset.py logic with enable_testbed=True)
        float normalized_size = std::log2f((float)flow_size + 1.0f);

        h_vec[flow_id].zero_();
        h_vec[flow_id][0] = 1.0f;  // Constant feature
        h_vec[flow_id][2] = normalized_size;  // Normalized flow size  
        h_vec[flow_id][3] = static_cast<float>(g_flow_links[flow_id].size());  // Number of links along path
        
        // Set flow-specific parameters based on size (matches dataset.py parameter conditioning)
        if (flow_size >= 1000) {
            // Large flows get parameter vector: [1, 1, 1, ..., 1] (13 ones)
            flow_params_tensor[flow_id] = 1.0f;
        } else {
            // Small flows get parameter vector: [0, 0, 0, ..., 0] (13 zeros)  
            flow_params_tensor[flow_id] = 0.0f;
        }
        
        // Update link states for this flow
        auto cpu_int32 = torch::TensorOptions().dtype(torch::kInt32);
        torch::Tensor links_tensor = torch::from_blob(g_flow_links[flow_id].data(), {(long)g_flow_links[flow_id].size()}, cpu_int32).clone().to(device);
        if (links_tensor.numel() > 0) {
        link_to_nflows.index_add_(0, links_tensor, ones_cache.slice(0, 0, links_tensor.size(0)));
        }
        
        // Graph ID assignment (simplified)
        if (graph_id_counter == 0) {
            graph_id_cur = graph_id_counter++;
        }
        flow_to_graph_id[flow_id] = graph_id_cur;
        if (links_tensor.numel() > 0) {
        link_to_graph_id.index_put_({links_tensor}, graph_id_cur);
        }
        
        
        // PROPER M4 ML PIPELINE: LSTM + GNN + MLP for state updates and prediction
        
        // Step 1: Update LSTM time states for ALL active flows
        auto active_flow_indices_time = torch::nonzero(flowid_active_mask).flatten();
        int n_active_time = active_flow_indices_time.size(0);
        if (n_active_time > 0) {
            // Compute elapsed time since last update (ns), then use max delta in μs
            auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
            auto last_times = time_last.index_select(0, active_flow_indices_time);
            auto dt_ns = (torch::tensor(current_time, options_float) - last_times).view({-1, 1});
            float max_dt_ns = torch::max(dt_ns).item<float>();
            if (max_dt_ns > 0.0f) {
                auto dt_us = torch::full({n_active_time, 1}, max_dt_ns / 1000.0f, options_float);
                auto h_vec_time = h_vec.index_select(0, active_flow_indices_time);
                h_vec_time = lstmcell_time.forward(std::vector<torch::jit::IValue>{dt_us, h_vec_time}).toTensor();
                h_vec.index_copy_(0, active_flow_indices_time, h_vec_time);

                // Update link time state for all links used by any active flow
                std::vector<int64_t> union_links;
                {
                    std::unordered_set<int64_t> seen;
                    for (int i = 0; i < n_active_time; i++) {
                        int fid = active_flow_indices_time[i].item<int>();
                        for (int l : g_flow_links[fid]) {
                            if (seen.insert(l).second) union_links.push_back(l);
                        }
                    }
                }
                if (!union_links.empty()) {
                    auto link_idx = torch::from_blob(union_links.data(), {(int)union_links.size()}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(device);
                auto z_link_time = z_t_link.index_select(0, link_idx);
                    auto dt_us_link = torch::full({(int)union_links.size(), 1}, max_dt_ns / 1000.0f, options_float);
                z_link_time = lstmcell_time_link.forward(std::vector<torch::jit::IValue>{dt_us_link, z_link_time}).toTensor();
                z_t_link.index_copy_(0, link_idx, z_link_time);
                }

                // Update time_last for all active flows
                time_last.index_put_({active_flow_indices_time}, current_time);
            }
        }
        
        // Step 2: GNN inference for spatial message passing
        if (n_flows_active > 1) { // Only run GNN if multiple active flows
            // Get all active flows
            auto active_flow_indices = torch::nonzero(flowid_active_mask).flatten();
            int n_active = active_flow_indices.size(0);
            auto h_flows_active = h_vec.index_select(0, active_flow_indices); // [n_active, h_dim]

            // Build union of active links and map to local indices
            std::vector<int64_t> union_links;
            {
                std::unordered_set<int64_t> seen;
                for (int i = 0; i < n_active; i++) {
                    int fid = active_flow_indices[i].item<int>();
                    for (int l : g_flow_links[fid]) {
                        if (seen.insert(l).second) union_links.push_back(l);
                    }
                }
            }
            if (!union_links.empty()) {
                auto cpu_int64 = torch::TensorOptions().dtype(torch::kInt64);
                torch::Tensor link_idx_global = torch::from_blob(union_links.data(), {(int)union_links.size()}, cpu_int64).clone().to(device);
                torch::Tensor z_links_active = z_t_link.index_select(0, link_idx_global);

                // Build bipartite edges: flows [0..n_active-1], links [n_active..n_active+m-1]
                std::unordered_map<int64_t, int64_t> local_link_pos;
                for (size_t i = 0; i < union_links.size(); i++) local_link_pos[union_links[i]] = (int64_t)i;
                std::vector<int64_t> src, dst;
                for (int i = 0; i < n_active; i++) {
                    int fid = active_flow_indices[i].item<int>();
                    for (int l : g_flow_links[fid]) {
                        auto it = local_link_pos.find(l);
                        if (it == local_link_pos.end()) continue;
                        int64_t link_local = (int64_t)n_active + it->second;
                        src.push_back(i); dst.push_back(link_local); // flow -> link
                        src.push_back(link_local); dst.push_back(i); // link -> flow
                    }
                }
                torch::Tensor edges = torch::stack({torch::from_blob(src.data(), {(int)src.size()}, cpu_int64).clone().to(device), torch::from_blob(dst.data(), {(int)dst.size()}, cpu_int64).clone().to(device)}, 0);
            
            // Combined node features: [flows, links]
                torch::Tensor x_combined = torch::cat({h_flows_active, z_links_active}, 0);
            
            // Forward through GNN layers
            torch::Tensor gnn_out_0 = gnn_layer_0.forward(std::vector<torch::jit::IValue>{x_combined, edges}).toTensor();
            torch::Tensor gnn_out_1 = gnn_layer_1.forward(std::vector<torch::jit::IValue>{gnn_out_0, edges}).toTensor();
            torch::Tensor gnn_out_2 = gnn_layer_2.forward(std::vector<torch::jit::IValue>{gnn_out_1, edges}).toTensor();
            
            // Split back into flow and link features
            torch::Tensor h_flows_updated = gnn_out_2.slice(0, 0, n_active); // [n_active, h_dim]
                torch::Tensor z_links_updated = gnn_out_2.slice(0, n_active, n_active + (long)union_links.size());
            
            // Step 3: LSTM rate updates with GNN output
            torch::Tensor params_active = flow_params_tensor.index_select(0, active_flow_indices); // [n_active, 13]
            torch::Tensor h_rate_input = torch::cat({h_flows_updated, params_active}, 1); // [n_active, h_dim+13]
            // CRITICAL: Use TIME-UPDATED h_vec as second LSTM input, not the old one!
            torch::Tensor h_flows_time_updated = h_vec.index_select(0, active_flow_indices);
            
            torch::Tensor h_flows_final = lstmcell_rate.forward({h_rate_input, h_flows_time_updated}).toTensor();
            // Use the current time-updated link hidden state as second input
                auto z_link_time_cur = z_links_active;
            torch::Tensor z_links_final = lstmcell_rate_link.forward({z_links_updated, z_link_time_cur}).toTensor();
            
            // Update global state tensors
            h_vec.index_copy_(0, active_flow_indices, h_flows_final);
                z_t_link.index_copy_(0, link_idx_global, z_links_final);
            } else {
                // No links to update; propagate only flow states without GNN
                torch::Tensor params_active = flow_params_tensor.index_select(0, active_flow_indices);
                torch::Tensor h_rate_input = torch::cat({h_flows_active, params_active}, 1);
                torch::Tensor h_flows_old = h_vec.index_select(0, active_flow_indices);
                torch::Tensor h_flows_final = lstmcell_rate.forward({h_rate_input, h_flows_old}).toTensor();
                h_vec.index_copy_(0, active_flow_indices, h_flows_final);
            }
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
            auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
            int nlinks_val = (int)g_flow_links[flow_id].size();
            torch::Tensor nlinks_single = torch::full({1, 1}, (float)nlinks_val, options_float);
            torch::Tensor params_single = flow_params_tensor.slice(0, flow_id, flow_id + 1); // [1, 13]
            torch::Tensor h_single = h_vec.slice(0, flow_id, flow_id + 1); // [1, h_dim]
            
            torch::Tensor mlp_input_single = torch::cat({nlinks_single, params_single, h_single}, 1);
            torch::Tensor sldn_single = output_layer.forward(std::vector<torch::jit::IValue>{mlp_input_single}).toTensor().view(-1);
            sldn_single = torch::clamp(sldn_single, 1.0f, std::numeric_limits<float>::infinity());
            
            sldn_pred = sldn_single[0].item<float>();
            int nlinks_val_int = (int)g_flow_links[flow_id].size();
            ideal_fct = compute_ideal_fct(flow_size, nlinks_val_int);
            predicted_fct = sldn_pred * ideal_fct;
        } else if (n_active > 1) {
            // Multiple flows active - use full ML pipeline with contention
            auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
            // Build per-flow n_links counts
            std::vector<float> nlinks_vals(n_active);
            for (int i = 0; i < n_active; i++) {
                int fid = active_flow_indices[i].item<int>();
                nlinks_vals[i] = (float)g_flow_links[fid].size();
            }
            auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor nlinks_expanded = torch::from_blob(nlinks_vals.data(), {n_active, 1}, cpu_float).clone().to(device);
            torch::Tensor params_active = flow_params_tensor.index_select(0, active_flow_indices); // [n_active, 13]
            torch::Tensor h_active = h_vec.index_select(0, active_flow_indices); // [n_active, h_dim]
            
            torch::Tensor mlp_input = torch::cat({nlinks_expanded, params_active, h_active}, 1); // [n_active, 1+13+h_dim]
            
            // Get slowdown predictions for all active flows
            torch::Tensor sldn_all = output_layer.forward(std::vector<torch::jit::IValue>{mlp_input}).toTensor().view(-1); // [n_active]
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
                int nlinks_cur = (int)g_flow_links[flow_id].size();
                ideal_fct = compute_ideal_fct(flow_size, nlinks_cur);
                predicted_fct = sldn_pred * ideal_fct;
            } else {
                // This should never happen - flow must be in active list
                throw std::runtime_error("Flow ID " + std::to_string(flow_id) + " not found in active flows list");
            }
        } else {
            // This should never happen - we just added the current flow to active mask
            throw std::runtime_error("No active flows found after adding current flow - this is a bug!");
        }
        
        // Debug output showing parameter conditioning based on flow size
        bool is_large_flow = (flow_size >= 1000);
        std::cout << "ML STATE: flow_id=" << flow_id << " size=" << flow_size 
                  << "B, active_flows=" << n_flows_active 
                  << ", params=" << (is_large_flow ? "[1,1,...,1]" : "[0,0,...,0]")
                  << ", slowdown=" << sldn_pred
                  << ", ideal_fct=" << ideal_fct << "ns"
                  << ", predicted_fct=" << predicted_fct << "ns" << std::endl;
        
        // Store prediction for completion handling
        res_fct_tensor[flow_id][0] = predicted_fct;
        res_sldn_tensor[flow_id][0] = sldn_pred;
        
        // Store predicted FCT in the correct field based on flow phase
        // UD phase: small request/response (before handshake)
        // RDMA phase: handshake and large response (after handshake)
        if (flow_ctx->is_handshake) {
            flow_ctx->rdma_predicted_fct_ns = (uint64_t)predicted_fct;
        } else {
            flow_ctx->ud_predicted_fct_ns = (uint64_t)predicted_fct;
        }
        
        // Schedule completion with predicted FCT + server overhead
        // predicted_fct = slowdown * network_time (pure network delay)
        // Then scaled by WINDOW_SIZE and add server processing delay
        EventTime completion_time = g_event_queue->get_current_time() + (EventTime)predicted_fct*(float)WINDOW_SIZE + (EventTime)SERVER_OVERHEAD_NS;
        
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
            {
                auto cpu_int32 = torch::TensorOptions().dtype(torch::kInt32);
                const auto& links_v = g_flow_links[comp_ctx->flow_id];
                if (!links_v.empty()) {
                    torch::Tensor links_tensor = torch::from_blob((void*)links_v.data(), {(long)links_v.size()}, cpu_int32).clone().to(device);
        link_to_nflows.index_add_(0, links_tensor, -ones_cache.slice(0, 0, links_tensor.size(0)));
                    // Reset link state if no active flows remain on those links
                    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
                    auto no_flow_mask = (link_to_nflows.index({links_tensor}) == 0);
                    auto no_flow_links_tensor = links_tensor.masked_select(no_flow_mask);
                    if (no_flow_links_tensor.numel() > 0) {
                        link_to_graph_id.index_put_({no_flow_links_tensor}, -1);
                        auto reset_values = torch::zeros({no_flow_links_tensor.size(0), z_t_link.size(1)}, options_float);
                        auto ones_vals = torch::ones({no_flow_links_tensor.size(0)}, options_float);
                        z_t_link.index_put_({no_flow_links_tensor, torch::indexing::Slice()}, reset_values);
                        z_t_link.index_put_({no_flow_links_tensor, 1}, ones_vals);
                        z_t_link.index_put_({no_flow_links_tensor, 2}, ones_vals);
                    }
                }
            }
            
            // std::cout << "ML COMPLETE: flow_id=" << comp_ctx->flow_id 
            //           << ", active_flows=" << n_flows_active << std::endl;
            
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

// ======================== Helper Functions ========================

// Create and add FlowRecord efficiently
void add_flow_record(int op_index, int client_id, int worker_id, int slot,
                    uint64_t req_bytes, uint64_t resp_bytes,
                    EventTime start_ns, EventTime end_ns, const std::string& stage) {
    FlowRecord rec;
    rec.op_index = op_index;
    rec.client_id = client_id;
    rec.worker_id = worker_id;
    rec.slot = slot;
    rec.req_bytes = req_bytes;
    rec.resp_bytes = resp_bytes;
    rec.start_ns = start_ns;
    rec.end_ns = end_ns;
    rec.fct_ns = (uint64_t)(end_ns - start_ns);
    rec.stage = stage;
    g_flow_records.push_back(rec);
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

// HERD Protocol Callbacks

static void on_request_arrival(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    if (ctx->is_handshake) {
        ctx->resp_bytes = RESP_RDMA_BYTES;
    } else {
        ctx->resp_bytes = RESP_UD_BYTES;
    }
    add_flow_record(ctx->op_index, ctx->client_id, ctx->worker_id, ctx->slot,
                   ctx->req_bytes, 0, ctx->start_time, g_event_queue->get_current_time(),
                   ctx->is_handshake ? "c2s_handshake" : "c2s_get");
    EventTime when = g_event_queue->get_current_time();
    g_event_queue->schedule_event(when, (void (*)(void*)) &worker_recv, ctx);
}

static void worker_recv(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    EventTime when = g_event_queue->get_current_time();
    
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
    }
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

static void client_recv_ud(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log with ACTUAL simulation timestamp (includes ML prediction + server delay + scheduling)
    // This matches testbed's full application-level FCT measurement
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
    // Note: grouped_flows still uses actual timestamps (for app completion time calculation)
    add_flow_record(ctx->op_index, ctx->client_id, ctx->worker_id, ctx->slot,
                   0, ctx->resp_bytes, ctx->server_send_time, g_event_queue->get_current_time(),
                   "s2c_ud");
    // Schedule handshake after HANDSHAKE_DELAY_NS
    ctx->is_handshake = true;
    ctx->req_bytes = HANDSHAKE_BYTES;
    EventTime when = g_event_queue->get_current_time() + HANDSHAKE_DELAY_NS;
    g_event_queue->schedule_event(when, (void (*)(void*)) &client_send_handshake, ctx);
}

static void client_send_handshake(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    ctx->start_time = g_event_queue->get_current_time();
    ctx->handshake_send_time = ctx->start_time;
    
    // Log with ACTUAL simulation timestamp (when handshake actually starts)
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

static void client_recv_rdma_finalize(void* arg) {
    auto* ctx = static_cast<FlowCtx*>(arg);
    // Log with ACTUAL simulation timestamps (includes all delays for fair comparison with testbed)
    uint64_t now_ns = (uint64_t)g_event_queue->get_current_time();
    uint64_t dur_ns = (ctx->handshake_send_time <= now_ns) ? (now_ns - (uint64_t)ctx->handshake_send_time) : 0;
    
    if (ctx->client_id >= 0 && ctx->client_id < (int)g_client_logs.size() && g_client_logs[ctx->client_id].is_open()) {
        // Log with actual simulation timestamps (same as NS3/FlowSim)
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
    // Note: grouped_flows still uses actual timestamps (for app completion time calculation)
    add_flow_record(ctx->op_index, ctx->client_id, ctx->worker_id, ctx->slot,
                   0, ctx->resp_bytes, ctx->server_send_time, g_event_queue->get_current_time(),
                   "s2c_rdma");
    
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
    ctx->req_send_time = g_event_queue->get_current_time();  // Save for UD FCT calculation
    ctx->ud_predicted_fct_ns = 0;    // Will be set by UD phase ML prediction
    ctx->rdma_predicted_fct_ns = 0;  // Will be set by RDMA phase ML prediction

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
    // Update counters BEFORE ML prediction so serialization delay sees correct inflight count
    g_total_sent_per_client[client_id]++;
    g_inflight_per_client[client_id]++;
    // Use M4 ML pipeline to predict completion time and schedule callback
    ml_predict_and_schedule_herd(ctx->req_bytes, (void (*)(void*)) &on_request_arrival, ctx);
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
    int topology_mode = 12;
    if (argc >= 4) {
        int topo_arg = std::atoi(argv[3]);
        if (topo_arg == 1 || topo_arg == 4 || topo_arg == 12) {
            topology_mode = topo_arg;
        } else {
            std::cerr << "[warn] Unsupported topology argument " << topo_arg
                      << "; defaulting to 12-node topology.\n";
        }
    }
    
    // GPU ID argument (argv[4]) - for multi-GPU systems
    if (argc >= 5) {
        int gpu_id = std::atoi(argv[4]);
        if (gpu_id >= 0 && gpu_id < torch::cuda::device_count()) {
            device = torch::Device(torch::kCUDA, gpu_id);
            std::cout << "[INFO] Using GPU " << gpu_id << std::endl;
        } else {
            std::cerr << "[WARN] Invalid GPU ID " << gpu_id 
                      << " (available: 0-" << torch::cuda::device_count() - 1 
                      << "). Using GPU 0." << std::endl;
            device = torch::Device(torch::kCUDA, 0);
        }
    }
    
    // Model directory argument (argv[5]) - for testing different models
    std::string model_dir = "checkpoints";  // Default
    if (argc >= 6) {
        model_dir = argv[5];
        std::cout << "[INFO] Using model directory: " << model_dir << std::endl;
    }
    // Run ML+HERD simulation (original condition was argc >= 2)
    // Now we accept up to 6 arguments: window, rdma, topology, gpu_id, model_dir
    if (argc >= 2) {
        g_event_queue = std::make_shared<EventQueue>();
        Topology::set_event_queue(g_event_queue);

        // Topology switch: single client/server or multi-client tree (matching flowsim)
        bool multi_client_topo = (topology_mode != 1); // use multi-client topologies when >1 endpoint
        bool twelve_node_topo = (topology_mode == 12); // true: 12-endpoint tree, false: original 4-endpoint
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
            // 12-endpoint three-tier topology from topology.md
            // Node indexing:
            // - 0..11  : serv0..serv11 (end-hosts; serv1 will act as the server)
            // - 12..17 : tor12..tor17 (ToR switches)
            // - 18..19 : agg18..agg19 (Aggregation switches)
            // - 20     : core20       (Core switch)
            // All links are 10 Gbps with ~800 ns latency.
            g_topology = std::make_shared<Topology>(21, 4);

            // Build physical links and an adjacency list for shortest-path routing.
            const int N = 21;
            std::vector<std::vector<int>> adj(N);
            auto connect_bidir = [&](int u, int v, double bw, float lat){
                g_topology->connect(u, v, bw, lat, true);
                adj[u].push_back(v);
                adj[v].push_back(u);
            };

            // Access layer: servX <-> torY (per topology.md table)
            // All links use 1000ns (1μs) delay to match FlowSim and NS3
            connect_bidir(0, 12, bw_bpns, 1000.0f);  // serv0  <-> tor12
            connect_bidir(1, 12, bw_bpns, 1000.0f);  // serv1  <-> tor12 (SERVER)
            connect_bidir(2, 13, bw_bpns, 1000.0f);  // serv2  <-> tor13
            connect_bidir(3, 13, bw_bpns, 1000.0f);  // serv3  <-> tor13
            connect_bidir(4, 14, bw_bpns, 1000.0f);  // serv4  <-> tor14
            connect_bidir(5, 14, bw_bpns, 1000.0f);  // serv5  <-> tor14
            connect_bidir(6, 15, bw_bpns, 1000.0f);  // serv6  <-> tor15
            connect_bidir(7, 15, bw_bpns, 1000.0f);  // serv7  <-> tor15
            connect_bidir(8, 16, bw_bpns, 1000.0f);  // serv8  <-> tor16
            connect_bidir(9, 16, bw_bpns, 1000.0f);  // serv9  <-> tor16
            connect_bidir(10, 17, bw_bpns, 1000.0f); // serv10 <-> tor17
            connect_bidir(11, 17, bw_bpns, 1000.0f); // serv11 <-> tor17

            // ToR -> Aggregation layer
            connect_bidir(12, 18, bw_bpns, 1000.0f); // tor12 <-> agg18
            connect_bidir(13, 18, bw_bpns, 1000.0f); // tor13 <-> agg18
            connect_bidir(14, 18, bw_bpns, 1000.0f); // tor14 <-> agg18
            connect_bidir(15, 19, bw_bpns, 1000.0f); // tor15 <-> agg19
            connect_bidir(16, 19, bw_bpns, 1000.0f); // tor16 <-> agg19
            connect_bidir(17, 19, bw_bpns, 1000.0f); // tor17 <-> agg19

            // Aggregation -> Core layer
            connect_bidir(18, 20, bw_bpns, 1000.0f); // agg18 <-> core20
            connect_bidir(19, 20, bw_bpns, 1000.0f); // agg19 <-> core20

            // Client set excludes the server node (serv1). Map client indices 0..10
            // to physical endpoints [0,2,3,4,5,6,7,8,9,10,11].
            std::vector<int> client_nodes = {0,2,3,4,5,6,7,8,9,10,11};
            NUM_CLIENTS = (int)client_nodes.size();
            g_clients.assign(NUM_CLIENTS, ClientState());
            g_client_logs.resize(NUM_CLIENTS);
            for (int i = 0; i < NUM_CLIENTS; i++) {
                g_clients[i].id = i;
                g_clients[i].key_perm = herd_get_random_permutation(HERD_NUM_KEYS, i, &g_clients[i].seed);
            }
            g_routes_c2s.resize(NUM_CLIENTS);
            g_routes_s2c.resize(NUM_CLIENTS);

            // Pre-allocate container for per-client forward paths (client -> server)
            std::vector<std::vector<int>> c2s_paths(NUM_CLIENTS);

            // Shortest path (BFS) between each client endpoint and the server endpoint (serv1).
            auto shortest_path = [&](int src, int dst){
                std::vector<int> parent(N, -1);
                std::vector<char> vis(N, 0);
                std::queue<int> q; q.push(src); vis[src] = 1;
                while(!q.empty()){
                    int u = q.front(); q.pop();
                    if (u == dst) break;
                    for (int v : adj[u]) if (!vis[v]) { vis[v]=1; parent[v]=u; q.push(v);}        
                }
                std::vector<int> path; if (!vis[dst]) return path; // disconnected (shouldn't happen)
                for (int v = dst; v != -1; v = parent[v]) path.push_back(v);
                std::reverse(path.begin(), path.end());
                return path;
            };

            auto build_route = [&](const std::vector<int>& path){ Route r; for(int id: path) r.push_back(g_topology->get_device(id)); return r; };
            const int server_node = 1; // serv1 is the server
            for (int cid = 0; cid < NUM_CLIENTS; cid++) {
                int src_node = client_nodes[cid];
                auto p = shortest_path(src_node, server_node);
                c2s_paths[cid] = p;
                g_routes_c2s[cid] = build_route(p);  // client -> server
                std::reverse(p.begin(), p.end());
                g_routes_s2c[cid] = build_route(p);  // server -> client
            }
            for (int i = 0; i < NUM_CLIENTS; i++) {
                g_client_logs[i].open((std::string("client_") + std::to_string(i) + ".log").c_str());
            }

            // Build link index mapping from all unique undirected edges in routes
            g_link_index_map.clear();
            auto key_of = [](int a, int b) -> long long {
                int u = std::min(a, b);
                int v = std::max(a, b);
                return (static_cast<long long>(u) << 32) | static_cast<unsigned long long>(v);
            };
            int next_link_idx = 0;
            auto add_path_edges = [&](const std::vector<int>& path) {
                for (size_t i = 1; i < path.size(); i++) {
                    long long key = key_of(path[i - 1], path[i]);
                    if (g_link_index_map.find(key) == g_link_index_map.end()) {
                        g_link_index_map[key] = next_link_idx++;
                    }
                }
            };
            for (const auto& p : c2s_paths) {
                add_path_edges(p);
                auto rev = p; std::reverse(rev.begin(), rev.end()); add_path_edges(rev);
            }

            // Precompute per-client link index sequences for both directions
            g_c2s_link_indices.assign(NUM_CLIENTS, {});
            g_s2c_link_indices.assign(NUM_CLIENTS, {});
            for (int cid = 0; cid < NUM_CLIENTS; cid++) {
                const auto& path_fwd = c2s_paths[cid];
                std::vector<int> links_fwd; links_fwd.reserve(path_fwd.size());
                for (size_t i = 1; i < path_fwd.size(); i++) {
                    links_fwd.push_back(g_link_index_map[key_of(path_fwd[i - 1], path_fwd[i])]);
                }
                g_c2s_link_indices[cid] = std::move(links_fwd);
                auto rev = path_fwd; std::reverse(rev.begin(), rev.end());
                std::vector<int> links_rev; links_rev.reserve(rev.size());
                for (size_t i = 1; i < rev.size(); i++) {
                    links_rev.push_back(g_link_index_map[key_of(rev[i - 1], rev[i])]);
                }
                g_s2c_link_indices[cid] = std::move(links_rev);
            }
            g_n_links_override = (int)g_link_index_map.size();

            // Debug print: precalculated path map and per-client link indices
            std::cout << "[INFO] Precalculated link index map (total " << g_n_links_override << ")\n";
            for (const auto &kv : g_link_index_map) {
                long long key = kv.first;
                int idx = kv.second;
                int u = (int)(key >> 32);
                int v = (int)(key & 0xffffffff);
                std::cout << "  link_idx=" << idx << " maps to (" << u << "," << v << ")\n";
            }
            std::cout << "[INFO] Per-client link index sequences (client -> server and server -> client)\n";
            for (int cid = 0; cid < NUM_CLIENTS; cid++) {
                std::cout << "  client " << cid << " c2s: [";
                for (size_t i = 0; i < g_c2s_link_indices[cid].size(); i++) {
                    if (i) std::cout << ",";
                    std::cout << g_c2s_link_indices[cid][i];
                }
                std::cout << "]  s2c: [";
                for (size_t i = 0; i < g_s2c_link_indices[cid].size(); i++) {
                    if (i) std::cout << ",";
                    std::cout << g_s2c_link_indices[cid][i];
                }
                std::cout << "]\n";
            }
        } else {
            // Original 4-endpoint topology: 1 server + 3 clients (8 devices total with switches)
            // Server A (0) -> S (1) -> Root R (2)
            // Client B (3) -> S1 (7) -> R (2)
            // Client C (4) -> S2 (5) -> R (2)
            // Client D (6) -> S2 (5) -> R (2)
            g_topology = std::make_shared<Topology>(8, 4); // 8 devices total
            // Links with 1000ns (1μs) latency and 10B/ns bandwidth (matching FlowSim & NS3)
            g_topology->connect(0, 1, bw_bpns, 1000.0, true); // A<->S
            g_topology->connect(1, 2, bw_bpns, 1000.0, true); // S<->R
            g_topology->connect(3, 7, bw_bpns, 1000.0, true); // B<->S1
            g_topology->connect(4, 5, bw_bpns, 1000.0, true); // C<->S2
            g_topology->connect(5, 2, bw_bpns, 1000.0, true); // S2<->R
            g_topology->connect(6, 5, bw_bpns, 1000.0, true); // D<->S2
            g_topology->connect(7, 2, bw_bpns, 1000.0, true); // S1<->R
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
            // Hardcoded config values from test_config_testbed.yaml
            int32_t hidden_size = 200;  // model.hidden_size
            int32_t n_links = 100;      // dataset.n_links_max
            
            if (g_n_links_override > 0) {
                n_links = g_n_links_override;
                std::cout << "[INFO] Overriding n_links with computed value from routes: " << n_links << "\n";
            }

            // Load ML models
            setup_m4(device, model_dir);
            
            // Initialize ML state tensors for HERD flows
            int max_herd_flows = 4 * 650 * NUM_CLIENTS; // 4 * 650 operations (request, UD, handshake, RDMA per op)
            setup_m4_tensors_for_herd(device, max_herd_flows, n_links, hidden_size);
            std::cout << "M4 ML models and tensors loaded successfully for HERD prediction service\n";
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Could not load ML models: " << e.what() << std::endl;
            return 1;
        }

        // Initialize per-client limits and state - copied exactly from flowsim
        const int default_ops = 650 * NUM_CLIENTS;
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

