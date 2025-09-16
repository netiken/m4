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

// CityHash + MICA (as in flowsim HERD path)
#include "rdma_bench/mica/mica.h"
#include <iomanip>


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
        const std::string model_dir = "/home/zabreyko/m4/inference/model";
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
    size_tensor = torch::log2(size_tensor / 1000.0f + 1.0f);

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
        // Update active flow mask to mark the flow as completed
        flowid_active_mask[completed_flow_id] = false;

        // Decrement the count of active flows and increment completed flows
        n_flows_active--;
        n_flows_completed++;
        flow_counts[get_tor(completed_flow_id)] -= 1;
        if (!tor_queue[get_tor(completed_flow_id)].empty()) {
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


// ======================== HERD-like simulation bits (mirrors flowsim) ========================
static std::shared_ptr<EventQueue> herd_event_queue;
static std::shared_ptr<Topology> herd_topology;

static constexpr int HERD_NUM_KEYS = (8 * 1024 * 1024);
static constexpr int NUM_WORKERS = 12;
static int HERD_NUM_CLIENTS = 1;
static constexpr int WINDOW_SIZE = 16;

static constexpr uint64_t RESP_UD_BYTES = 41;
static constexpr uint64_t HANDSHAKE_BYTES = 10;
static constexpr uint64_t RESP_RDMA_BYTES = 1024008;

static constexpr uint64_t SERVER_OVERHEAD_NS = 24267;
static constexpr uint64_t SEND_SPACING_NS = 2500;
static constexpr uint64_t STARTUP_DELAY_NS = 0;
static constexpr uint64_t HANDSHAKE_DELAY_NS = 8647;

static inline uint32_t hrd_fastrand(uint64_t* seed) {
    *seed = *seed * 1103515245 + 12345;
    return (uint32_t)(*seed >> 32);
}

static inline uint8_t herd_val_len_from_key_parts(uint64_t part0, uint64_t part1) {
    const uint8_t min_len = 8;
    const uint8_t max_len = MICA_MAX_VALUE;
    const uint32_t range = (uint32_t)(max_len - min_len + 1);
    uint64_t mix = part0 ^ (part1 >> 32) ^ (part1 & 0xffffffffULL);
    return (uint8_t)(min_len + (mix % range));
}

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

struct HerdClientState {
    int id;
    uint64_t seed;
    int* key_perm;
    int ws[NUM_WORKERS];
    HerdClientState() : id(0), seed(0xdeadbeef), key_perm(nullptr) { for (int i = 0; i < NUM_WORKERS; i++) ws[i] = 0; }
};

struct HerdFlowCtx {
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
    EventTime server_send_time;
    EventTime handshake_send_time;
    Route route_fwd;
    Route route_rev;
};

struct HerdFlowRecord { int op_index; int client_id; int worker_id; int slot; uint64_t req_bytes; uint64_t resp_bytes; EventTime start_ns; EventTime end_ns; uint64_t fct_ns; std::string stage; };

static std::vector<HerdClientState> herd_clients;
static std::vector<Route> herd_routes_c2s;
static std::vector<Route> herd_routes_s2c;
static std::vector<HerdFlowRecord> herd_flow_records;
static std::vector<std::ofstream> herd_client_logs; // client_*.log
static std::ofstream herd_server_log; // server.log
static std::vector<uint64_t> herd_total_sent_per_client;
static std::vector<uint64_t> herd_client_limit;
static std::vector<uint32_t> herd_inflight_per_client;
static uint64_t herd_next_op_index = 0;

static void herd_on_request_arrival(void* arg);
static void herd_worker_recv(void* arg);
static void herd_worker_send(void* arg);
static void herd_on_response_arrival(void* arg);
static void herd_client_recv_ud(void* arg);
static void herd_client_send_handshake(void* arg);
static void herd_client_recv_rdma_finalize(void* arg);
static void herd_client_start_batch(void* arg);
static void herd_add_flow_for_client(void* arg);

static void herd_on_request_arrival(void* arg) {
    auto* ctx = static_cast<HerdFlowCtx*>(arg);
    if (ctx->is_handshake) ctx->resp_bytes = RESP_RDMA_BYTES; else ctx->resp_bytes = RESP_UD_BYTES;
    {
        HerdFlowRecord rec{ctx->op_index, ctx->client_id, ctx->worker_id, ctx->slot, ctx->req_bytes, 0, ctx->start_time, herd_event_queue->get_current_time(), (uint64_t)(herd_event_queue->get_current_time() - ctx->start_time), ctx->is_handshake ? "c2s_handshake" : "c2s_get"};
        herd_flow_records.push_back(rec);
    }
    herd_event_queue->schedule_completion(herd_event_queue->get_current_time(), (void (*)(void*)) &herd_worker_recv, ctx);
}

static void herd_worker_recv(void* arg) {
    auto* ctx = static_cast<HerdFlowCtx*>(arg);
    EventTime when;
    if (!ctx->is_handshake) {
        herd_server_log << "event=reqq_recv ts_ns=" << herd_event_queue->get_current_time() << " id=" << ctx->op_index << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot << " size=" << ctx->req_bytes << " src=client:" << ctx->client_id << " dst=worker:" << ctx->worker_id << "\n";
        when = herd_event_queue->get_current_time() + SERVER_OVERHEAD_NS;
    } else {
        herd_server_log << "event=hand_recv ts_ns=" << herd_event_queue->get_current_time() << " id=" << ctx->op_index << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot << " size=" << ctx->req_bytes << " src=client:" << ctx->client_id << " dst=worker:" << ctx->worker_id << "\n";
        when = herd_event_queue->get_current_time();
    }
    herd_event_queue->schedule_completion(when, (void (*)(void*)) &herd_worker_send, ctx);
}

static void herd_worker_send(void* arg) {
    auto* ctx = static_cast<HerdFlowCtx*>(arg);
    if (ctx->is_handshake) {
        herd_server_log << "event=hand_conf ts_ns=" << herd_event_queue->get_current_time() << " id=" << ctx->op_index << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot << " size=" << ctx->resp_bytes << " src=worker:" << ctx->worker_id << " dst=client:" << ctx->client_id << "\n";
    } else {
        herd_server_log << "event=resp_send ts_ns=" << herd_event_queue->get_current_time() << " id=" << ctx->op_index << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot << " size=" << ctx->resp_bytes << " src=worker:" << ctx->worker_id << " dst=client:" << ctx->client_id << "\n";
    }
    ctx->server_send_time = herd_event_queue->get_current_time();
    auto resp_chunk = std::make_unique<Chunk>(ctx->op_index, ctx->resp_bytes, ctx->route_rev, (void (*)(void*)) &herd_on_response_arrival, ctx);
    herd_topology->send(std::move(resp_chunk));
}

static void herd_on_response_arrival(void* arg) {
    auto* ctx = static_cast<HerdFlowCtx*>(arg);
    EventTime when = herd_event_queue->get_current_time();
    if (ctx->is_handshake) herd_event_queue->schedule_completion(when, (void (*)(void*)) &herd_client_recv_rdma_finalize, ctx);
    else herd_event_queue->schedule_completion(when, (void (*)(void*)) &herd_client_recv_ud, ctx);
}

static void herd_client_recv_ud(void* arg) {
    auto* ctx = static_cast<HerdFlowCtx*>(arg);
    herd_client_logs[ctx->client_id] << "event=resp_recv_ud ts_ns=" << herd_event_queue->get_current_time() << " id=" << ctx->op_index << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot << " size=" << ctx->resp_bytes << " src=worker:" << ctx->worker_id << " dst=client:" << ctx->client_id << "\n";
    {
        HerdFlowRecord rec{ctx->op_index, ctx->client_id, ctx->worker_id, ctx->slot, 0, ctx->resp_bytes, ctx->server_send_time, herd_event_queue->get_current_time(), (uint64_t)(herd_event_queue->get_current_time() - ctx->server_send_time), "s2c_ud"};
        herd_flow_records.push_back(rec);
    }
    ctx->is_handshake = true;
    ctx->req_bytes = HANDSHAKE_BYTES;
    EventTime when = herd_event_queue->get_current_time() + HANDSHAKE_DELAY_NS;
    herd_event_queue->schedule_completion(when, (void (*)(void*)) &herd_client_send_handshake, ctx);
}

static void herd_client_send_handshake(void* arg) {
    auto* ctx = static_cast<HerdFlowCtx*>(arg);
    ctx->start_time = herd_event_queue->get_current_time();
    ctx->handshake_send_time = ctx->start_time;
    herd_client_logs[ctx->client_id] << "event=hand_send ts_ns=" << ctx->start_time << " id=" << ctx->op_index << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot << " size=" << ctx->req_bytes << " src=client:" << ctx->client_id << " dst=worker:" << ctx->worker_id << "\n";
    auto hs_chunk = std::make_unique<Chunk>(ctx->op_index, ctx->req_bytes, ctx->route_fwd, (void (*)(void*)) &herd_on_request_arrival, ctx);
    herd_topology->send(std::move(hs_chunk));
}

static void herd_client_recv_rdma_finalize(void* arg) {
    auto* ctx = static_cast<HerdFlowCtx*>(arg);
    uint64_t now_ns = (uint64_t)herd_event_queue->get_current_time();
    uint64_t dur_ns = (ctx->handshake_send_time <= now_ns) ? (now_ns - (uint64_t)ctx->handshake_send_time) : 0;
    herd_client_logs[ctx->client_id] << "event=resp_rdma_read ts_ns=" << now_ns << " id=" << ctx->op_index << " start_ns=" << ctx->handshake_send_time << " dur_ns=" << dur_ns << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot << " size=" << ctx->resp_bytes << " src=worker:" << ctx->worker_id << " dst=client:" << ctx->client_id << "\n";
    {
        HerdFlowRecord rec{ctx->op_index, ctx->client_id, ctx->worker_id, ctx->slot, 0, ctx->resp_bytes, ctx->server_send_time, herd_event_queue->get_current_time(), (uint64_t)(herd_event_queue->get_current_time() - ctx->server_send_time), "s2c_rdma"};
        herd_flow_records.push_back(rec);
    }
    int client_id = ctx->client_id;
    if (herd_inflight_per_client[client_id] > 0) herd_inflight_per_client[client_id]--;
    delete ctx;
    if (herd_total_sent_per_client[client_id] < herd_client_limit[client_id]) {
        int* cid = (int*)malloc(sizeof(int)); *cid = client_id;
        herd_event_queue->schedule_completion(herd_event_queue->get_current_time(), (void (*)(void*)) &herd_add_flow_for_client, cid);
    }
}

static void herd_add_flow_for_client(void* client_id_ptr) {
    int client_id = *(int*)client_id_ptr; free(client_id_ptr);
    int op_index = (int)herd_next_op_index; herd_next_op_index++;
    int wn = (int)(hrd_fastrand(&herd_clients[client_id].seed) % (uint32_t)NUM_WORKERS);
    int is_update = 0;
    int key_i = (int)(hrd_fastrand(&herd_clients[client_id].seed) % (uint32_t)HERD_NUM_KEYS);
    uint128 key128 = CityHash128((char*)&herd_clients[client_id].key_perm[key_i], 4);
    uint64_t part0 = key128.first; uint64_t part1 = key128.second;
    uint8_t vlen = herd_val_len_from_key_parts(part0, part1);
    uint64_t req_bytes = is_update ? (uint64_t)(16 + 1 + 1 + vlen) : (uint64_t)(16 + 1);
    auto* ctx = new HerdFlowCtx();
    ctx->op_index = op_index; ctx->client_id = client_id; ctx->worker_id = wn;
    ctx->slot = herd_clients[ctx->client_id].ws[wn]; ctx->is_update = (is_update != 0);
    ctx->vlen = vlen; ctx->req_bytes = req_bytes; ctx->is_handshake = false;
    ctx->route_fwd = herd_routes_c2s[ctx->client_id]; ctx->route_rev = herd_routes_s2c[ctx->client_id];
    ctx->start_time = herd_event_queue->get_current_time();
    herd_clients[ctx->client_id].ws[wn] = (herd_clients[ctx->client_id].ws[wn] + 1) % WINDOW_SIZE;
    herd_client_logs[ctx->client_id] << "event=req_send ts_ns=" << herd_event_queue->get_current_time() << " id=" << ctx->op_index << " clt=" << ctx->client_id << " wrkr=" << ctx->worker_id << " slot=" << ctx->slot << " size=" << ctx->req_bytes << " src=client:" << ctx->client_id << " dst=worker:" << ctx->worker_id << "\n";
    auto req_chunk = std::make_unique<Chunk>(ctx->op_index, ctx->req_bytes, ctx->route_fwd, (void (*)(void*)) &herd_on_request_arrival, ctx);
    herd_topology->send(std::move(req_chunk));
    herd_total_sent_per_client[client_id]++;
    herd_inflight_per_client[client_id]++;
}

static void herd_client_start_batch(void* arg) {
    int client_id = arg ? *(int*)arg : 0; if (arg) free(arg);
    EventTime base = herd_event_queue->get_current_time();
    uint32_t to_send = WINDOW_SIZE;
    for (uint32_t i = 0; i < to_send && herd_total_sent_per_client[client_id] < herd_client_limit[client_id]; i++) {
        int* cid = (int*)malloc(sizeof(int)); *cid = client_id;
        EventTime extra = (i > 0 ? STARTUP_DELAY_NS : 0);
        EventTime send_time = base + extra + (EventTime)(i * SEND_SPACING_NS);
        herd_event_queue->schedule_completion(send_time, (void (*)(void*)) &herd_add_flow_for_client, cid);
    }
}

static int herd_main() {
    herd_event_queue = std::make_shared<EventQueue>();
    Topology::set_event_queue(herd_event_queue);
    bool multi_client_topo = false;
    double bw_bpns = 10.0/8;
    if (!multi_client_topo) {
        herd_topology = std::make_shared<Topology>(2, 2);
        herd_topology->connect(0, 1, bw_bpns, 3500, true);
        HERD_NUM_CLIENTS = 1;
        herd_clients.assign(HERD_NUM_CLIENTS, HerdClientState());
        herd_client_logs.resize(HERD_NUM_CLIENTS);
        for (int i = 0; i < HERD_NUM_CLIENTS; i++) { herd_clients[i].id = i; herd_clients[i].key_perm = herd_get_random_permutation(HERD_NUM_KEYS, i, &herd_clients[i].seed); }
        herd_routes_c2s.resize(HERD_NUM_CLIENTS); herd_routes_s2c.resize(HERD_NUM_CLIENTS);
        Route c2s; c2s.push_back(herd_topology->get_device(0)); c2s.push_back(herd_topology->get_device(1));
        Route s2c; s2c.push_back(herd_topology->get_device(1)); s2c.push_back(herd_topology->get_device(0));
        herd_routes_c2s[0] = c2s; herd_routes_s2c[0] = s2c; herd_client_logs[0].open("client_0.log");
    } else {
        herd_topology = std::make_shared<Topology>(8, 4);
        herd_topology->connect(0, 1, bw_bpns, 800.0, true);
        herd_topology->connect(1, 2, bw_bpns, 800.0, true);
        herd_topology->connect(3, 7, bw_bpns, 800.0, true);
        herd_topology->connect(4, 5, bw_bpns, 800.0, true);
        herd_topology->connect(5, 2, bw_bpns, 800.0, true);
        herd_topology->connect(6, 5, bw_bpns, 800.0, true);
        herd_topology->connect(7, 2, bw_bpns, 800.0, true);
        HERD_NUM_CLIENTS = 3;
        herd_clients.assign(HERD_NUM_CLIENTS, HerdClientState());
        herd_client_logs.resize(HERD_NUM_CLIENTS);
        for (int i = 0; i < HERD_NUM_CLIENTS; i++) { herd_clients[i].id = i; herd_clients[i].key_perm = herd_get_random_permutation(HERD_NUM_KEYS, i, &herd_clients[i].seed); }
        herd_routes_c2s.resize(HERD_NUM_CLIENTS); herd_routes_s2c.resize(HERD_NUM_CLIENTS);
        auto build_route = [&](std::vector<int> path){ Route r; for(int id: path) r.push_back(herd_topology->get_device(id)); return r; };
        herd_routes_c2s[0] = build_route({3,7,2,1,0}); herd_routes_s2c[0] = build_route({0,1,2,7,3});
        herd_routes_c2s[1] = build_route({4,5,2,1,0}); herd_routes_s2c[1] = build_route({0,1,2,5,4});
        herd_routes_c2s[2] = build_route({6,5,2,1,0}); herd_routes_s2c[2] = build_route({0,1,2,5,6});
        herd_client_logs[0].open("client_0.log"); herd_client_logs[1].open("client_1.log"); herd_client_logs[2].open("client_2.log");
    }

    herd_server_log.open("server.log");
    const int default_ops = 650;
    std::vector<int64_t> herd_fat; herd_fat.clear(); herd_fat.reserve(default_ops);
    herd_total_sent_per_client.assign(HERD_NUM_CLIENTS, 0);
    herd_client_limit.assign(HERD_NUM_CLIENTS, default_ops / HERD_NUM_CLIENTS);
    herd_inflight_per_client.assign(HERD_NUM_CLIENTS, 0);
    EventTime start_time = herd_event_queue->get_current_time() + 1;
    for (int cid = 0; cid < HERD_NUM_CLIENTS; cid++) { int* arg_c = (int*)malloc(sizeof(int)); *arg_c = cid; herd_event_queue->schedule_completion(start_time, (void (*)(void*)) &herd_client_start_batch, arg_c); }
    while (!herd_event_queue->finished()) { herd_event_queue->proceed(); }
    for (int i = 0; i < (int)herd_clients.size(); i++) { if (herd_clients[i].key_perm != nullptr) { free(herd_clients[i].key_perm); herd_clients[i].key_perm = nullptr; } }
    for (auto& cl : herd_client_logs) if (cl.is_open()) cl.close(); if (herd_server_log.is_open()) herd_server_log.close();
    std::ofstream ofs("flows.txt");
    for (const auto& r : herd_flow_records) {
        ofs << r.op_index << " " << r.client_id << " " << r.worker_id << " " << r.slot << " " << r.req_bytes << " " << r.resp_bytes << " " << r.start_ns << " " << r.end_ns << " " << r.fct_ns << " " << r.stage << "\n";
    }
    std::cout << "HERD-sim completed ops: " << herd_flow_records.size() << ", wrote flows.txt\n";
    return 0;
}

int main(int argc, char *argv[]) {
    // Mirror flowsim: default to HERD-mode when insufficient args
    if (argc < 6) {
        return herd_main();
    }
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
    while (infile_routing >> num_hops) {
        int host_id;
        auto route = Route();
        infile_routing >> host_id;
        host_ids.push_back(host_id);
        route.push_back(topology->get_device(host_id));
        for (int i = 1; i < num_hops; i++) {
            infile_routing >> host_id;
            route.push_back(topology->get_device(host_id));
        }
        routing.push_back(route);
    }

    for (int i = 0; i < fat.size(); i++) {
        double prop_delay = latency * (routing.at(i).size() - 1);
        double trans_delay = (((fsize.at(i) + std::ceil(fsize.at(i) / MTU) * BYTES_PER_HEADER)) / bandwidth);
        double first_packet = (std::min(MTU, (double) fsize.at(i)) + BYTES_PER_HEADER) / bandwidth * (routing.at(i).size() - 2);
        fct_i.push_back(trans_delay + prop_delay + first_packet);
    }

    npy::npy_data d_param = npy::read_npy<double>(param_path);
    params = d_param.data;

    std::filesystem::path cwd = std::filesystem::current_path() / flow_link_path;
    std::ifstream infile(cwd);
    int num_links;
    int32_t offset = 0;
    int32_t flow_id = 0;
    while (infile >> num_links) {
        std::vector<int> hops;
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

    uint32_t n_edges = flowid_to_linkid_flat.size();

    infile.close();
    infile.open(config_path);
    std::ostringstream contents;
    contents << infile.rdbuf();
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

    int flow_index = 0;
    int flows_completed = 0;
    while (n_flows_arrived < n_flows || n_flows_completed < n_flows) {
        std::cout << "provoking " << n_flows_arrived << " " << n_flows_completed << "\n";
        update_times_m4();
        step_m4();
    }

    std::vector<float> fct_vector;
    for (int i = 0; i < res_fct_tensor.sizes()[0]; i++) {
        fct_vector.push_back(res_fct_tensor[i][0].item<float>());
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

