#include "npy.hpp"
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>
#include <ryml_std.hpp>
#include <ryml.hpp>


// flowsim parameters
std::vector<int64_t> fat;
std::vector<int64_t> fsize;
std::vector<int64_t> fct;
std::vector<int64_t> fct_i;
std::vector<double> params;
std::unordered_map<int, int64_t> fct_map;
uint64_t limit;

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
int gpu_id = 3;
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
torch::Tensor fct_tensor;
torch::Tensor i_fct_tensor;
torch::Tensor sldn_tensor;
torch::Tensor params_tensor;

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
int n_flows_active;
int n_flows_completed;
float time_clock;
int completed_flow_id;
int min_idx;

float flow_arrival_time;
float flow_completion_time;

void setup_m4(torch::Device device) {
    if (!torch::cuda::is_available()) {
        std::cerr << "[ERROR] CUDA is not available!" << std::endl;
        return;
    }

    //torch::Device device(torch::kCUDA, gpu_id);
    // torch::Device device(torch::kCPU);

    // Disable gradient calculations
    torch::NoGradGuard no_grad;

    // Load models

    static bool models_loaded = false;
    if (!models_loaded) {
        //const std::string model_dir = "/data1/lichenni/projects/per-flow-sim/flowsim/new_model/";
        const std::string model_dir = "/data1/lichenni/projects/per-flow-sim/flowsim/model_no_flowsim/";
        try {
            lstmcell_time = torch::jit::load(model_dir + "lstmcell_time.pt", device);
            lstmcell_rate = torch::jit::load(model_dir + "lstmcell_rate.pt", device);
            lstmcell_rate_link = torch::jit::load(model_dir + "lstmcell_rate_link.pt", device);
            lstmcell_time_link = torch::jit::load(model_dir + "lstmcell_time_link.pt", device);
            output_layer = torch::jit::load(model_dir + "output_layer.pt", device);
            gnn_layer_0 = torch::jit::load(model_dir + "gnn_layer_0.pt", device);
            gnn_layer_1 = torch::jit::load(model_dir + "gnn_layer_1.pt", device);
            gnn_layer_2 = torch::jit::load(model_dir + "gnn_layer_2.pt", device);
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

        //models_loaded = true;
    }
}

void setup_m4_tensors(torch::Device device, int32_t n_edges, int32_t n_links, int32_t h_vec_dim) {
    // Define tensor options
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool);
    auto options_double = torch::TensorOptions().dtype(torch::kFloat64);
    //n_flows = fsize.size();

    // Clone tensors to ensure ownership
    //auto size_tensor = torch::from_blob(size, {n_flows}, options_float).to(device);
    size_tensor = torch::from_blob(fsize.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    size_tensor = torch::log2(size_tensor / 1000.0f + 1.0f);

    //fat_tensor = torch::from_blob(fat, {n_flows}, options_float).to(device);
    fat_tensor = torch::from_blob(fat.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    //i_fct_tensor = torch::from_blob(i_fct, {n_flows}, options_float).to(device);
    i_fct_tensor = torch::from_blob(fct_i.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    //fct_tensor = torch::from_blob(fct, {n_flows}, options_float).to(device);
    fct_tensor = torch::from_blob(fct.data(), {n_flows}, options_int64).to(torch::kFloat32).to(device);
    sldn_tensor = torch::div(fct_tensor, i_fct_tensor);
    params_tensor = torch::from_blob(params.data(), {13}, options_double).to(torch::kFloat32).to(device);

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



    // Initialize result tensors
    res_fct_tensor = torch::zeros({n_flows, 2}, options_float).to(device); //torch::from_blob(res_fct.data(), {n_flows, 2}, options_float).to(device);
    res_sldn_tensor = torch::zeros({n_flows, 2}, options_float).to(device); //torch::from_blob(res_sldn.data(), {n_flows, 2}, options_float).to(device);

    // Initialize counters
    flow_id_in_prop = 0;
    n_flows_active = 0;
    n_flows_completed = 0;
    time_clock = 0.0f;
    completed_flow_id = -1; // Initialize with invalid ID
    min_idx = -1;

    ones_cache = torch::ones({n_links}, options_int32).to(device);
}

void update_times_m4() {
    torch::NoGradGuard no_grad;
    // Determine next flow arrival and completion times
    flow_arrival_time = (flow_id_in_prop < n_flows) ? fat_tensor[flow_id_in_prop].item<float>() : std::numeric_limits<float>::infinity();
    flow_completion_time = std::numeric_limits<float>::infinity();

    if (n_flows_active > 0) {
        // Get indices of active flows
        auto flowid_active_indices = torch::nonzero(flowid_active_mask).flatten();
        auto h_vec_active = h_vec.index_select(0, flowid_active_indices);

        auto size_cur = size_tensor.index_select(0, flowid_active_indices).unsqueeze(1); // [n_active,1]
        auto nlinks_cur = flowid_to_nlinks_tensor.index_select(0, flowid_active_indices).unsqueeze(1); // [n_active,1]
        auto params_data_cur = params_tensor.repeat({n_flows_active, 1});
        //auto input_tensor = torch::cat({size_cur, sldn_flowsim_cur, nlinks_cur, params_data_cur, h_vec_active}, 1); // [n_active, 3 + h_vec_dim]
        auto input_tensor = torch::cat({nlinks_cur, params_data_cur, h_vec_active}, 1);

        // Perform inference
        sldn_est = output_layer.forward({ input_tensor }).toTensor().view(-1);; // [n_active]

        sldn_est = torch::clamp(sldn_est, 1.0f, std::numeric_limits<float>::infinity());

        auto fct_stamp_est = fat_tensor.index_select(0, flowid_active_indices) + sldn_est * i_fct_tensor.index_select(0, flowid_active_indices); // [n_active]

        // Find the flow with the minimum estimated completion time
        min_idx = torch::argmin(fct_stamp_est).item<int>();
        flow_completion_time = fct_stamp_est[min_idx].item<float>();
        completed_flow_id = flowid_active_indices[min_idx].item<int>();

        //std::cout << "calc fct " << completed_flow_id << " " << flow_completion_time << " " << min_idx << "\n";
    }
}


void step_m4() {
    torch::NoGradGuard no_grad;

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
    
    // Decide whether the next event is a flow arrival or completion
    if (flow_arrival_time < flow_completion_time) {
        // New flow arrives before the next completion
        time_clock = flow_arrival_time;

        flowid_active_mask[flow_id_in_prop] = true;
        
        time_last[flow_id_in_prop] = time_clock;

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
        n_flows_active += 1;
        flow_id_in_prop += 1;
    }
    else {
        // Flow completes before the next arrival
        time_clock = flow_completion_time;
        // Actual FCT and SLDN
        res_fct_tensor[completed_flow_id][0] = flow_completion_time - fat_tensor[completed_flow_id];
        res_fct_tensor[completed_flow_id][1] = fct[completed_flow_id];
        res_sldn_tensor[completed_flow_id][0] = sldn_est[min_idx];
        res_sldn_tensor[completed_flow_id][1] = sldn_tensor[completed_flow_id];
        // Update active flow mask to mark the flow as completed
        flowid_active_mask[completed_flow_id] = false;

        // Decrement the count of active flows and increment completed flows
        n_flows_active--;
        n_flows_completed++;
        std::cout << "m4: flow completed " << completed_flow_id << "\n";

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
        std::cout << slice.size(0) << " " << slice.size(1) << "\n";
        std::cout << reset_values.size(0) << " " << reset_values.size(1) << "\n";
        
        z_t_link.index_put_({no_flow_links_tensor, torch::indexing::Slice()}, reset_values);
        z_t_link.index_put_({no_flow_links_tensor, 1}, ones);
        z_t_link.index_put_({no_flow_links_tensor, 2}, ones);
        /*
        auto z_values = torch::stack({
             torch::zeros({no_flow_links_tensor.size(0)}, options_float),
             //torch::ones({no_flow_links_tensor.size(0)}, options_float),
             //torch::ones({no_flow_links_tensor.size(0)}, options_float),
        }, 1);
        //.to(device);

        // Assign the new values to 'z_t_link' in bulk
        std::cout << "cat " << no_flow_links_tensor.size(0) << "\n";
        std::cout << no_flow_links_tensor[0].item<int64_t>() << " " << no_flow_links_tensor[1].item<int64_t>() << "\n";
        std::cout << z_t_link.size(0) << " " << z_t_link.size(1) << "\n";
        //z_t_link.index_put_({no_flow_links_tensor, torch::indexing::Slice(), torch::indexing::Slice()}, z_values);
        z_t_link.index_put_({no_flow_links_tensor, torch::indexing::Slice()}, z_values);
        */
    }
    // Update h_vec for active flows
    auto flowid_active_mask_cur = torch::logical_and(flowid_active_mask, flow_to_graph_id == graph_id_cur);
    auto flowid_active_list_cur = torch::nonzero(flowid_active_mask_cur).flatten();
    std::cout << "actual var: " << n_flows_active << ", n_active_flows: "<<flowid_active_list_cur.numel()<< ", graph_id_cur: " << graph_id_cur<< ", fat: " << flow_arrival_time/1000.0 << ", fct: " << flow_completion_time/1000.0 << std::endl;
    std::cout <<"m4: " << flow_to_graph_id[0].item<int32_t>() << "\n";
    if (flowid_active_list_cur.numel() > 0) {
        
        // Calculate time deltas for active flows
        auto time_deltas = (time_clock - time_last.index_select(0, flowid_active_list_cur).squeeze()).view({-1, 1});

        // Create a mask for the edges corresponding to the active flows
        // auto edge_mask = torch::isin(edge_index[0], flowid_active_list_cur);
        auto edge_mask = torch::isin(edge_index[0], flowid_active_list_cur);
        auto selected_indices = edge_mask.nonzero().squeeze();
        auto edge_index_cur = edge_index.index_select(1, selected_indices);

        // Determine the number of active flows
        auto n_flows_active_cur = flowid_active_list_cur.size(0);
        auto new_flow_indices=torch::searchsorted(flowid_active_list_cur,edge_index_cur[0]);

        // Extract return_inverse from the tuple (index 1 of the tuple)
        auto unique_result_tuple = torch::_unique(edge_index_cur[1], true, true);
        auto active_link_idx = std::get<0>(unique_result_tuple); //.to(torch::kInt64);  // Unique link IDs
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
    const std::string scenario_path = argv[1];
    const std::string fat_path = scenario_path + "/fat.npy";
    const std::string fsize_path = scenario_path + "/fsize.npy";
    const std::string topo_path = scenario_path + "/topology.txt";
    const std::string routing_path = scenario_path + "/flow_to_path.txt";
    const std::string fct_path = scenario_path + "/fct_topology_flows.npy";
    const std::string fct_i_path = scenario_path + "/fct_i_topology_flows.npy";
    const std::string flow_link_path = scenario_path + "/flow_to_links.txt";
    const std::string config_path = argv[2];
    const std::string param_path = scenario_path + "/param_topology_flows.npy";
    const std::string write_path = argv[3];

    std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();
 
    npy::npy_data d_fat = npy::read_npy<int64_t>(fat_path);
    std::vector<int64_t> arrival_times = d_fat.data;

    npy::npy_data d_fsize = npy::read_npy<int64_t>(fsize_path);
    std::vector<int64_t> flow_sizes = d_fsize.data;

    limit = arrival_times.size();
    n_flows = arrival_times.size();

    for (int i = 0; i < arrival_times.size() & i < limit; i++) {
        int64_t flow_size = flow_sizes.at(i);
        fat.push_back(arrival_times.at(i));
        fsize.push_back(flow_size);
    }

    npy::npy_data d_fct = npy::read_npy<int64_t>(fct_path);
    fct = d_fct.data;

    npy::npy_data d_fct_i = npy::read_npy<int64_t>(fct_i_path);
    fct_i = d_fct_i.data;

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

    setup_m4(device);
    setup_m4_tensors(device, n_edges, n_links, hidden_size);

    int flow_index = 0;
    int flows_completed = 0;
    while (flow_index < n_flows || flows_completed < n_flows) {
        update_times_m4();
        if (flow_arrival_time < flow_completion_time) {
            flow_index++;
        } else {
            flows_completed++;
        }
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

    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();    
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start).count() << " seconds\n";
}

