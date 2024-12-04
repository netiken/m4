// inference_optimized.cpp

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <chrono>

// Function definition
extern "C" int interactive_inference(
    float* size,
    float* fat,
    float* i_fct,
    float* fct,
    float* sldn_flowsim,
    int n_flows,
    float* res_fct,
    float* res_sldn,
    int n_flows_active_max,
    int n_links_active_max,
    int gpu_id,
    int h_vec_dim = 128,
    float rtt = 0.0f
) {
    try {
        if (!torch::cuda::is_available()) {
            std::cerr << "[ERROR] CUDA is not available!" << std::endl;
            return -1;
        }

        torch::Device device(torch::kCUDA, gpu_id);
        // torch::Device device(torch::kCPU);

        std::cout << "Using device: " << device << std::endl;

        // Disable gradient calculations
        torch::NoGradGuard no_grad;

        // Load models
        std::cout << "Loading models..." << std::endl;

        static torch::jit::script::Module lstmcell_time;
        static torch::jit::script::Module lstmcell_rate;
        static torch::jit::script::Module mlp_model;
        static torch::jit::script::Module gnn_layer1;
        static torch::jit::script::Module gnn_layer2;

        static bool models_loaded = false;
        if (!models_loaded) {
            lstmcell_time = torch::jit::load("../models/lstmcell_time.pt", device);
            lstmcell_rate = torch::jit::load("../models/lstmcell_rate.pt", device);
            mlp_model = torch::jit::load("../models/output_layer.pt", device);
            gnn_layer1 = torch::jit::load("../models/gnn_layer_0.pt", device);
            gnn_layer2 = torch::jit::load("../models/gnn_layer_1.pt", device);

            lstmcell_time = torch::jit::optimize_for_inference(lstmcell_time);
            lstmcell_rate = torch::jit::optimize_for_inference(lstmcell_rate);
            mlp_model = torch::jit::optimize_for_inference(mlp_model);
            gnn_layer1 = torch::jit::optimize_for_inference(gnn_layer1);
            gnn_layer2 = torch::jit::optimize_for_inference(gnn_layer2);

            lstmcell_time.eval();
            lstmcell_rate.eval();
            mlp_model.eval();
            gnn_layer1.eval();
            gnn_layer2.eval();

            models_loaded = true;
        }

        std::cout << "Models loaded successfully." << std::endl;

        // Define options before using it
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto options_int64 = torch::TensorOptions().dtype(torch::kInt64);

        // Convert input data to tensors directly on GPU
        std::cout << "Creating tensors from input data..." << std::endl;

        auto size_tensor = torch::from_blob(size, {n_flows}, options).to(device);
        size_tensor = torch::log2(size_tensor / 1000.0f + 1.0f);
        auto fat_tensor = torch::from_blob(fat, {n_flows}, options).to(device);
        auto i_fct_tensor = torch::from_blob(i_fct, {n_flows}, options).to(device);
        auto fct_tensor = torch::from_blob(fct, {n_flows}, options).to(device);
        auto sldn_tensor = torch::div(fct_tensor, i_fct_tensor);
        auto sldn_flowsim_tensor = torch::from_blob(sldn_flowsim, {n_flows}, options).to(device);
        std::cout << "Tensors created and moved to device." << std::endl;

         // initialize tensors for active flows
        auto h_vec = torch::zeros({n_flows_active_max, h_vec_dim}, options).to(device);
        auto time_deltas = torch::zeros({n_flows_active_max, 1}, options).to(device);
        auto one_hot_type_a = torch::tensor({1.0f, 0.0f}, options).to(device); // Type A (flows)
        auto one_hot_type_a_expanded_full=one_hot_type_a.expand({n_flows_active_max, -1});
        auto one_hot_type_b = torch::tensor({0.0f, 1.0f}, options).to(device); // Type B (links)
        auto one_hot_type_b_full=one_hot_type_b.expand({n_links_active_max, -1});

        auto edges_a_to_b_active_full = torch::zeros({2, n_flows_active_max * 6}, options_int64).to(device);

        // Initialize z_t_link (features for type_b nodes)
        auto z_t_link = torch::zeros({n_links_active_max, h_vec_dim}, options).to(device);
        z_t_link.index_put_(
            {torch::indexing::Slice(), 0},
            torch::ones({n_links_active_max}, options).to(device)
        );

        float time_last = 0.0f; // Initialize the current time
        float time_last_tmp=0.0f;
        int flow_id_in_prop = 0;
        int flow_id_start = 0;

        // Active flows (maintained as tensors on GPU)
        auto flowid_active_list = torch::empty({n_flows_active_max}, options_int64).to(device);
        auto flowidx_active_list = torch::empty({n_flows_active_max}, options_int64).to(device);
        int64_t active_flows_count = 0;

         // Flow metric map to store per-flow results
        auto res_fct_tensor = torch::from_blob(res_fct, {n_flows * 2}, options).to(device);
        auto res_sldn_tensor = torch::from_blob(res_sldn, {n_flows * 2}, options).to(device);

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        int64_t completed_flow_idx = -1;
        int64_t completed_flow_id = -1;
        int64_t min_idx = -1;
        float flow_arrival_time;
        float flow_completion_time;
        float time_max=std::numeric_limits<float>::infinity();
        torch::Tensor sldn_min_tensor;
        torch::Tensor fat_min_tensor;
        torch::Tensor fct_est_tensor;
        torch::Tensor flowid_active_tensor;
        torch::Tensor flowidx_active_tensor;
        torch::Tensor sldn_est;
        torch::Tensor h_vec_active;
        torch::Tensor fct_stamp_est;
        torch::Tensor time_deltas_active;
        torch::Tensor edges_a_to_b_active;
        // Main loop: Process flows until all are completed
        while (flow_id_in_prop < n_flows || active_flows_count > 0) {
            // Determine next flow arrival and completion times
            flow_arrival_time = time_max;
            flow_completion_time = time_max;

            if (flow_id_in_prop < n_flows) {
                flow_arrival_time = fat[flow_id_in_prop];
            }

            if (active_flows_count > 0) {
                // Get active flows
                flowid_active_tensor = flowid_active_list.narrow(0, 0, active_flows_count);
                flowidx_active_tensor = flowidx_active_list.narrow(0, 0, active_flows_count);

                h_vec_active = h_vec.index_select(0, flowidx_active_tensor);

                // Compute sldn_est
                sldn_est = mlp_model.forward({h_vec_active}).toTensor().select(1, 0) + 1.0f;

                // Compute estimated flow completion times
                fct_stamp_est = fat_tensor.index_select(0, flowid_active_tensor) + sldn_est * i_fct_tensor.index_select(0, flowid_active_tensor);

                // Find the flow with the minimum estimated completion time
                min_idx = torch::argmin(fct_stamp_est).item<int64_t>();

                completed_flow_idx = static_cast<int>(min_idx);
                completed_flow_id= flowid_active_tensor[min_idx].item<int64_t>();
                fct_est_tensor= fct_stamp_est[min_idx];
                flow_completion_time = fct_est_tensor.item<float>();
                sldn_min_tensor = sldn_est[min_idx];
                
                fat_min_tensor = fat_tensor[completed_flow_id];
                // sldn_min = sldn_est[min_idx].item<float>();
                // fat_min = fat[flowid_active_tensor[min_idx].item<int64_t>()];
            }

            float previous_time_last = time_last;
            if (flow_arrival_time < flow_completion_time) {
                time_last_tmp = flow_arrival_time;

                int flow_idx = flow_id_in_prop - flow_id_start;
                h_vec[flow_idx][0] = size_tensor[flow_id_in_prop];

                flowid_active_list[active_flows_count] = flow_id_in_prop;
                flowidx_active_list[active_flows_count] = flow_idx;
                active_flows_count++;
                flow_id_in_prop++;
            } else {
                time_last_tmp = flow_completion_time;

                res_fct_tensor[completed_flow_id * 2] = fct_est_tensor-fat_min_tensor;
                res_fct_tensor[completed_flow_id * 2 + 1] = fct_tensor[completed_flow_id];
                
                res_sldn_tensor[completed_flow_id * 2] = sldn_min_tensor;
                res_sldn_tensor[completed_flow_id * 2 + 1] = sldn_tensor[completed_flow_id];

                // std::cout << "Flow ID: " << completed_flow_id << ", FCT: " << fct_est_tensor.item<float>() << ", SLDN: " << sldn_min_tensor.item<float>() << std::endl;

                // Remove completed flow
                active_flows_count--;
                if (completed_flow_idx != active_flows_count) {
                    // Gather the last active flow's data
                    auto last_flow_id = flowid_active_list[active_flows_count];
                    auto last_flow_idx = flowidx_active_list[active_flows_count];
                    auto last_h_vec = h_vec[active_flows_count];
                    
                    // Assign the last active flow's data to the completed flow's position
                    flowid_active_list[completed_flow_idx] = last_flow_id;
                    flowidx_active_list[completed_flow_idx] = last_flow_idx;
                    h_vec[completed_flow_idx] = last_h_vec;
                }
            }

            if (active_flows_count == 0) {
                h_vec.zero_();
                flow_id_start = flow_id_in_prop;
            } else {
                // Update h_vec with update_rate
                flowidx_active_tensor = flowidx_active_list.narrow(0, 0, active_flows_count);
                h_vec_active = h_vec.index_select(0, flowidx_active_tensor);

                float time_delta = time_last_tmp - previous_time_last;
                if (time_delta >= rtt) {
                    // Update time for active flows
                    time_deltas_active = time_deltas.narrow(0, 0, active_flows_count);
                    time_deltas_active.fill_(time_delta / 1000.0f);

                    // LSTM time update
                    std::vector<c10::IValue> lstm_inputs = {time_deltas_active, h_vec_active};
                    h_vec_active = lstmcell_time.forward(lstm_inputs).toTensor();
                

                    // GNN forward pass
                    auto type_a_nodes = torch::cat({one_hot_type_a_expanded_full.narrow(0, 0, active_flows_count), h_vec_active}, 1);
                    auto type_b_nodes = torch::cat({one_hot_type_b_full, z_t_link}, 1);

                    // Combine nodes into x_combined
                    auto x_combined = torch::cat({type_a_nodes, type_b_nodes}, 0);

                    // Number of type_a nodes
                    int64_t num_type_a = h_vec_active.size(0);

                    // Prepare edges_a_to_b_active
                    for (int i=0; i<=active_flows_count; i++) {
                        edges_a_to_b_active_full[0][i] = i;
                    }
                    edges_a_to_b_active = edges_a_to_b_active_full.narrow(1, 0, active_flows_count);
                    // Adjust edge indices for bidirectional edges
                    auto edges_src = edges_a_to_b_active[0];
                    auto edges_dst = edges_a_to_b_active[1];

                    // Edge indices from type_a to type_b
                    auto edge_index_a_to_b = torch::stack({edges_src, edges_dst + num_type_a}, 0);
                    auto edge_index_b_to_a = torch::stack({edges_dst + num_type_a, edges_src}, 0);
                    auto edge_index_combined = torch::cat({edge_index_a_to_b, edge_index_b_to_a}, 1);

                    // GNN forward pass
                    x_combined = gnn_layer1.forward({x_combined, edge_index_combined}).toTensor();
                    x_combined = gnn_layer2.forward({x_combined, edge_index_combined}).toTensor();

                    // LSTM cell forward pass
                    lstm_inputs = {x_combined.slice(0, 0, num_type_a), h_vec_active};
                    h_vec_active = lstmcell_rate.forward(lstm_inputs).toTensor();

                    // Update h_vec
                    h_vec.index_copy_(0, flowidx_active_tensor, h_vec_active);

                    time_last = time_last_tmp;
                }
            }
        }

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_elapsed = end_time - start_time;
        std::cout << "Time elapsed: " << time_elapsed.count() << " seconds" << std::endl;

        // Move tensors to CPU
        auto res_fct_cpu = res_fct_tensor.cpu();
        auto res_sldn_cpu = res_sldn_tensor.cpu();

        // Use memcpy for efficient bulk copying
        std::memcpy(res_fct, res_fct_cpu.data_ptr<float>(), sizeof(float) * n_flows * 2);
        std::memcpy(res_sldn, res_sldn_cpu.data_ptr<float>(), sizeof(float) * n_flows * 2);
        std::cout << "Inference completed successfully." << std::endl;

        return 0; // Success
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return -1; // Error
    }
}
