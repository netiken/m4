// interactive_inference.cpp

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <cassert>

#include <ATen/cuda/CUDAGraph.h>

// Function definition
extern "C" int interactive_inference(
    float* size,                        // [n_flows]
    float* fat,                         // [n_flows]
    float* i_fct,                       // [n_flows]
    float* fct,                         // [n_flows]
    float* sldn_flowsim,                // [n_flows]
    int* flowid_to_linkid_offsets,      // [n_flows + 1]
    int* flowid_to_linkid_flat,         // [n_edges]
    int* edges_flow_ids,                // [n_edges]
    int* edges_link_ids,                // [n_edges]
    int n_links,                        // Total number of links
    int n_flows,                        // Number of flows
    int n_edges,                        // Number of edges
    float* res_fct,                     // Output: [n_flows * 2]
    float* res_sldn,                    // Output: [n_flows * 2]
    int gpu_id,                         // GPU device ID
    int h_vec_dim,                      // Hidden vector dimension
    float rtt,                          // Round-trip time (unused)
    bool enable_flowsim                  // Enable flowsim_diff flag
) {
    try {
        if (!torch::cuda::is_available()) {
            std::cerr << "[ERROR] CUDA is not available!" << std::endl;
            return -1;
        }

        torch::Device device(torch::kCUDA, gpu_id);
        // torch::Device device(torch::kCPU);

        // Disable gradient calculations
        torch::NoGradGuard no_grad;

        // Load models
        static torch::jit::script::Module lstmcell_time;
        static torch::jit::script::Module lstmcell_rate;
        static torch::jit::script::Module output_layer;
        static torch::jit::script::Module gnn_layer_0;
        static torch::jit::script::Module gnn_layer_1;

        static bool models_loaded = false;
        if (!models_loaded) {
            const std::string model_dir = "../models_topo/"; // Consider making this configurable
            try {
                lstmcell_time = torch::jit::load(model_dir + "lstmcell_time.pt", device);
                lstmcell_rate = torch::jit::load(model_dir + "lstmcell_rate.pt", device);
                output_layer = torch::jit::load(model_dir + "output_layer.pt", device);
                gnn_layer_0 = torch::jit::load(model_dir + "gnn_layer_0.pt", device);
                gnn_layer_1 = torch::jit::load(model_dir + "gnn_layer_1.pt", device);
            }
            catch (const c10::Error& e) {
                std::cerr << "[ERROR] Failed to load one or more models: " << e.what() << std::endl;
                return -1;
            }

            // Set models to evaluation mode
            lstmcell_time.eval();
            lstmcell_rate.eval();
            output_layer.eval();
            gnn_layer_0.eval();
            gnn_layer_1.eval();

            // Optimize models for inference
            lstmcell_time = torch::jit::optimize_for_inference(lstmcell_time);
            lstmcell_rate = torch::jit::optimize_for_inference(lstmcell_rate);
            output_layer = torch::jit::optimize_for_inference(output_layer);
            gnn_layer_0 = torch::jit::optimize_for_inference(gnn_layer_0);
            gnn_layer_1 = torch::jit::optimize_for_inference(gnn_layer_1);

            models_loaded = true;
        }

        // Define tensor options
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32);
        auto options_int32 = torch::TensorOptions().dtype(torch::kInt32);
        auto options_bool = torch::TensorOptions().dtype(torch::kBool);

        // Clone tensors to ensure ownership
        auto size_tensor = torch::from_blob(size, {n_flows}, options_float).to(device);
        size_tensor = torch::log2(size_tensor / 1000.0f + 1.0f);

        auto fat_tensor = torch::from_blob(fat, {n_flows}, options_float).to(device);
        auto i_fct_tensor = torch::from_blob(i_fct, {n_flows}, options_float).to(device);
        auto fct_tensor = torch::from_blob(fct, {n_flows}, options_float).to(device);
        auto sldn_tensor = torch::div(fct_tensor, i_fct_tensor);
        auto sldn_flowsim_tensor = torch::from_blob(sldn_flowsim, {n_flows}, options_float).to(device);

        // Convert flowid_to_linkid to tensors
        auto flowid_to_linkid_flat_tensor = torch::from_blob(flowid_to_linkid_flat, {n_edges}, options_int32).to(device);
        auto flowid_to_linkid_offsets_tensor = torch::from_blob(flowid_to_linkid_offsets, {n_flows + 1}, options_int32).to(device);
        auto flowid_to_nlinks_tensor = flowid_to_linkid_offsets_tensor.slice(0, 1, n_flows+1) - flowid_to_linkid_offsets_tensor.slice(0, 0, n_flows);
        
        // Convert edges_flow_ids and edges_link_ids to tensors
        auto edges_flow_ids_tensor = torch::from_blob(edges_flow_ids, {n_edges}, options_int32).to(device);
        auto edges_link_ids_tensor = torch::from_blob(edges_link_ids, {n_edges}, options_int32).to(device);

        // Construct edge_index tensor [2, 2 * n_edges] for bidirectional edges
        auto edge_index = torch::stack({edges_flow_ids_tensor, edges_link_ids_tensor}, 0); // [2, n_edges]

        // Initialize tensors for active flows
        auto h_vec = torch::zeros({n_flows, h_vec_dim}, options_float).to(device);
        h_vec.index_put_({torch::arange(n_flows, device=device), 0}, 1.0f);
        h_vec.index_put_({torch::arange(n_flows, device=device), 2}, size_tensor);
        h_vec.index_put_({torch::arange(n_flows, device=device), 3}, flowid_to_nlinks_tensor.to(options_float));

        // Initialize z_t_link as in Python
        auto z_t_link = torch::zeros({n_links, h_vec_dim}, options_float).to(device); // [n_links, h_vec_dim]
        z_t_link.index_put_({torch::arange(n_links, device=device), 1}, 1.0f);
        z_t_link.index_put_({torch::arange(n_links, device=device), 2}, 1.0f);

        // Initialize graph management tensors
        auto link_to_graph_id = -torch::ones({n_links}, options_int32).to(device);
        auto link_to_nflows = torch::zeros({n_links}, options_int32).to(device);
        auto flow_to_graph_id = -torch::ones({n_flows}, options_int32).to(device);

        int graph_id_counter = 0;
        int graph_id_cur = 0;

        // Initialize time_last and flowid_active_mask
        auto time_last = torch::zeros({n_flows}, options_float).to(device);
        auto flowid_active_mask = torch::zeros({n_flows}, options_bool).to(device);

        // Initialize result tensors
        auto res_fct_tensor = torch::from_blob(res_fct, {n_flows, 2}, options_float).to(device);
        auto res_sldn_tensor = torch::from_blob(res_sldn, {n_flows, 2}, options_float).to(device);

        // Initialize counters
        int flow_id_in_prop = 0;
        int n_flows_active = 0;
        int n_flows_completed = 0;
        float time_clock = 0.0f;
        int completed_flow_id = -1; // Initialize with invalid ID
        int min_idx = -1;
        torch::Tensor sldn_est;

        static torch::Tensor ones_cache = torch::ones({n_links}, options_int32).to(device);
        auto time_deltas_scaled = torch::empty({n_flows, 1}, options_float).to(device);

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        auto capture_stream = at::cuda::getStreamFromPool();
        at::cuda::setCurrentCUDAStream(capture_stream);
        at::cuda::CUDAGraph cuda_graph;
        bool graphCreated = false;

        auto mlp_input = torch::zeros({4000, 515}, device);
        auto x_combined_padded = torch::zeros({4000, 512}, device);
        auto edges_list_active_padded = torch::tensor({{0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 1}, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 1}}, torch::kLong).to(device);
        // warmup
        auto gnn_output_0 = gnn_layer_0.forward({x_combined_padded, edges_list_active_padded}).toTensor();
        gnn_layer_1.forward({gnn_output_0, edges_list_active_padded}).toTensor();

        auto stuff = output_layer.forward({mlp_input}).toTensor();

        std::cout << "begin capture\n";
        cuda_graph.capture_begin();
        torch::InferenceMode guard;
        std::cout << "trying the first op\n";
        stuff = output_layer.forward({mlp_input}).toTensor();
        //gnn_output_0 = gnn_layer_0.forward({x_combined_padded, edges_list_active_padded}).toTensor();
        std::cout << "trying the second op\n";
        //auto gnn_output_1_padded = gnn_layer_1.forward({gnn_output_0, edges_list_active_padded}).toTensor();
        cuda_graph.capture_end();
        std::cout << "end capture\n";
        capture_stream.synchronize();
                        //auto gnn_output_0 = gnn_layer_0.forward({x_combined_padded, edges_list_active_padded}).toTensor();
                        //std::cout << "second layer\n";
                        //gnn_output_1_padded = gnn_layer_1.forward({gnn_output_0, edges_list_active}).toTensor();
                        //std::cout << "nothing remains\n";
        
        // Main loop: Process flows until all are completed
        while (n_flows_completed < n_flows) {
            // Determine next flow arrival and completion times
            float flow_arrival_time = (flow_id_in_prop < n_flows) ? fat[flow_id_in_prop] : std::numeric_limits<float>::infinity();
            float flow_completion_time = std::numeric_limits<float>::infinity();

            if (n_flows_active > 0) {
                // Get indices of active flows
                auto flowid_active_indices = torch::nonzero(flowid_active_mask).flatten();
                auto h_vec_active = h_vec.index_select(0, flowid_active_indices);

                if (enable_flowsim) {
                    // Prepare input tensor by concatenating size and sldn_flowsim
                    auto size_cur = size_tensor.index_select(0, flowid_active_indices).unsqueeze(1); // [n_active,1]
                    auto sldn_flowsim_cur = sldn_flowsim_tensor.index_select(0, flowid_active_indices).unsqueeze(1); // [n_active,1]
                    auto nlinks_cur = flowid_to_nlinks_tensor.index_select(0, flowid_active_indices).unsqueeze(1); // [n_active,1]
                    auto input_tensor = torch::cat({size_cur, sldn_flowsim_cur, nlinks_cur, h_vec_active}, 1); // [n_active, 3 + h_vec_dim]

                    // Perform inference
                    sldn_est = output_layer.forward({ input_tensor }).toTensor().view(-1);; // [n_active]
                }
                else {
                    // Perform inference directly on h_vec
                    sldn_est = output_layer.forward({ h_vec_active }).toTensor().view(-1) + 1.0f; // [n_active]
                }
                sldn_est = torch::clamp(sldn_est, 1.0f, std::numeric_limits<float>::infinity());

                auto fct_stamp_est = fat_tensor.index_select(0, flowid_active_indices) + sldn_est * i_fct_tensor.index_select(0, flowid_active_indices); // [n_active]

                // Find the flow with the minimum estimated completion time
                min_idx = torch::argmin(fct_stamp_est).item<int>();
                flow_completion_time = fct_stamp_est[min_idx].item<float>();
                completed_flow_id = flowid_active_indices[min_idx].item<int>();
            }
            // Decide whether the next event is a flow arrival or completion
            if (flow_arrival_time < flow_completion_time) {
                // New flow arrives before the next completion
                time_clock = flow_arrival_time;

                flowid_active_mask[flow_id_in_prop] = true;
                
                time_last[flow_id_in_prop] = time_clock;

                // Assign graph IDs
                int start_idx = flowid_to_linkid_offsets[flow_id_in_prop];
                int end_idx = flowid_to_linkid_offsets[flow_id_in_prop + 1];
                auto links_tensor = flowid_to_linkid_flat_tensor.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx);

                link_to_nflows.index_add_(0, links_tensor, ones_cache.slice(/*dim=*/0, /*start=*/0, links_tensor.size(0)));

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

                // Get graph ID of the completed flow
                graph_id_cur = flow_to_graph_id[completed_flow_id].item<int64_t>();
                // Get links for this flow
                int start_idx = flowid_to_linkid_offsets[completed_flow_id];
                int end_idx = flowid_to_linkid_offsets[completed_flow_id + 1];
                auto links_tensor = flowid_to_linkid_flat_tensor.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx);

                link_to_nflows.index_add_(0, links_tensor, -ones_cache.slice(/*dim=*/0, /*start=*/0, links_tensor.size(0)));
                flow_to_graph_id.index_put_({completed_flow_id}, -1);

                // Find links with no active flows using tensor operations
                auto no_flow_mask = (link_to_nflows.index({links_tensor}) == 0);
                auto no_flow_links_tensor = links_tensor.masked_select(no_flow_mask);

                // Update link_to_graph_id and reset z_t_link for links with no active flows in bulk

                // Assign -1 to 'link_to_graph_id' for all 'no_flow_links' at once
                link_to_graph_id.index_put_({no_flow_links_tensor}, -1);

                // // Create tensors with desired reset values for 'z_t_link'
                // auto z_values = torch::stack({
                //     torch::zeros({no_flow_links_tensor.size(0)}, options_float),
                //     torch::ones({no_flow_links_tensor.size(0)}, options_float),
                //     torch::ones({no_flow_links_tensor.size(0)}, options_float)
                // }, 1).to(device);

                // // Assign the new values to 'z_t_link' in bulk
                // z_t_link.index_put_({no_flow_links_tensor, torch::indexing::Slice(), torch::indexing::Slice()}, z_values);
            }
            // Update h_vec for active flows
            auto flowid_active_mask_cur = torch::logical_and(flowid_active_mask, flow_to_graph_id == graph_id_cur);
            auto flowid_active_list_cur = torch::nonzero(flowid_active_mask_cur).flatten();
            std::cout <<"n_active_flows: "<<flowid_active_list_cur.numel()<< ", graph_id_cur: " << graph_id_cur<< ", fat: " << flow_arrival_time/1000.0 << ", fct: " << flow_completion_time/1000.0 << std::endl;
            if (flowid_active_list_cur.numel() > 0) {
                
                // Calculate time deltas for active flows
                auto time_deltas = (time_clock - time_last.index_select(0, flowid_active_list_cur).squeeze()).view({-1, 1});

                // Check if any time delta is greater than zero
                auto h_vec_time_updated = h_vec.index_select(0, flowid_active_list_cur);
                auto max_time_delta = torch::max(time_deltas).item<float>();
                if (max_time_delta>0.0f) {
                    // Update time using lstmcell_time
                    time_deltas.fill_(max_time_delta/1000.0f);
                    h_vec_time_updated = lstmcell_time.forward({ time_deltas, h_vec_time_updated}).toTensor();
                }

                // Create a mask for the edges corresponding to the active flows
                // auto edge_mask = torch::isin(edge_index[0], flowid_active_list_cur);
                auto edge_mask = torch::isin(edge_index[0], flowid_active_list_cur);
                auto selected_indices = edge_mask.nonzero().squeeze();
                auto edge_index_cur = edge_index.index_select(1, selected_indices);

                // Determine the number of active flows
                auto n_flows_active_cur = flowid_active_list_cur.size(0);
                auto new_flow_indices=torch::searchsorted(flowid_active_list_cur,edge_index_cur[0]);

                // Extract return_inverse from the tuple (index 1 of the tuple)
                auto unique_result_tuple = torch::_unique(edge_index_cur[1], /*sorted=*/true, /*return_inverse=*/true);
                auto active_link_idx = std::get<0>(unique_result_tuple);  // Unique link IDs
                auto new_link_indices = std::get<1>(unique_result_tuple); // Inverse indices for reindexing

                new_link_indices += n_flows_active_cur;
                auto edges_list_active=torch::cat({ torch::stack({new_flow_indices, new_link_indices}, /*dim=*/0), torch::stack({new_link_indices, new_flow_indices}, /*dim=*/0)}, /*dim=*/1);

                // Forward pass through the GNN layers
                auto z_t_link_cur=z_t_link.index_select(0,active_link_idx);
                auto x_combined=torch::cat({h_vec_time_updated, z_t_link_cur}, 0);

                //x_combined_padded = torch::constant_pad_nd(x_combined, torch::IntList{10 - x_combined.sizes()[0], 0}, 0);
                //edges_list_active_padded = torch::constant_pad_nd(edges_list_active, torch::IntList{0, 12 - edges_list_active.sizes()[1]}, 0);

                //at::Tensor gnn_output_1_padded;
                //if (!graphCreated) {
                //    cuda_graph.capture_begin(); {
                //        std::cout << "inference guard\n";
                //        torch::InferenceMode guard;
                //        std::cout << "first layer\n";
                //        auto gnn_output_0 = gnn_layer_0.forward({x_combined_padded, edges_list_active_padded}).toTensor();
                //        std::cout << "second layer\n";
                //        gnn_output_1_padded = gnn_layer_1.forward({gnn_output_0, edges_list_active_padded}).toTensor();
                //        std::cout << "nothing remains\n";
                //    }
                //    cuda_graph.capture_end();
                //    graphCreated = true;
                //} else {
                //    cuda_graph.replay();
                //}
                //capture_stream.synchronize();
                //std::cout << "this stuff\n";
                //cuda_graph.replay();
                //auto gnn_output_1 = gnn_output_1_padded.slice(0, 0, x_combined.sizes()[0]);

                auto gnn_output_0 = gnn_layer_0.forward({x_combined, edges_list_active}).toTensor();
                auto gnn_output_1 = gnn_layer_1.forward({gnn_output_0, edges_list_active}).toTensor();
                std::cout << x_combined.sizes() << " " << edges_list_active.sizes() << "\n";

                // Update rate using lstmcell_rate
                auto h_vec_rate_updated=gnn_output_1.slice(0,0,n_flows_active_cur);

                h_vec_rate_updated = lstmcell_rate.forward({ h_vec_rate_updated, h_vec_time_updated }).toTensor();

                // Update h_vec with the new hidden states
                h_vec.index_copy_(0, flowid_active_list_cur, h_vec_rate_updated);

                // auto z_t_link_updated = gnn_output_1.slice(0, n_flows_active_cur, n_flows_active_cur + active_link_idx.size(0));
                // z_t_link.index_copy_(0, active_link_idx, z_t_link_updated);

                // Update time_last to the current time for active flows
                time_last.index_put_({flowid_active_list_cur}, time_clock);
            }
        }

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_elapsed = end_time - start_time;
        std::cout << "Time elapsed: " << time_elapsed.count() << " seconds." << std::endl;

        // Copy results back to output arrays
        std::memcpy(res_fct, res_fct_tensor.cpu().data_ptr<float>(), sizeof(float) * n_flows * 2);
        std::memcpy(res_sldn, res_sldn_tensor.cpu().data_ptr<float>(), sizeof(float) * n_flows * 2);

        return 0; // Success
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
        return -1;
    }
}
