#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAGraph.h>
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    int num_iterations = 1000;
    if (argc > 1) {
        num_iterations = std::stoi(argv[1]);
    }

    // Set device (GPU)
    torch::Device device(torch::kCUDA, 3);
    if (!torch::cuda::is_available() || device.index() >= torch::cuda::device_count()) {
        std::cerr << "[ERROR] CUDA device " << device.index() << " is not available!" << std::endl;
        return -1;
    }
    // Set the current CUDA device
    at::cuda::set_device(device.index());

    // Disable gradient calculations
    torch::NoGradGuard no_grad;

    // Load models
    auto gru_time = torch::jit::load("../models/lstmcell_time.pt", device);
    auto gru_rate = torch::jit::load("../models/lstmcell_rate.pt", device);
    auto mlp_model = torch::jit::load("../models/output_layer.pt", device);
    auto gnn_layer1 = torch::jit::load("../models/gnn_layer_0.pt", device);
    auto gnn_layer2 = torch::jit::load("../models/gnn_layer_1.pt", device);

    // Set models to evaluation mode
    gru_time.eval();
    gru_rate.eval();
    mlp_model.eval();
    gnn_layer1.eval();
    gnn_layer2.eval();

    int batch_size = 4000;

    // Pre-allocate input tensors
    auto time_deltas = torch::zeros({batch_size, 1}, device);
    auto rate_vec = torch::zeros({batch_size, 128}, device);
    auto h_vec = torch::zeros({batch_size, 128}, device);
    auto node_features = torch::zeros({batch_size, 130}, device);
    auto edge_index = torch::tensor({{0, 1, 2}, {1, 2, 0}}, torch::kLong).to(device);

    // Prepare inputs
    std::vector<torch::jit::IValue> lstm_inputs_time = {time_deltas, h_vec};
    std::vector<torch::jit::IValue> lstm_inputs_rate = {rate_vec, h_vec};
    std::vector<torch::jit::IValue> mlp_inputs = {h_vec};
    std::vector<torch::jit::IValue> gnn_inputs = {node_features, edge_index};

    // Warm-up
    gru_time.forward(lstm_inputs_time);
    gru_rate.forward(lstm_inputs_rate);
    mlp_model.forward(mlp_inputs);
    auto gnn_output = gnn_layer1.forward(gnn_inputs).toTensor();
    std::vector<torch::jit::IValue> gnn_inputs_layer2 = {gnn_output, edge_index};
    gnn_layer2.forward(gnn_inputs_layer2);

    torch::cuda::synchronize();

    // Disable JIT profiling and executor optimizations
    torch::jit::getProfilingMode() = true;
    torch::jit::getExecutorMode() = true;

    // Create a CUDA stream and set it as current
    auto capture_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(capture_stream);

    // Instantiate CUDAGraph object
    at::cuda::CUDAGraph cuda_graph;

    // Capture the CUDA Graph
    cuda_graph.capture_begin();
    {
        torch::InferenceMode guard;

        // Forward passes inside graph capture
        auto lstm_time_output = gru_time.forward(lstm_inputs_time).toTensor();
        auto lstm_rate_output = gru_rate.forward(lstm_inputs_rate).toTensor();
        auto mlp_output = mlp_model.forward(mlp_inputs).toTensor();
        auto gnn_output = gnn_layer1.forward(gnn_inputs).toTensor();

        gnn_inputs_layer2[0] = gnn_output;
        gnn_output = gnn_layer2.forward(gnn_inputs_layer2).toTensor();
    }
    cuda_graph.capture_end();
    /*
    std::vector<at::cuda::CUDAGraph> vec;
    int num = 10;
    for (int i = 0; i < num; i++) {
	at::cuda::CUDAGraph cuda_graph_i;
	cuda_graph_i.capture_begin();
	{
	    torch::InferenceMode guard;
	    auto lstm_time_output = gru_time.forward(lstm_inputs_time).toTensor();
	    auto lstm_rate_output = gru_rate.forward(lstm_inputs_rate).toTensor();
	    auto mlp_output = mlp_model.forward(mlp_inputs).toTensor();
	    auto gnn_output = gnn_layer1.forward(gnn_inputs).toTensor();

	    gnn_inputs_layer2[0] = gnn_output;
	    gnn_output = gnn_layer2.forward(gnn_inputs_layer2).toTensor();
	}
	cuda_graph_i.capture_end();
	vec.push_back(cuda_graph_i);
    }
    */
    capture_stream.synchronize();

    // Run inference using CUDA Graph
    auto start_graph = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        // Update input data in-place
        time_deltas.copy_(torch::randn({batch_size, 1}, device));
        rate_vec.copy_(torch::randn({batch_size, 128}, device));
        h_vec.copy_(torch::randn({batch_size, 128}, device));
        node_features.copy_(torch::randn({batch_size, 130}, device));

        cuda_graph.replay();
	//for (int j = 0; j < num; j++) {
	//    vec.at(j).replay();
	//}
    }

    capture_stream.synchronize();
    auto end_graph = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_graph = end_graph - start_graph;

    std::cout << "[INFO] CUDA Graph inference time for " << num_iterations << " iterations: "
              << duration_graph.count() << " s" << std::endl;

    return 0;
}
