#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <chrono>

// Function to load and optimize a model
torch::jit::script::Module load_and_optimize_model(const std::string& model_path, torch::Device device) {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.to(device);
        module = torch::jit::optimize_for_inference(module);  // Optimize the model for inference
        std::cout << "[INFO] Model loaded and optimized successfully from: " << model_path << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "[ERROR] Error loading the model: " << e.what() << std::endl;
        throw;
    }
    return module;
}

// Function to run LSTM inference
torch::Tensor run_lstm_inference(torch::jit::script::Module& model, std::vector<torch::jit::IValue>& inputs) {
    return model.forward(inputs).toTensor();
}

// Function to run MLP inference
torch::Tensor run_mlp_inference(torch::jit::script::Module& model, std::vector<torch::jit::IValue>& inputs) {
    return model.forward(inputs).toTensor();
}

// Function to run GNN inference
torch::Tensor run_gnn_inference(torch::jit::script::Module& model, std::vector<torch::jit::IValue>& inputs) {
    return model.forward(inputs).toTensor();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "[ERROR] Please provide num_iterations as a command line argument." << std::endl;
        return -1;
    }

    int num_iterations = std::stoi(argv[1]);
    std::cout << "[INFO] Running for " << num_iterations << " iterations." << std::endl;

    // Set device (CPU or GPU)
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "[INFO] CUDA is available! Running on GPU." << std::endl;
    } else {
        std::cout << "[INFO] Running on CPU." << std::endl;
    }

    // Load and optimize models using torch::jit::optimize_for_inference
    torch::jit::script::Module gru_time = load_and_optimize_model("../models/lstmcell_time.pt", device);
    torch::jit::script::Module gru_rate = load_and_optimize_model("../models/lstmcell_rate.pt", device);
    torch::jit::script::Module mlp_model = load_and_optimize_model("../models/output_layer.pt", device);
    torch::jit::script::Module gnn_layer1 = load_and_optimize_model("../models/gnn_layer_0.pt", device);
    torch::jit::script::Module gnn_layer2 = load_and_optimize_model("../models/gnn_layer_1.pt", device);

    int batch_size = 10;  // Batch size for inference
    // Pre-allocate input tensors outside the loop
    torch::Tensor time_deltas = torch::randn({batch_size, 1}, device);
    torch::Tensor rate_vec = torch::randn({batch_size, 256}, device);
    torch::Tensor h_vec = torch::randn({batch_size, 256}, device);
    torch::Tensor node_features = torch::randn({batch_size, 258}, device);
    torch::Tensor edge_index = torch::tensor({{0, 1}, {1, 0}}, torch::TensorOptions().dtype(torch::kLong).device(device));

    std::vector<torch::jit::IValue> lstm_inputs_time = {time_deltas, h_vec};
    std::vector<torch::jit::IValue> lstm_inputs_rate = {rate_vec, h_vec};
    std::vector<torch::jit::IValue> mlp_inputs = {h_vec};
    std::vector<torch::jit::IValue> gnn_inputs = {node_features, edge_index};
    std::vector<torch::jit::IValue> gnn_intermediate_output = {torch::Tensor(), edge_index};

    // Warm-up models to reduce overhead during timing
    torch::Tensor lstm_time_output = run_lstm_inference(gru_time, lstm_inputs_time);
    torch::Tensor lstm_rate_output = run_lstm_inference(gru_rate, lstm_inputs_rate);
    torch::Tensor mlp_output = run_mlp_inference(mlp_model, mlp_inputs);
    torch::Tensor gnn_output = run_gnn_inference(gnn_layer1, gnn_inputs);
    gnn_intermediate_output[0] = gnn_output;
    gnn_output = run_gnn_inference(gnn_layer2, gnn_intermediate_output);

    // Measure runtime
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        lstm_time_output = run_lstm_inference(gru_time, lstm_inputs_time);
        lstm_rate_output = run_lstm_inference(gru_rate, lstm_inputs_rate);
        mlp_output = run_mlp_inference(mlp_model, mlp_inputs);
        gnn_output = run_gnn_inference(gnn_layer1, gnn_inputs);
        gnn_intermediate_output[0] = gnn_output;
        gnn_output = run_gnn_inference(gnn_layer2, gnn_intermediate_output);
    }

    // Synchronize the GPU at the end of the operations
    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();  // Ensure all GPU work is completed before measuring
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time taken for " << num_iterations << " iterations: " << duration.count() << " s" << std::endl;
    return 0;
}
