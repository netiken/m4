#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAGraph.h> // Include for CUDAGraph
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Helper function for detailed CUDA error checking
void check_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA Error at " << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}
#define CUDA_CHECK(err) check_cuda_error(err, __FILE__, __LINE__)

// Function to load the model
torch::jit::script::Module load_model(const std::string& model_path, torch::Device device) {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path, device);
        module.to(device);
        std::cout << "[INFO] Model loaded successfully from: " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] Error loading the model: " << e.what() << std::endl;
        exit(-1);
    }
    return module;
}

int main(int argc, char* argv[]) {
    // Set device (GPU)
    torch::Device device(torch::kCUDA, 0);
    if (!torch::cuda::is_available()) {
        std::cerr << "[ERROR] CUDA is not available!" << std::endl;
        return -1;
    }

    int num_iterations = 1000;
    if (argc > 1) {
        num_iterations = std::stoi(argv[1]);
    }
    std::cout << "[INFO] Running on GPU." << std::endl;

    // Disable gradient calculations
    torch::NoGradGuard no_grad;

    // Load the model and ensure it's in evaluation mode
    torch::jit::script::Module model = load_model("../models/output_layer.pt", device);
    model.eval(); // Set model to inference mode

    // Prepare input tensor (batch size: 10, input size: 256)
    const int batch_size = 10;
    torch::Tensor input_tensor = torch::randn({batch_size, 256}, device);

    // Warm-up the model
    std::cout << "[INFO] Warming up the model." << std::endl;
    for (int i = 0; i < 5; ++i) {
        model.forward({input_tensor});
    }

    // Ensure all operations have finished before capturing
    torch::cuda::synchronize();

    // Create a new CUDA stream for graph capture
    at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool(true, device.index());

    // Set the current CUDA stream to capture_stream
    at::cuda::setCurrentCUDAStream(capture_stream);

    // Instantiate CUDAGraph object
    at::cuda::CUDAGraph cuda_graph;

    // Capture the CUDA Graph
    std::cout << "[INFO] Capturing CUDA Graph." << std::endl;
    try {
        // Begin graph capture
        cuda_graph.capture_begin();

        // Forward pass inside graph capture
        torch::Tensor output_tensor = model.forward({input_tensor}).toTensor();

        // End graph capture
        cuda_graph.capture_end();

    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] Exception during CUDA Graph capture: " << e.what() << std::endl;
        return -1;
    }

    // Ensure all operations have finished
    capture_stream.synchronize();

    // Run inference using CUDA Graph
    std::cout << "[INFO] Running inference using CUDA Graph with updated inputs." << std::endl;
    auto start_graph = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        // Update input tensor data
        {
            // Generate new data on CPU
            torch::Tensor new_input_data = torch::randn({batch_size, 256});

            // Copy data to GPU tensor without changing its memory location
            input_tensor.copy_(new_input_data);
        }

        // Optionally, synchronize if necessary
        // torch::cuda::synchronize();

        // Replay the CUDA Graph
        cuda_graph.replay();
    }
    capture_stream.synchronize();
    auto end_graph = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_graph = end_graph - start_graph;
    std::cout << "[INFO] CUDA Graph inference time: " << duration_graph.count() << " s" << std::endl;

    return 0;
}
