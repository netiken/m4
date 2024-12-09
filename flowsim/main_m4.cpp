#include "npy.hpp"
#include "Topology.h"
#include "TopologyBuilder.h"
#include "Type.h"
#include <vector>
#include <string>
#include <filesystem>

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

std::shared_ptr<EventQueue> event_queue;
std::shared_ptr<Topology> topology;
std::vector<Route> routing;
std::vector<int64_t> fat;
std::vector<int64_t> fsize;
std::unordered_map<int, int64_t> fct;
uint64_t limit;

void add_flow(int* index_ptr);


void record_fct(int* index) {
    int64_t fct_value = event_queue->get_current_time() - fat.at(*index);
    fct[*index] = fct_value;
    free(index);
}

void schedule_next_arrival(int index) {
    int* index_ptr = (int *)malloc(sizeof(int));
    *index_ptr = index;
    event_queue->schedule_event(fat.at(index), (void (*)(void*)) &add_flow, index_ptr);
}

void add_flow(int* index_ptr) {
    std::cout << "flow arrival " << *index_ptr << " " << fsize.at(*index_ptr) << "\n";
    Route route = routing.at(*index_ptr);
    int64_t flow_size = fsize.at(*index_ptr);
    auto chunk = std::make_unique<Chunk>(flow_size, route, (void (*)(void*)) &record_fct, index_ptr);
    topology->send(std::move(chunk));
    if (*index_ptr + 1 < limit) {
        schedule_next_arrival(*index_ptr + 1);
    }
}

void setup_m4() {
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
            const std::string model_dir = "../inference/models_topo/"; // Consider making this configurable
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
    }
}

int main(int argc, char *argv[]) {
    const std::string fat_path = argv[1];
    const std::string fsize_path = argv[2];
    const std::string topo_path = argv[3];
    const std::string routing_path = argv[4];
    const std::string write_path = argv[5];
 
    //const std::string fat_path {"0/ns3/fat.npy"};
    npy::npy_data d_fat = npy::read_npy<int64_t>(fat_path);
    std::vector<int64_t> arrival_times = d_fat.data;

    //const std::string fsize_path {"0/ns3/fsize.npy"};
    npy::npy_data d_fsize = npy::read_npy<int64_t>(fsize_path);
    std::vector<int64_t> flow_sizes = d_fsize.data;

    //const std::string topo_file = "0/ns3/topology.txt";
    topology = construct_fat_tree_topology(topo_path);

    std::filesystem::path cwd = std::filesystem::current_path() / routing_path;
    std::ifstream infile(cwd);
    int num_hops;

    limit = arrival_times.size();

    while (infile >> num_hops) {
        std::vector<int> hops;
        auto route = Route();
        for (int i = 0; i < num_hops; i++) {
            int64_t device_id;
            infile >> device_id;
            route.push_back(topology->get_device(device_id));
        }
        routing.push_back(route);
    }


    event_queue = std::make_shared<EventQueue>();
    topology->set_event_queue(event_queue);
    
    for (int i = 0; i < arrival_times.size() & i < limit; i++) {
        int64_t flow_size = flow_sizes.at(i);
        fat.push_back(arrival_times.at(i));
        fsize.push_back(flow_size);
        
        //int* index_ptr = (int *)malloc(sizeof(int));
        //*index_ptr = i;
        //event_queue->schedule_event(fat.at(i), (void (*)(void*)) &add_flow, index_ptr);
    }
    int* index_ptr = (int *)malloc(sizeof(int));
    *index_ptr = 0;
    event_queue->schedule_event(fat.at(0), (void (*)(void*)) &add_flow, index_ptr);

    while (!event_queue->finished()) {
        //event_queue->log_events();
        event_queue->proceed();
    }

    std::vector<int64_t> fct_vector;
    for (int i = 0; i < arrival_times.size() & i < limit; i++) {
        fct_vector.push_back(fct[i]);
    }

    //const std::string write_path(write_path);
    npy::npy_data<int64_t> d;
    d.data = fct_vector;
    d.shape = {limit};
    d.fortran_order = false;

    npy::write_npy(write_path, d);

}

