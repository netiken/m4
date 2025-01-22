#include "npy.hpp"
#include "Topology.h"
#include "TopologyBuilder.h"
#include "Type.h"
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <queue>


// flowsim parameters
std::shared_ptr<EventQueue> event_queue;
std::shared_ptr<Topology> topology;
std::vector<Route> routing;
std::vector<int64_t> fat;
std::vector<int64_t> fsize;
std::vector<int64_t> host_ids;
std::unordered_map<int, int64_t> fct_map;
std::vector<int64_t> release_times;
uint64_t limit;

int32_t n_flows;

const uint32_t num_per_tor = 16;
const uint32_t num_tors = 70;

int get_tor(int flow_id) {
    return host_ids.at(flow_id) / num_per_tor;
}

int main(int argc, char *argv[]) {
    const std::string scenario_path = argv[1];
    const std::string fat_path = scenario_path + "/fat.npy";
    const std::string fsize_path = scenario_path + "/fsize.npy";
    const std::string topo_path = scenario_path + "/topology.txt";
    const std::string routing_path = scenario_path + "/flow_to_path.txt";
    const std::string write_path = argv[2];
    uint32_t tor_limit = std::stoi(argv[3]);
    const bool use_m4 = true;
    std::string release_path;
    if (argc == 5) {
        release_path = argv[4];
    }

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

    topology = construct_fat_tree_topology(topo_path);

    std::filesystem::path cwd = std::filesystem::current_path() / routing_path;
    std::ifstream infile(cwd);
    int num_hops;
    while (infile >> num_hops) {
        std::vector<int> hops;
        auto route = Route();
        int64_t device_id;
        infile >> device_id;
        host_ids.push_back(device_id);
        route.push_back(topology->get_device(device_id));
        for (int i = 1; i < num_hops; i++) {
            int64_t device_id;
            infile >> device_id;
            route.push_back(topology->get_device(device_id));
        }
        routing.push_back(route);
    }
    
    event_queue = std::make_shared<EventQueue>();
    topology->set_event_queue(event_queue);

    int flow_index = 0;
    int flows_completed = 0;
    float latency = topology->get_latency();
    n_flows = fat.size();

    int flow_counter = 0;
    std::queue<int> flow_queue;
    std::unordered_map<int, std::queue<int>> tor_map;
    std::unordered_map<int, uint32_t> flow_counts;
    for (int i = 0; i < num_tors; i++) {
        flow_counts[i] = 0;
    }

    for (int i = 0; i < n_flows; i++) {
        release_times.push_back(0);
    }

    while (flows_completed < n_flows) {
        EventTime arrival_time = std::numeric_limits<uint64_t>::max();
        EventTime completion_time = std::numeric_limits<uint64_t>::max();
        int chunk_id = -1;

        bool arrival;
        bool queued;

        /*
        if (flow_counter < n_flows) {
            arrival_time = fat.at(flow_counter);
            flow_index = flow_counter;
            queued = false;
        }
        */
        

        
        if (flow_queue.size() > 0) {
            flow_index = flow_queue.front();
            arrival_time = fat.at(flow_index) < topology->get_current_time() ? topology->get_current_time() : fat.at(flow_index); //-1; //fat.at(flow_index);
            queued = true;
        } else {
            while (flow_counter < n_flows) {
                int tor = get_tor(flow_counter);
                if (tor_limit == 0 || flow_counts[tor] < tor_limit) {
                    arrival_time = fat.at(flow_counter);
                    flow_index = flow_counter;
                    break;
                } else {
                    tor_map[tor].push(flow_counter);
                    flow_counter++;
                }
            }
            queued = false;
        }

        if (topology->has_completion_time()) {
            completion_time = topology->get_next_completion_time();
            chunk_id = topology->get_next_completion();
        }
        if (arrival_time < completion_time) {
            arrival = true;
        } else {
            arrival = false;
        }

        if (arrival) {
            std::cout << arrival_time << " flow arrival " << flow_index << " " << get_tor(flow_index) << "\n";
            release_times[flow_index] = arrival_time;
            topology->set_time(arrival_time);
            Route route = routing.at(flow_index);
            int64_t flow_size = fsize.at(flow_index);
            auto chunk = std::make_unique<Chunk>(flow_index, flow_size, route, (void (*)(void*)) nullptr, nullptr);
            topology->send(std::move(chunk));
            //flow_index++;

            //flow_counter++;
            int tor = get_tor(flow_index);
            flow_counts[tor]++;
            if (queued) {
                flow_queue.pop();
            } else {
                flow_counter++;
            }
        } else {
            std::cout << completion_time << " flow completed " << chunk_id << " " << get_tor(chunk_id) << "\n";
            topology->set_time(completion_time);
            //int64_t fct_value = topology->get_current_time() - fat.at(chunk_id) + routing.at(chunk_id).size() * latency;
            int64_t fct_value = topology->get_current_time() - release_times.at(chunk_id) + (routing.at(chunk_id).size() - 1) * latency;
            fct_map[chunk_id] = fct_value;
            topology->chunk_completion(chunk_id);
            flows_completed++;
            int tor = get_tor(chunk_id);
            flow_counts[tor]--;
            if (tor_map[tor].size() > 0) {
                int flow_id = tor_map[tor].front();
                flow_queue.push(flow_id);
                tor_map[tor].pop();
            }
        }
    }

    std::vector<float> fct_vector;
    for (int i = 0; i < arrival_times.size() & i < limit; i++) {
        fct_vector.push_back(std::max(0, (int) fct_map[i]));
    }

    npy::npy_data<float> d;
    d.data = fct_vector;
    d.shape = {limit};
    d.fortran_order = false;

    npy::write_npy(write_path, d);

    if (argc == 5) {
        std::vector<float> release_float;
        for (int i = 0; i < limit; i++) {
            release_float.push_back((float) release_times.at(i));
        }
        npy::npy_data<float> d;
        d.data = release_float;
        d.shape = {limit};
        d.fortran_order = false;
        npy::write_npy(release_path, d);
    }

    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();    
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start).count() << " seconds\n";
}