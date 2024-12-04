#include "TopologyBuilder.h"
#include "Topology.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>

std::tuple<int, int, std::vector<int>, std::vector<std::tuple<int, int, double, double, double>>> parse_fat_tree_topology_file(const std::string& topology_file) {
    std::ifstream file(topology_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open topology file");
    }

    int npus_count;
    int switch_node_count;
    int link_count;
    std::vector<int> switch_node_ids;
    std::vector<std::tuple<int, int, double, double, double>> links;

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> npus_count >> switch_node_count >> link_count;
    npus_count-=switch_node_count;

    std::getline(file, line);
    std::istringstream iss_switches(line);
    int switch_node_id;
    while (iss_switches >> switch_node_id) {
        switch_node_ids.push_back(switch_node_id);
    }

    while (std::getline(file, line)) {
        int src, dst;
        double rate, delay, error_rate;
        std::string rate_str, delay_str, error_rate_str;
        std::istringstream iss_link(line);
        iss_link >> src >> dst >> rate_str >> delay_str >> error_rate_str;
        rate = std::stod(rate_str.substr(0, rate_str.size() - 3)); // Removing "bps"
        rate = bw_GBps_to_Bpns(rate / 8.0);
        
        delay = std::stod(delay_str.substr(0, delay_str.size() - 2)); // Removing "ns"
        error_rate = std::stod(error_rate_str);
        links.emplace_back(src, dst, rate, delay, error_rate);
    }

    return std::make_tuple(npus_count, switch_node_count, switch_node_ids, links);
}

std::shared_ptr<Topology> construct_fat_tree_topology(const std::string& topology_file) noexcept {
    //std::cerr << "Constructing Fat-Tree topology from file: " << topology_file << std::endl;

    // Parse the topology file
    auto [npus_count, switch_node_count, switch_node_ids, links] = parse_fat_tree_topology_file(topology_file);

    // Create an instance of FatTreeTopology
    auto fat_tree_topology = std::make_shared<Topology>(npus_count + switch_node_count, npus_count);
    for (const auto& link : links) {
        int src = std::get<0>(link);
        int dest = std::get<1>(link);
        double bandwidth = std::get<2>(link);
        double latency = std::get<3>(link);
        bool bidirectional = true; // Assuming bidirectional links
        //std::cerr << "Connecting " << src << " to " << dest << " with bandwidth " << bandwidth << " GBps and latency " << latency << " ns" << std::endl;
        fat_tree_topology->connect(src, dest, bandwidth, latency, bidirectional);
    }

    return fat_tree_topology;
}

Bandwidth bw_GBps_to_Bpns(const Bandwidth bw_GBps) noexcept {
    assert(bw_GBps > 0);

    // 1 GB is 2^30 B
    // 1 s is 10^9 ns
    return bw_GBps / (1'000'000'000);
}
