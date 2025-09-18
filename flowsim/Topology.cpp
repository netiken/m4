#include "Topology.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <set>

std::shared_ptr<EventQueue> Topology::event_queue = nullptr;

void Topology::set_event_queue(std::shared_ptr<EventQueue> event_queue) noexcept {
    assert(event_queue != nullptr);
    Topology::event_queue = std::move(event_queue);
}

Topology::Topology(int devices_count, int npus_count) noexcept : npus_count(-1), devices_count(-1), dims_count(-1) {
    npus_count_per_dim = {};
    this->devices_count = devices_count;
    this->npus_count = npus_count;
    for (int i = 0; i < devices_count; ++i) {
        devices.push_back(std::make_shared<Device>(i));
    }
}
int Topology::get_devices_count() const noexcept {
    assert(devices_count > 0);
    assert(npus_count > 0);
    assert(devices_count >= npus_count);
    return devices_count;
}

int Topology::get_npus_count() const noexcept {
    assert(devices_count > 0);
    assert(npus_count > 0);
    assert(devices_count >= npus_count);
    return npus_count;
}

int Topology::get_dims_count() const noexcept {
    assert(dims_count > 0);
    return dims_count;
}

std::vector<int> Topology::get_npus_count_per_dim() const noexcept {
    assert(npus_count_per_dim.size() == dims_count);
    return npus_count_per_dim;
}

std::vector<Bandwidth> Topology::get_bandwidth_per_dim() const noexcept {
    assert(bandwidth_per_dim.size() == dims_count);
    return bandwidth_per_dim;
}

void Topology::send(std::unique_ptr<Chunk> chunk) noexcept {
    std::cout << "Sending chunk with size " << chunk->get_size() << " from device " << chunk->current_device()->get_id() << " to device " << chunk->get_dest_device()->get_id() << std::endl;
    assert(chunk != nullptr);

    chunk->set_topology(this);
    cancel_all_events();

    active_chunks_ptrs.push_back(std::move(chunk));
    Chunk* chunk_ptr = active_chunks_ptrs.back().get();
    active_chunks.push_back(chunk_ptr);

    std::cout << "Total active chunks: " << active_chunks.size() << std::endl;
    for (Chunk* ch : active_chunks) {
        std::cout << "Debug: Chunk ID: " << ch->get_completion_event_id() << ", Remaining size: " << ch->get_remaining_size() << ", Size: " << ch->get_size() << " from device " << ch->current_device()->get_id() << " to device " << ch->get_dest_device()->get_id() << std::endl;
    }

    add_chunk_to_links(chunk_ptr);

    // Initialize per-chunk path latency once, based on its route
    Latency total_latency = 0.0;
    const auto& route = chunk_ptr->get_route();
    auto it = route.begin();
    while (it != route.end()) {
        auto src_device = (*it)->get_id();
        ++it;
        if (it == route.end()) break;
        auto dest_device = (*it)->get_id();
        auto link_it = link_map.find(std::make_pair(src_device, dest_device));
        if (link_it != link_map.end()) {
            total_latency += link_it->second->get_latency();
        }
    }
    chunk_ptr->set_initial_path_latency(total_latency);
    update_link_states();
    reschedule_active_chunks();
}

void Topology::connect(DeviceId src, DeviceId dest, Bandwidth bandwidth, Latency latency, bool bidirectional) noexcept {
    assert(0 <= src && src < devices_count);
    assert(0 <= dest && dest < devices_count);
    assert(bandwidth > 0);
    assert(latency >= 0);

    auto link = std::make_shared<Link>(bandwidth, latency);
    link_map[std::make_pair(src, dest)] = link;

    if (bidirectional) {
        auto reverse_link = std::make_shared<Link>(bandwidth, latency);
        link_map[std::make_pair(dest, src)] = reverse_link;
    }

    std::cout << "Debug: Connecting src: " << src << " dest: " << dest << " with bandwidth: " << bandwidth << std::endl;
    if (bidirectional) {
        std::cout << "Debug: Connecting dest: " << dest << " src: " << src << " with bandwidth: " << bandwidth << std::endl;
    }
}

//void Topology::instantiate_devices() noexcept {
//    for (int i = 0; i < devices_count; ++i) {
//        devices.push_back(std::make_shared<Device>(i));
//    }
//}

std::shared_ptr<Device> Topology::get_device(int index) {
    return this->devices.at(index);
}

void Topology::update_link_states() {
    std::set<Chunk*> fixed_chunks; 
    while (fixed_chunks.size() < active_chunks.size()) {
        double bottleneck_rate = std::numeric_limits<double>::max();
        std::pair<DeviceId, DeviceId> bottleneck_link;
        for (const auto& link : active_links) {
            double fair_rate = calculate_bottleneck_rate(link, fixed_chunks);
            if (fair_rate < bottleneck_rate) {
                bottleneck_rate = fair_rate;
                bottleneck_link = link;
            }
        }
        if (bottleneck_rate < std::numeric_limits<double>::max()) {
            for (Chunk* chunk : link_map[bottleneck_link]->active_chunks) {
                if (fixed_chunks.find(chunk) == fixed_chunks.end()) {
                    chunk->set_rate(bottleneck_rate);
                    fixed_chunks.insert(chunk);
                }
            }
        } else {
            std::cout << "Debug: No active chunks on bottleneck link (" << bottleneck_link.first << " -> " << bottleneck_link.second << ")" << std::endl;
            break;
        }
    }
    
    std::cout << "New link state\n";
    for (auto const& chunk : active_chunks) {
        std::cout << "Route ";
        for (auto const& step : chunk->get_route()) {
            std::cout << step->get_id() << " ";
        }
        std::cout << "\n\tRate: " << chunk->get_rate() << "\n";
    }
    
}

double Topology::calculate_bottleneck_rate(const std::pair<DeviceId, DeviceId>& link, const std::set<Chunk*>& fixed_chunks) {
    double remaining_bandwidth = link_map[link]->get_bandwidth();
    int active_chunks = 0;

    for (Chunk* chunk : link_map[link]->active_chunks) {
        if (fixed_chunks.find(chunk) == fixed_chunks.end()) {
            ++active_chunks;
        } else {
            remaining_bandwidth -= chunk->get_rate();
        }
    }

    double fair_rate = active_chunks > 0 ? remaining_bandwidth / active_chunks : std::numeric_limits<double>::max();
    std::cout << "Debug: Link (" << link.first << " -> " << link.second << "), Fair rate: " << fair_rate << ", Remaining bandwidth: " << remaining_bandwidth << ", Active chunks: " << active_chunks << std::endl;
    return fair_rate;
}

void Topology::reschedule_active_chunks() {
    const auto current_time = event_queue->get_current_time();
    std::cout << "Debug: Rescheduling num active chunks: " << active_chunks.size() << std::endl;
    uint64_t min = -1;
    double completion_time;
    std::cout << "Debug: Rescheduling num active chunks: " << active_chunks.size() << std::endl;
    //Chunk* next_chunk = nullptr;
    std::vector<Chunk*> next_chunks;
    for (Chunk* chunk : active_chunks) {
        //if(chunk->get_completion_event_id() == 0){
            double remaining_size = chunk->get_remaining_size();
            double new_rate = chunk->get_rate();
            double new_completion_time = std::max(1.0, (remaining_size / new_rate));
            std::cout << "Debug: Calculating completion time for chunk ID " << chunk->get_completion_event_id() << ", Remaining size: " << remaining_size << ", New rate: " << new_rate << ", New completion time: " << new_completion_time << std::endl;

            // Add only the remaining path latency (do not re-add full latency every reschedule)
            Latency remaining_latency = chunk->get_remaining_path_latency();
            new_completion_time += remaining_latency;

            // Set baseline for next interval's progress accounting
            chunk->set_transmission_start_time(current_time);
            // Update remaining size explicitly
            chunk->set_remaining_size(remaining_size);

            auto* chunk_ptr = static_cast<void*>(chunk);  // Ensure proper chunk pointer
            std::cout << "Debug: Scheduling chunk ID " << chunk->get_completion_event_id() << ", Chunk rate: " << new_rate << ", Remaining size: " << remaining_size << ", Current time: " << current_time << ", New completion time: " << new_completion_time << std::endl;
            if (min == -1 || min > current_time + new_completion_time) {
                next_chunks.clear();
                min = current_time + new_completion_time;
                next_chunks.push_back(chunk);
            } else if (min == current_time + new_completion_time) {
                next_chunks.push_back(chunk);
            }
            std::cout << "Estimated duration: " << new_completion_time - current_time << std::endl;
        //}
    }
    std::cout << "min " << min << "\n";
    for (Chunk* chunk : next_chunks) {
        auto* chunk_ptr = static_cast<void*>(chunk);

        int new_event_id = event_queue->schedule_event(min, chunk_completion_callback, chunk_ptr);
        chunk->set_completion_event_id(new_event_id);
    }
}

void Topology::add_chunk_to_links(Chunk* chunk) {
    const auto& route = chunk->get_route();
    auto it = route.begin();
    while (it != route.end()) {
        auto src_device = (*it)->get_id();
        ++it;
        if (it == route.end()) break;
        auto dest_device = (*it)->get_id();
        link_map[std::make_pair(src_device, dest_device)]->active_chunks.push_back(chunk);
        active_links.insert(std::make_pair(src_device, dest_device));
    }
}

void Topology::remove_chunk_from_links(Chunk* chunk) {
    const auto& route = chunk->get_route();
    auto it = route.begin();
    while (it != route.end()) {
        auto src_device = (*it)->get_id();
        ++it;
        if (it == route.end()) break;
        auto dest_device = (*it)->get_id();
        auto& active_chunks = link_map[std::make_pair(src_device, dest_device)]->active_chunks;
        active_chunks.remove(chunk);
        if (active_chunks.empty()) {
            active_links.erase(std::make_pair(src_device, dest_device));
        }
        std::cout << "Debug: Removed chunk from link (" << src_device << " -> " << dest_device << "). It now has " << active_chunks.size() << " active chunks." << std::endl;
    }
}

void Topology::chunk_completion_callback(void* arg) noexcept {
    Chunk* chunk = static_cast<Chunk*>(arg);
    Topology* topology = chunk->get_topology();

    // Cancel all events
    topology->cancel_all_events();

    // Perform necessary updates and remove chunk from active chunks
    topology->remove_chunk_from_links(chunk);
    topology->active_chunks.remove(chunk);

    std::cout << "Debug: Chunk completion callback completed for chunk ID: " << chunk->get_completion_event_id() << std::endl;

    // Update link states and reschedule active chunks
    topology->update_link_states();
    topology->reschedule_active_chunks();

    // Invoke the chunk's callback
    chunk->invoke_callback();
}

void Topology::cancel_all_events() noexcept {
    const auto current_time = event_queue->get_current_time();
    std::cout << "Debug: Cancelling num of events: " << active_chunks.size() << std::endl;
    for (Chunk* chunk : active_chunks) {
        // Account for elapsed time since last schedule
        double elapsed_time = current_time - chunk->get_transmission_start_time();
        if (elapsed_time < 0) elapsed_time = 0; // safety
        // Consume path latency first
        chunk->consume_path_latency(elapsed_time);
        // Then consume data transmission
        double transmitted_size = elapsed_time * chunk->get_rate();
        chunk->update_remaining_size(transmitted_size);
        event_queue->cancel_event(chunk->get_completion_event_id());
        chunk->set_completion_event_id(0);
    }
}
