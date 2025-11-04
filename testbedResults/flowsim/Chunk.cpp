#include "Chunk.h"
#include <iostream>
#include <cassert>

Chunk::Chunk(ChunkSize chunk_size, Route route, Callback callback, CallbackArg callback_arg) noexcept
    : chunk_size(chunk_size), remaining_size(chunk_size), route(std::move(route)),
      callback(callback), callback_arg(callback_arg), transmission_start_time(0), rate(0), completion_event_id_(0), topology(nullptr), remaining_path_latency(0.0) {
        
        assert(chunk_size > 0);
        assert(!this->route.empty());
        assert(callback != nullptr);

      }

std::shared_ptr<Device> Chunk::current_device() const noexcept {
    return route.front();
}

std::shared_ptr<Device> Chunk::next_device() const noexcept {
    auto it = route.begin();
    std::advance(it, 1); // Move iterator to the second element
    return *it;
}

void Chunk::mark_arrived_next_device() noexcept {
    route.pop_front();
}

bool Chunk::arrived_dest() const noexcept {
    return route.size() == 1;
}

ChunkSize Chunk::get_size() const noexcept {
    return chunk_size;
}

ChunkSize Chunk::get_remaining_size() const noexcept {
    return remaining_size;
}

void Chunk::set_remaining_size(ChunkSize size) noexcept {
    this->remaining_size = size;
}

void Chunk::update_remaining_size(ChunkSize transmitted_size) noexcept {
    if (transmitted_size >= remaining_size) {
        this->remaining_size = 0;
    } else {
        this->remaining_size -= transmitted_size;
    }
}

double Chunk::get_rate() const noexcept {
    return rate;
}

void Chunk::set_rate(double rate) noexcept {
    this->rate = rate;
}

void Chunk::invoke_callback() noexcept {
    // std::cerr << "Debug: Invoking callback for chunk ID: " << completion_event_id_ << std::endl;
    
    (*callback)(callback_arg);

    // std::cerr << "Debug: Callback invoked for chunk ID: " << completion_event_id_ << std::endl;
}

void Chunk::set_transmission_start_time(EventTime start_time) noexcept {
    transmission_start_time = start_time;
}

EventTime Chunk::get_transmission_start_time() const noexcept {
    return transmission_start_time;
}

EventId Chunk::get_completion_event_id() const noexcept {
    return completion_event_id_;
}

void Chunk::set_completion_event_id(EventId event_id) noexcept {
    completion_event_id_ = event_id;
}

const Route& Chunk::get_route() const noexcept {
    return route;
}

Topology* Chunk::get_topology() const noexcept {
    return topology;
}

void Chunk::set_topology(Topology* topology) noexcept {
    this->topology = topology;
}

std::shared_ptr<Device> Chunk::get_dest_device() const noexcept {
    return route.back();
}

void Chunk::set_initial_path_latency(Latency total_latency) noexcept {
    remaining_path_latency = total_latency;
}

Latency Chunk::get_remaining_path_latency() const noexcept {
    return remaining_path_latency;
}

void Chunk::consume_path_latency(Latency elapsed_ns) noexcept {
    if (elapsed_ns >= remaining_path_latency) {
        remaining_path_latency = 0.0;
    } else {
        remaining_path_latency -= elapsed_ns;
    }
}
