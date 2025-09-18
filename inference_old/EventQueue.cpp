#include "EventQueue.h"
#include <cassert>
#include <iostream>
#include <limits>

EventQueue::EventQueue() noexcept : current_time(0), next_event_id(0) {
  // Create empty event queue
  //event_queue = std::list<EventList>();
  //event_queue = std::vector<EventEntry>();
    next_arrival = nullptr;
    next_completion = nullptr;
}

EventTime EventQueue::get_current_time() const noexcept {
  return current_time;
}

bool EventQueue::finished() const noexcept {
  // Check whether event queue is empty
  // std::cerr << "Checking if event queue is empty" << std::endl;
    if (next_arrival == nullptr && next_completion == nullptr) {
        return true;
    }
    return false;
}

void EventQueue::proceed() noexcept {
  // To proceed, the next event should exist
  assert(!finished());

  // Proceed to the next event time
  //auto& current_event_list = event_queue.front();
    EventTime arrival_time = std::numeric_limits<uint64_t>::max();
    EventTime completion_time = std::numeric_limits<uint64_t>::max();

    if (next_arrival != nullptr) {
        arrival_time = next_arrival->get_time();
    }

    if (next_completion != nullptr) {
        completion_time = next_completion->get_time();
    }

    if (arrival_time < completion_time) {
        assert(next_arrival != nullptr);
        Event arrival = *next_arrival;
        delete next_arrival;
        next_arrival = nullptr;
        current_time = arrival_time;
        arrival.invoke_event();
    } else {
        assert(next_completion != nullptr);
        Event completion = *next_completion;
        delete next_completion;
        next_completion = nullptr;
        current_time = completion_time;
        completion.invoke_event();
    }
}

void EventQueue::log_events() {
    std::cout << "Event logs\n";
}

void EventQueue::schedule_arrival(
    const EventTime arrival_time,
    const Callback callback,
    const CallbackArg callback_arg) noexcept {

    assert(arrival_time >= current_time);

    delete next_arrival;

    next_arrival = new Event(arrival_time, callback, callback_arg);
}

void EventQueue::schedule_completion(
    const EventTime completion_time,
    const Callback callback,
    const CallbackArg callback_arg) noexcept {

    assert(completion_time >= current_time);

    delete next_completion;
    next_completion = new Event(completion_time, callback, callback_arg);

}

void EventQueue::cancel_completion() {
    //delete next_completion;
    //next_completion = nullptr;
}

