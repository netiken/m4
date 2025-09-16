#include "EventQueue.h"
#include <cassert>
#include <iostream> 

EventQueue::EventQueue() noexcept : current_time(0), next_event_id(0) {
  // Create empty event queue
  event_queue = std::list<EventList>();
}

EventTime EventQueue::get_current_time() const noexcept {
  return current_time;
}

bool EventQueue::finished() const noexcept {
  // Check whether event queue is empty
  // std::cerr << "Checking if event queue is empty" << std::endl;
  return event_queue.empty();
}

void EventQueue::proceed() noexcept {
  // To proceed, the next event should exist
  assert(!finished());

  // Proceed to the next event time
  auto& current_event_list = event_queue.front();

  // Check the validity and update current time
  assert(current_event_list.get_event_time() >= current_time);
  current_time = current_event_list.get_event_time();

  // Invoke events
  while (!current_event_list.empty()) {
    current_event_list.invoke_event();
  }
  //current_event_list.invoke_events();

  // Drop processed event list
  event_queue.pop_front();
}

void EventQueue::log_events() {
    std::cout << "Event lists: " << event_queue.size();
    int eventCount = 0;
    for (auto it = event_queue.begin(); it != event_queue.end(); it++) {
        eventCount += it->num_events();
    }
    std::cout << " " << eventCount << "\n";
}

EventId EventQueue::schedule_event(
    const EventTime event_time,
    const Callback callback,
    const CallbackArg callback_arg) noexcept {
  // Time should be at least larger than current time
  // std::cerr << "Scheduling event time: " << event_time << ", Current time: " << current_time << std::endl;
  assert(event_time >= current_time);

  // Find the entry to insert the event
  auto event_list_it = event_queue.begin();
  while (event_list_it != event_queue.end() &&
         event_list_it->get_event_time() < event_time) {
    event_list_it++;
  }

  // There can be three scenarios:
  // (1) Event list matching with event_time is found
  // (2) There's no event list matching with event_time
  //   (2-1) The event_time requested is
  //   larger than the largest event time scheduled
  //   (2-2) The event_time requested is
  //   smaller than the largest event time scheduled
  // For both (2-1) or (2-2), a new event should be created
  if (event_list_it == event_queue.end() ||
      event_time < event_list_it->get_event_time()) {
    // Insert new event_list
    event_list_it = event_queue.insert(event_list_it, EventList(event_time));
  }

  // Generate a new event ID
  EventId event_id = next_event_id++;

  // Now, whether (1) or (2), the entry to insert the event is found
  // Add event to event_list
  event_list_it->add_event(callback, callback_arg, event_id);

  // Store event in map for cancellation
  event_map[event_id] = event_list_it;
  // std::cerr << "Event scheduled at time " << event_time << " with ID " << event_id << std::endl;
  return event_id;
}

void EventQueue::cancel_event(EventId event_id) noexcept {
  // std::cerr << "Cancelling event with ID " << event_id << std::endl;
  auto it = event_map.find(event_id);
  if (it != event_map.end()) {
    auto& event_list_it = it->second;
    event_list_it->remove_event(event_id);
    event_map.erase(it);
  }
}
