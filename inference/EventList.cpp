#include "EventList.h"
#include <algorithm>
#include <cassert>
#include <iostream>

EventList::EventList(const EventTime event_time) noexcept
    : event_time(event_time) {
  assert(event_time >= 0);
}

EventTime EventList::get_event_time() const noexcept {
  return event_time;
}

void EventList::add_event(
    const Callback callback,
    const CallbackArg callback_arg,
    const EventId event_id) noexcept {
  assert(callback != nullptr);

  // add the event to the event list
  events.emplace_back(callback, callback_arg, event_id);
}

void EventList::invoke_events() noexcept {
  // invoke all events in the event list
  for (const auto& event : events) {
    event.callback(event.callback_arg);
  }
  events.clear();
}

void EventList::invoke_event() noexcept {
  if (empty()) {
    return;
  }
  auto const& event = events.front();
  events.pop_front();
  event.callback(event.callback_arg);
}

bool EventList::empty() noexcept {
  if (events.size() > 0) {
    return false;
  }
  return true;
}

void EventList::remove_event(EventId event_id) noexcept {
  // remove the event with the given ID from the event list
  events.remove_if([event_id](const EventEntry& event) {
    return event.event_id == event_id;
  });
}

int EventList::num_events() {
    return events.size();
}

