#include "Event.h"
#include <cassert>

Event::Event(EventTime event_time, const Callback callback, const CallbackArg callback_arg) noexcept
    : event_time(event_time),
      callback(callback),
      callback_arg(callback_arg) {
    assert(callback != nullptr);
}

void Event::invoke_event() noexcept {
    // check the validity of the event
    assert(callback != nullptr);

    // invoke the callback function
    (*callback)(callback_arg);
}

EventTime Event::get_time() noexcept {
    return event_time;
}

std::pair<Callback, CallbackArg> Event::get_handler_arg() const noexcept {
    // check the validity of the event
    assert(callback != nullptr);

    return {callback, callback_arg};
}
