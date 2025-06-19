#include <list>
#include "Type.h"

/**
 * EventList encapsulates a number of Events along with its event time.
 */
class EventList {
 public:
  /**
   * Constructor.
   *
   * @param event_time event time of the event list
   */
  explicit EventList(EventTime event_time) noexcept;

  /**
   * Get the registered event time.
   *
   * @return event time
   */
  [[nodiscard]] EventTime get_event_time() const noexcept;

  /**
   * Register an event into the event list.
   *
   * @param callback callback function pointer
   * @param callback_arg argument of the callback function
   * @param event_id ID of the event
   */
  void add_event(Callback callback, CallbackArg callback_arg, EventId event_id) noexcept;

  /**
   * Invoke all events in the event list.
   */
  void invoke_events() noexcept;

  /**
   * Remove an event from the event list.
   *
   * @param event_id ID of the event to remove
   */
  void remove_event(EventId event_id) noexcept;

  void invoke_event() noexcept;
  bool empty() noexcept;

  int num_events();

 private:
  /// event time of the event list
  EventTime event_time;

  /// struct to hold event details
  struct EventEntry {
    Callback callback;
    CallbackArg callback_arg;
    EventId event_id;

    EventEntry(Callback cb, CallbackArg arg, EventId id)
        : callback(cb), callback_arg(arg), event_id(id) {}
  };

  /// list of registered events
  std::list<EventEntry> events;
};
