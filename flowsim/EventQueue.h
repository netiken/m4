#ifndef _EVENTQUEUE_
#define _EVENTQUEUE_

#include <list>
#include <unordered_map>
#include "Type.h"
#include "EventList.h"
#include "Event.h"

/**
 * EventQueue manages scheduled EventLists.
 */
class EventQueue {
 public:
  /**
   * Constructor.
   */
  EventQueue() noexcept;

  /**
   * Get current event time of the event queue.
   *
   * @return current event time
   */
  [[nodiscard]] EventTime get_current_time() const noexcept;

  /**
   * Check all registered events are invoked.
   * i.e., check if the event queue is empty.
   *
   * @return true if the event queue is empty, false otherwise
   */
  [[nodiscard]] bool finished() const noexcept;

  /**
   * Proceed the event queue.
   * i.e., first update the current event time to the next registered event time,
   * and then invoke all events registered at the current updated event time.
   */
  void proceed() noexcept;

  void schedule_arrival(EventTime arrival_time, Callback callback, CallbackArg callback_arg) noexcept;
  void schedule_completion(EventTime completion_time, Callback callback, CallbackArg callback_arg) noexcept;

  /**
   * Schedule an event with a given event time.
   *
   * @param event_time time of event
   * @param callback callback function pointer
   * @param callback_arg argument of the callback function
   * @return EventId ID of the scheduled event
   */
  /*
  EventId schedule_event(
      EventTime event_time,
      Callback callback,
      CallbackArg callback_arg) noexcept;
  */

  /**
   * Cancel a scheduled event.
   *
   * @param event_id ID of the event to cancel
   */
  //void cancel_event(EventId event_id) noexcept;

  void cancel_completion();

  void log_events();

 private:
  /// current time of the event queue
  EventTime current_time;

  /// next event ID to be assigned
  EventId next_event_id;

  /// list of EventLists
  //std::list<EventList> event_queue;

  //std::vector<Event> event_queue;
  //std::priority_queue<Event> event_queue;

  Event* next_arrival;

  Event* next_completion;


  /// map of event IDs to their corresponding event list iterator
  //std::unordered_map<EventId, std::list<EventList>::iterator> event_map;
};

#endif // _EVENTQUEUE_
