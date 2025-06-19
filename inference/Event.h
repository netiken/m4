#include "Type.h"
#include <tuple>

/**
 * Event is a wrapper for a callback function and its argument.
 */
class Event {
  public:
    /**
     * Constructor.
     *
     * @param callback function pointer
     * @param callback_arg argument of the callback function
     */
    Event(EventTime event_time, Callback callback, CallbackArg callback_arg) noexcept;

    /**
     * Invoke the callback function.
     */
    void invoke_event() noexcept;

    EventTime get_time() noexcept;

    /**
     * Get the callback function and the argument.
     *
     * @return callback function and its argument
     */
    [[nodiscard]] std::pair<Callback, CallbackArg> get_handler_arg() const noexcept;


  private:

    EventTime event_time;

    /// pointer to the callback function
    Callback callback;

    /// argument of the callback function
    CallbackArg callback_arg;
};
