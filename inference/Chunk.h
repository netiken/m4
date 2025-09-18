#ifndef _CHUNK_
#define _CHUNK_

#include <memory>
#include <vector>
#include <list>
#include "Type.h"
#include "EventQueue.h"

/**
 * Chunk class represents a chunk.
 * Chunk is a basic unit of transmission.
 */
class Chunk {
 public:
  /**
   * Constructor.
   *
   * @param chunk_size: size of the chunk
   * @param route: route of the chunk from its source to destination
   * @param callback: callback to be invoked when the chunk arrives destination
   * @param callback_arg: argument of the callback
   */
  Chunk(
      int id,
      ChunkSize chunk_size,
      const Route& route,
      Callback callback,
      CallbackArg callback_arg) noexcept;

  int get_id();
  [[nodiscard]] std::shared_ptr<Node> current_device() const noexcept;
  [[nodiscard]] std::shared_ptr<Node> next_device() const noexcept;
  void mark_arrived_next_device() noexcept;
  [[nodiscard]] bool arrived_dest() const noexcept;
  [[nodiscard]] ChunkSize get_size() const noexcept;
  [[nodiscard]] ChunkSize get_remaining_size() const noexcept;
  void set_remaining_size(ChunkSize transmitted_size) noexcept;
  void update_remaining_size(ChunkSize transmitted_size) noexcept;
  [[nodiscard]] double get_rate() const noexcept;
  void set_rate(double rate) noexcept;
  void invoke_callback() noexcept;
  void set_transmission_start_time(EventTime start_time) noexcept;
  [[nodiscard]] EventTime get_transmission_start_time() const noexcept;
  [[nodiscard]] EventId get_completion_event_id() const noexcept;
  void set_completion_event_id(EventId event_id) noexcept;
  void set_topology(Topology* topology) noexcept;
  Topology* get_topology() const noexcept;
  [[nodiscard]] const Route& get_route() const noexcept;
  [[nodiscard]] std::shared_ptr<Node> get_dest_device() const noexcept; // New method

  // Path latency accounting helpers (mirrors flowsim)
  void set_initial_path_latency(Latency total_latency) noexcept;
  [[nodiscard]] Latency get_remaining_path_latency() const noexcept;
  void consume_path_latency(Latency elapsed_ns) noexcept;

 private:
  int id;
  ChunkSize chunk_size;
  ChunkSize remaining_size;
  Route route;
  Callback callback;
  CallbackArg callback_arg;
  EventTime transmission_start_time;
  double rate;
  EventId completion_event_id_;
  Topology* topology;
  Latency remaining_path_latency;
};

#endif // _CHUNK_
