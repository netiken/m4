#include <memory>
#include <vector>
#include "Type.h"
#include "Chunk.h"

/**
 * Link models physical links between two devices.
 */
class Link {
 public:
  /**
   * Constructor.
   *
   * @param bandwidth bandwidth of the link
   * @param latency latency of the link
   */
  Link(Bandwidth bandwidth, Latency latency) noexcept
      : bandwidth(bandwidth), latency(latency) {}

  /**
   * Get the bandwidth of the link.
   *
   * @return bandwidth of the link
   */
  [[nodiscard]] Bandwidth get_bandwidth() const noexcept {
    return bandwidth;
  }

  Latency get_latency() {
    return latency;
  }

  std::vector<Chunk*> active_chunks;

 private:
  Bandwidth bandwidth;
  Latency latency;

};
