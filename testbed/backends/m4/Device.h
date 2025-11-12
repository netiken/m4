#include <map>
#include <memory>
#include "Type.h"

/**
 * Device class represents a single device in the network.
 * Device is usually an NPU or a switch.
 */
class Node {
 public:
  /**
   * Constructor.
   *
   * @param id id of the device
   */
  explicit Node(DeviceId id) noexcept;

  /**
   * Get id of the device.
   *
   * @return id of the device
   */
  [[nodiscard]] DeviceId get_id() const noexcept;

  /**
   * Connect a device to another device.
   *
   * @param id id of the device to connect this device to
   * @param bandwidth bandwidth of the link
   * @param latency latency of the link
   */
  void connect(DeviceId id, Bandwidth bandwidth, Latency latency) noexcept;

 private:
  DeviceId device_id;
  std::map<DeviceId, std::shared_ptr<Link>> links;

  [[nodiscard]] bool connected(DeviceId dest) const noexcept;
};

