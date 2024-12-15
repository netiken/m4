#include "Device.h"
#include <cassert>
#include "Chunk.h"
#include "Link.h"


Node::Node(const DeviceId id) noexcept : device_id(id) {
  assert(id >= 0);
}

DeviceId Node::get_id() const noexcept {
  assert(device_id >= 0);
  return device_id;
}

void Node::connect(const DeviceId id, const Bandwidth bandwidth, const Latency latency) noexcept {
  assert(id >= 0);
  assert(bandwidth > 0);
  assert(latency >= 0);

  // assert there's no existing connection
  assert(!connected(id));

  // create link
  links[id] = std::make_shared<Link>(bandwidth, latency);
}

bool Node::connected(const DeviceId dest) const noexcept {
  assert(dest >= 0);
  // check whether the connection exists
  return links.find(dest) != links.end();
}
