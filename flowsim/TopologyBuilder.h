#include <memory>
#include "Topology.h"

/**
 * Construct a Fat-Tree topology from a topology file.
 *
 * @param topology_file path to the topology file
 * @return pointer to the constructed Fat-Tree topology
 */
[[nodiscard]] std::shared_ptr<Topology> construct_fat_tree_topology(
    const std::string& topology_file) noexcept;

Bandwidth bw_GBps_to_Bpns(Bandwidth bw_GBps) noexcept;
