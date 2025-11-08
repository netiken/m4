#include "rdma.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/mtp-interface.h"
#include "ns3/global-value.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include <atomic>
#include <iostream>
#include "ns3/config.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("RdmaThreadSafe");

bool g_disableRdmaProcessing = false;
bool g_rdmaThreadSafe = false;  // Default to disabled until MTP is enabled
uint32_t g_rdmaThreadCount = 0; // Number of threads for RDMA optimization

void OptimizeRdmaForMtp(uint32_t threadCount) {
    if (threadCount <= 1) {
        std::cout << "RDMA MTP: Single thread mode, no optimization needed" << std::endl;
        return; // No optimization needed for single thread
    }
    
    g_rdmaThreadCount = threadCount;
    g_rdmaThreadSafe = true;
    std::cout << "RDMA MTP: Optimizing for " << threadCount << " threads" << std::endl;
    
    // Configure MTP scheduling for RDMA workloads
    GlobalValue::Bind("PartitionSchedulingMethod", StringValue("ByExecutionTime"));
    GlobalValue::Bind("PartitionSchedulingPeriod", UintegerValue(threadCount));
    
    // Additional RDMA optimizations for large topologies
    if (threadCount > 16) {
        // Use event-based scheduling for better performance
        GlobalValue::Bind("PartitionSchedulingMethod", StringValue("ByEventCount"));
    }
}

} // namespace ns3
