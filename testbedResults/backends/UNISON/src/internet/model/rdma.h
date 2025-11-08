#ifndef RDMA_H
#define RDMA_H

// For thread-local storage, ensure C++11 or newer
#if (__cplusplus >= 201103L) || (_MSC_VER >= 1900)
#define NS3_THREAD_LOCAL_AVAILABLE 1
#else
#define NS3_THREAD_LOCAL_AVAILABLE 0
#endif

#define ENABLE_QP 1

namespace ns3 {

extern bool g_disableRdmaProcessing;
extern bool g_rdmaThreadSafe;  // Flag to enable thread-safe mode for MTP
extern uint32_t g_rdmaThreadCount; // Number of threads for RDMA optimization

// Optimize RDMA for multi-threaded operation
void OptimizeRdmaForMtp(uint32_t threadCount);

} // namespace ns3

#endif
