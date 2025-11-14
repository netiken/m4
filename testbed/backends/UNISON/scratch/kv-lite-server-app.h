/*
 * KvLiteServerApp skeleton
 */
#pragma once

#include "ns3/application.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include <ns3/rdma-driver.h>
#include "kv-lite-common.h"
#include <unordered_set>

namespace ns3 {

class KvLiteServerApp : public Application
{
  public:
    static TypeId GetTypeId(void);

    KvLiteServerApp();
    virtual ~KvLiteServerApp();

    uint64_t GetServerOverhead() const {
        return kvlite::KVL_OVERHEAD_NS;
    }

    // Hook: user handles request and decides response size/behavior
    virtual void ProcessReceive(const KvLiteRequest &req);

    // Hook: user can observe when server sends response
    virtual void ProcessSend(const KvLiteResponse &rsp);

  protected:
    virtual void StartApplication() override;
    virtual void StopApplication() override;

  private:
    // Internal callback when a client request flow is delivered to this server
    void OnRequestDelivered(Ptr<RdmaRxQueuePair> rxq);

    // Issues a response flow back to client (skeleton)
    void IssueResponse(const KvLiteResponse &rsp);

    // No-op finish callback for server-originated flows
    void OnServerFlowComplete();

  private:
    uint32_t m_defaultResponseBytes = 10240; // the only knob we vary via attribute
    uint32_t m_windowSize = 1; // client window size (for overhead scaling)
    uint32_t m_defaultDport = 4000; // fixed
    uint32_t m_priorityGroup = 3; // fixed

    // Dedup guards
    std::unordered_set<uint64_t> m_seenHandshakeBaseIds;
};

} // namespace ns3


