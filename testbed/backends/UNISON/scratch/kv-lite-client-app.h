/*
 * KvLiteClientApp skeleton
 */
#pragma once

#include "ns3/application.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include <ns3/rdma-driver.h>
#include <ns3/rdma-client-helper.h>
#include "kv-lite-common.h"
#include <unordered_set>

namespace ns3 {

class KvLiteClientApp : public Application
{
  public:
    static TypeId GetTypeId(void);

    KvLiteClientApp();
    virtual ~KvLiteClientApp();

    // Hook: user can implement request construction logic
    virtual void ProcessSend(const KvLiteRequest &req);

    // Hook: called when response is delivered; user can compute metrics
    virtual void ProcessReceive(const KvLiteResponse &rsp);

  protected:
    virtual void StartApplication() override;
    virtual void StopApplication() override;

  private:
    // Schedules issuing a request at an exact time
    void IssueRequest(const KvLiteRequest &req);

    // Internal callback for when a response is delivered (wired from RDMA trace)
    void OnResponseDelivered(Ptr<RdmaRxQueuePair> rxq);

    // No-op callback for RDMA flow completion
    void OnFlowComplete();

  private:
    // Only keep what's used/varies across runs
    uint32_t m_priorityGroup = 3; // fixed
    Ipv4Address m_dstAddr; // fixed to server ip in testbed
    uint32_t m_dstDport = 4000; // fixed
    uint32_t m_sport = 10000; // fixed base
    uint32_t m_requestBytes = 17; // fixed RDMA setup request size
    uint64_t m_startDelayNs = 0; // fixed

    // De-duplication guards
    std::unordered_set<uint64_t> m_seenRespIds;
    std::unordered_set<uint64_t> m_sentHandshakeReqIds;
    std::unordered_set<uint64_t> m_seenDataIds;

    // Iteration control
    uint32_t m_maxRequests = kvlite::KVL_CLIENT_MAX_REQUESTS; // iterations per client
    uint32_t m_completedRequests = 0;
    uint64_t m_iterDelayNs = 0;
    uint64_t m_reqCounter = 1; // base reqId seed
    uint64_t m_baseReqId = 0; // starting reqId for this client
    uint32_t m_totalSentRequests = 0; // Total req_send issued so far

    // Windowed sends
    uint32_t m_maxWindows = 1; // the only knob we vary
    uint64_t m_sendDelayNs = 0; // fixed spacing
};

} // namespace ns3


