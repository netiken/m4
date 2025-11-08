/*
 * KvLiteClientApp skeleton (methods only, no logic)
 */
#include "kv-lite-client-app.h"
#include "ns3/log.h"
#include "ns3/node-list.h"
#include "ns3/ipv4.h"
#include "ns3/qbb-net-device.h"
#include "ns3/qbb-channel.h"
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED(KvLiteClientApp);

TypeId KvLiteClientApp::GetTypeId(void)
{
    static TypeId tid = TypeId("ns3::KvLiteClientApp")
        .SetParent<Application>()
        .SetGroupName("Applications")
        .AddConstructor<KvLiteClientApp>()
        .AddAttribute("MaxWindows", "Number of outstanding request sends per round",
                      UintegerValue(kvlite::KVL_DEFAULT_MAX_WINDOWS), MakeUintegerAccessor(&KvLiteClientApp::m_maxWindows), MakeUintegerChecker<uint32_t>())
        ;
    return tid;
}

KvLiteClientApp::KvLiteClientApp() {}
KvLiteClientApp::~KvLiteClientApp() {}

void KvLiteClientApp::StartApplication()
{
    Ptr<Node> self = GetNode();
    Ptr<RdmaDriver> rdma = self->GetObject<RdmaDriver>();
    if (rdma)
    {
        // Connect client delivery callback to print when a response arrives
        rdma->TraceConnectWithoutContext("QpDelivered", MakeCallback(&KvLiteClientApp::OnResponseDelivered, this));
    }

    // Hardcode destination server IP to node 0's primary address (11.0.0.1)
    m_dstAddr = Ipv4Address(kvlite::KVL_SERVER_BASE_IP);

    // Initialize request ID counter: either global 0-based or per-client stride
    if (kvlite::KVL_REQID_USE_CLIENT_STRIDE) {
        m_baseReqId = self->GetId() * kvlite::KVL_REQID_CLIENT_STRIDE;
    } else {
        m_baseReqId = 0;
    }

    // Use configured window size for concurrent pipelines

    for (uint32_t w = 0; w < m_maxWindows && m_totalSentRequests < m_maxRequests; ++w) {
        KvLiteRequest req;
        req.reqId = m_baseReqId + m_reqCounter;
        m_reqCounter++;
        req.clientNodeId = self->GetId();
        req.serverNodeId = 0; // server node id in this testbed
        req.requestBytes = kvlite::KVL_CLIENT_REQUEST_BYTES; // 17B setup request
        req.responseBytes = 0;
        uint64_t delay = kvlite::KVL_CLIENT_START_DELAY_NS + (kvlite::KVL_CLIENT_SEND_DELAY_NS * w);
        req.sendNs = Simulator::Now().GetNanoSeconds() + delay;
        Simulator::Schedule(NanoSeconds(delay), &KvLiteClientApp::IssueRequest, this, req);
    }
}

void KvLiteClientApp::StopApplication()
{
}

void KvLiteClientApp::OnFlowComplete()
{
    // Flow completion is now handled in ProcessReceive for DATA messages
}

void KvLiteClientApp::ProcessSend(const KvLiteRequest &req)
{
    const char* tag = (req.requestBytes == kvlite::KVL_HANDSHAKE_REQ_BYTES) ? "hand_send" : "req_send";
    // Log all client sends with proper format
    std::cout << "[client " << tag << "] t=" << Simulator::Now().GetNanoSeconds()
              << " ns reqId=" << req.reqId
              << " size=" << req.requestBytes << "B"
              << " client_node_id=" << GetNode()->GetId()
              << std::endl;
}

void KvLiteClientApp::ProcessReceive(const KvLiteResponse &rsp)
{
    KvLiteMsgType msgType = rsp.type;
    if (msgType == KvLiteMsgType::UNKNOWN) {
        msgType = (rsp.responseBytes == kvlite::KVL_SERVER_SMALL_RESP_BYTES) ? KvLiteMsgType::RESP : KvLiteMsgType::DATA;
    }
    switch (msgType) {
        case KvLiteMsgType::RESP: {
            if (m_seenRespIds.count(rsp.reqId)) {
                break; // already handled
            }
            m_seenRespIds.insert(rsp.reqId);
            std::cout << "[client resp_recv] t=" << Simulator::Now().GetNanoSeconds()
                      << " ns reqId=" << rsp.reqId
                      << " size=" << rsp.responseBytes << "B"
                      << " client_node_id=" << GetNode()->GetId()
                      << std::endl;
            // Trigger handshake
            KvLiteRequest hs;
            hs.type = KvLiteMsgType::HANDSHAKE;
            hs.reqId = rsp.reqId;
            hs.clientNodeId = rsp.clientNodeId;
            hs.serverNodeId = rsp.serverNodeId;
            hs.requestBytes = kvlite::KVL_HANDSHAKE_REQ_BYTES;
            hs.responseBytes = 0;
            hs.sendNs = Simulator::Now().GetNanoSeconds();
            IssueRequest(hs);
            
            break;
        }
        case KvLiteMsgType::DATA: {
            if (m_seenDataIds.count(rsp.reqId)) {
                break;
            }
            m_seenDataIds.insert(rsp.reqId);
            // Log the complete RDMA flow finish with proper format
            std::cout << "[client rdma_recv] t=" << Simulator::Now().GetNanoSeconds()
                      << " ns reqId=" << rsp.reqId
                      << " size=" << rsp.responseBytes << "B"
                      << " client_node_id=" << GetNode()->GetId()
                      << std::endl;
            // Complete one pipeline; possibly start another
            m_completedRequests++;
            if (m_completedRequests < m_maxRequests && m_totalSentRequests < m_maxRequests) {
            // Fix: Generate truly unique reqIds using global counter
            uint64_t nextReqId = m_baseReqId + m_reqCounter;
            m_reqCounter++;
                KvLiteRequest next;
            next.reqId = nextReqId;
                next.clientNodeId = rsp.clientNodeId;
                next.serverNodeId = rsp.serverNodeId;
                next.requestBytes = m_requestBytes;
                next.responseBytes = 0;
            next.sendNs = Simulator::Now().GetNanoSeconds() + m_iterDelayNs + kvlite::KVL_CLIENT_SEND_DELAY_NS;
            Simulator::Schedule(NanoSeconds(m_iterDelayNs + kvlite::KVL_CLIENT_SEND_DELAY_NS), &KvLiteClientApp::IssueRequest, this, next);
            }
            break;
        }
        default: {
            std::cout << "[client unknown_recv] t=" << Simulator::Now().GetNanoSeconds()
                      << " ns reqId=" << rsp.reqId
                      << " size=" << rsp.responseBytes << "B"
                      << " client_node_id=" << GetNode()->GetId()
                      << std::endl;
            break;
        }
    }

    // Handshake triggering is handled above in RESP case with dedup guard
}

void KvLiteClientApp::IssueRequest(const KvLiteRequest &req)
{
    ProcessSend(req);
    if (req.type != KvLiteMsgType::HANDSHAKE && req.requestBytes == kvlite::KVL_CLIENT_REQUEST_BYTES) {
        m_totalSentRequests++;
    }

    Ptr<Node> node = GetNode();
    Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
    Ipv4Address srcAddr = ipv4->GetAddress(1, 0).GetLocal();

    uint64_t baseRtt = 0; // pure one-way
    uint16_t portOffset = (req.type == KvLiteMsgType::HANDSHAKE) ? kvlite::KVL_PORT_OFF_HS : kvlite::KVL_PORT_OFF_REQ;
    uint16_t sport = static_cast<uint16_t>(kvlite::KVL_CLIENT_BASE_SPORT + portOffset + (req.reqId % 30000));
    Ptr<RdmaDriver> rdma = node->GetObject<RdmaDriver>();
    rdma->AddQueuePair(static_cast<uint32_t>(req.reqId),
                       req.requestBytes,
                       static_cast<uint16_t>(kvlite::KVL_PRIORITY_GROUP),
                       srcAddr,
                       m_dstAddr,
                       sport,
                       static_cast<uint16_t>(kvlite::KVL_DEFAULT_SERVER_DPORT),
                       0,
                       baseRtt,
                       MakeCallback(&KvLiteClientApp::OnFlowComplete, this));
}

void KvLiteClientApp::OnResponseDelivered(Ptr<RdmaRxQueuePair> rxq)
{
    KvLiteResponse rsp;
    rsp.reqId = 0;
    // Follow the pattern used in scratch/third.cc's qp_delivered
    uint32_t sid = (rxq->dip >> 8) & 0xffff; // destination IP node id (this client)
    uint32_t did = (rxq->sip >> 8) & 0xffff; // source IP node id (server)
    Ptr<Node> srcNode = NodeList::GetNode(sid);
    Ptr<RdmaDriver> rdma = srcNode->GetObject<RdmaDriver>();
    Ptr<RdmaQueuePair> q = rdma->m_rdma->GetQp(rxq->sip, rxq->dport, rxq->pg);

    rsp.clientNodeId = sid;
    rsp.serverNodeId = did;
    rsp.responseBytes = q ? q->m_size : 0;
    rsp.completeNs = Simulator::Now().GetNanoSeconds();
    if (q) {
        rsp.reqId = q->m_flowId;
    }

    ProcessReceive(rsp);
}

} // namespace ns3


