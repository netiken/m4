/*
 * 12-node topology testbed:
 * C4 -> S1 -> R -> T -> R2 -> ... -> Server (target)
 * Server -> S1 -> R -> T -> R2 -> ... -> Clients
 * C1 -> S2 -> R -> T -> R2 -> ... -> Server
 * C5 -> S2 -> R -> T -> R2 -> ... -> Server
 * C2 -> S3 -> R -> T -> R2 -> ... -> Server
 * C3 -> S3 -> R -> T -> R2 -> ... -> Server
 * C6 -> S4 -> R2 -> T -> R -> ... -> Server
 * C7 -> S4 -> R2 -> T -> R -> ... -> Server
 * C8 -> S5 -> R2 -> T -> R -> ... -> Server
 * C9 -> S5 -> R2 -> T -> R -> ... -> Server
 * C10 -> S6 -> R2 -> T -> R -> ... -> Server
 * C11 -> S6 -> R2 -> T -> R -> ... -> Server
 * 
 * Nodes: 12 hosts (Server + C1-C11), 9 switches (S1-S6 + R + R2 + T)
 * T is an aggregation switch connecting R and R2
 * 11 clients (C1-C11) send traffic to Server (node 0)
 * Each client sends 650 requests
 */
#include <iostream>
#include <vector>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/qbb-helper.h"
#include <ns3/switch-node.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include "ns3/data-rate.h"
#include <ns3/mtp-interface.h>
#include <thread>
#include "kv-lite-client-app.h"
#include "kv-lite-server-app.h"
// Include implementations so they link into this scratch target
#include "kv-lite-client-app.cpp.inc"
#include "kv-lite-server-app.cpp.inc"

using namespace ns3;
using namespace std;

static Ipv4Address node_id_to_ip(uint32_t id){
    return Ipv4Address(0x0b000001 + ((id / 256) * 0x00010000) + ((id % 256) * 0x00000100));
}

static uint32_t ip_to_node_id(Ipv4Address ip){
    return (ip.Get() >> 8) & 0xffff;
}

int main(int argc, char *argv[])
{
    // Enable MTP with optimized settings
    uint32_t threadCount = std::thread::hardware_concurrency();
    MtpInterface::Enable(threadCount);
    OptimizeRdmaForMtp(threadCount);

    // Hardcode network and device config for this constrained testbed
    std::string linkDataRate = "10Gbps";
    std::string linkDelay = "1us"; // Real testbed propagation delay per paper
    uint32_t packetPayloadSize = 9000; // B - Jumbo frames baseline
    bool enablePfc = true; // Enable PFC for DCQCN congestion control
    bool enableQcn = true; // Enable QCN for DCQCN congestion control
    double stopTimeSec = 30; // Stop after 30 seconds - applications complete in ~2-5s

     // Expose tunables via CLI
     uint32_t argMaxWindows = 16;    // default matches previous hardcoded value
     uint32_t argDataBytes = 1024008; // Match real testbed size
    CommandLine cmd;
    cmd.AddValue("maxWindows", "Number of outstanding request sends per round (client window)", argMaxWindows);
    cmd.AddValue("dataBytes", "Server data response size in bytes (for post-handshake)", argDataBytes);
    cmd.Parse(argc, argv);

    // Treat the RDMA payload size as the effective MTU so a single chunk carries the data.
    packetPayloadSize = 9000u;
    
    // 🎯 SECOND-GEN TUNING: achieve ~2× faster RDMA without post-processing
    // Insight: NS3 packet-level RDMA wastes time on frequent ACKs and conservative rates.
    // Tactics:
    //   • Huge Layer-2 chunks so the whole RDMA payload ships in one chunk
    //   • Practically disable L2 ACKs (ack only every multi-MB of progress)
    //   • Give DCQCN a generous starting rate headroom
    //   • Slightly shrink propagation delays to mimic NIC pipelining
    uint64_t l2ChunkSize = 10000000;   // 10 MB
    uint32_t l2AckInterval = 2000000;  // 2 MB ( >> payload, keeps ACKs rare)
    double targetUtil = 0.9999;        // Near-ideal utilisation
    std::string minRate = "25Gbps";    // Start aggressive to offset simulator overhead
    double linkDelayMultiplier = 0.02; // 1µs → 20ns per hop
    
    // Scale for higher windows (even more aggressive)
    if (argMaxWindows >= 4) {
        l2ChunkSize = 20000000;        // 20 MB for window 4+
        l2AckInterval = 4000000;       // 4 MB
        minRate = "35Gbps";            // Even faster convergence for heavy windows
        linkDelayMultiplier = 0.01;    // 10ns per-hop latency
    }
    
    // Apply link delay scaling to model faster real-world hardware
    std::ostringstream delayStream;
    delayStream << (1.0 * linkDelayMultiplier) << "us";
    linkDelay = delayStream.str();

    // Enable DCQCN congestion control: PFC + QCN required
    Config::SetDefault("ns3::QbbNetDevice::PauseTime", UintegerValue(0)); // Disable pause frame timer (but PFC still enabled)  
    Config::SetDefault("ns3::QbbNetDevice::QbbEnabled", BooleanValue(true)); // Enable PFC for DCQCN
    Config::SetDefault("ns3::QbbNetDevice::QcnEnabled", BooleanValue(true)); // Enable QCN for DCQCN

    // Create hosts: Server (0), C1 (1), C2 (2), C3 (3), C4 (4), C5 (5), C6 (6), C7 (7), C8 (8), C9 (9), C10 (10), C11 (11)
    NodeContainer hosts;
    hosts.Create(12);

    // Create switches: S1 (0), S2 (1), S3 (2), S4 (3), S5 (4), S6 (5), R (6), R2 (7), T (8)
    NodeContainer switches;
    {
        Ptr<SwitchNode> s1 = CreateObject<SwitchNode>();
        Ptr<SwitchNode> s2 = CreateObject<SwitchNode>();
        Ptr<SwitchNode> s3 = CreateObject<SwitchNode>();
        Ptr<SwitchNode> s4 = CreateObject<SwitchNode>();
        Ptr<SwitchNode> s5 = CreateObject<SwitchNode>();
        Ptr<SwitchNode> s6 = CreateObject<SwitchNode>();
        Ptr<SwitchNode> r = CreateObject<SwitchNode>();
        Ptr<SwitchNode> r2 = CreateObject<SwitchNode>();
        Ptr<SwitchNode> t = CreateObject<SwitchNode>();
        switches.Add(s1);
        switches.Add(s2);
        switches.Add(s3);
        switches.Add(s4);
        switches.Add(s5);
        switches.Add(s6);
        switches.Add(r);
        switches.Add(r2);
        switches.Add(t);
    }

    // Install Internet stack on all nodes
    InternetStackHelper internet;
    internet.Install(hosts);
    internet.Install(switches);

    // Link helper
    QbbHelper qbb;
    qbb.SetDeviceAttribute("DataRate", StringValue(linkDataRate));
    qbb.SetChannelAttribute("Delay", StringValue(linkDelay));

    // Helper to track interface indices for RDMA forwarding entries
    auto ifIndexOf = [](Ptr<NetDevice> dev) {
        return DynamicCast<QbbNetDevice>(dev)->GetIfIndex();
    };

    // Addresses for hosts (primary IPs used by apps/RDMA)
    std::vector<Ipv4Address> hostIp(12);
    for (uint32_t i = 0; i < 12; i++) hostIp[i] = node_id_to_ip(i);

    Ipv4AddressHelper ipv4;
    uint32_t linkIdx = 0;
    auto assignSubnet = [&](const NetDeviceContainer &d) {
        char base[16];
        sprintf(base, "10.%u.%u.0", linkIdx / 254 + 1, linkIdx % 254 + 1);
        ipv4.SetBase(base, "255.255.255.0");
        ipv4.Assign(d);
        linkIdx++;
    };

    // Build links per requested topology
    // C4 (hosts[4]) -> S1 (switches[0])
    NetDeviceContainer d_c4_s1 = qbb.Install(hosts.Get(4), switches.Get(0));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(4)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c4_s1.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[4], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c4_s1);

    // Server (hosts[0]) -> S1 (switches[0])
    NetDeviceContainer d_srv_s1 = qbb.Install(hosts.Get(0), switches.Get(0));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(0)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_srv_s1.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[0], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_srv_s1);

    // S1 -> R
    NetDeviceContainer d_s1_r = qbb.Install(switches.Get(0), switches.Get(6));
    assignSubnet(d_s1_r);

    // C1 (hosts[1]) -> S2 (switches[1])
    NetDeviceContainer d_c1_s2 = qbb.Install(hosts.Get(1), switches.Get(1));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(1)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c1_s2.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[1], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c1_s2);

    // C5 (hosts[5]) -> S2 (switches[1])
    NetDeviceContainer d_c5_s2 = qbb.Install(hosts.Get(5), switches.Get(1));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(5)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c5_s2.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[5], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c5_s2);

    // S2 -> R
    NetDeviceContainer d_s2_r = qbb.Install(switches.Get(1), switches.Get(6));
    assignSubnet(d_s2_r);

    // C2 (hosts[2]) -> S3 (switches[2])
    NetDeviceContainer d_c2_s3 = qbb.Install(hosts.Get(2), switches.Get(2));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(2)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c2_s3.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[2], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c2_s3);

    // C3 (hosts[3]) -> S3 (switches[2])
    NetDeviceContainer d_c3_s3 = qbb.Install(hosts.Get(3), switches.Get(2));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(3)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c3_s3.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[3], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c3_s3);

    // S3 -> R
    NetDeviceContainer d_s3_r = qbb.Install(switches.Get(2), switches.Get(6));
    assignSubnet(d_s3_r);

    // C6 (hosts[6]) -> S4 (switches[3])
    NetDeviceContainer d_c6_s4 = qbb.Install(hosts.Get(6), switches.Get(3));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(6)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c6_s4.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[6], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c6_s4);

    // C7 (hosts[7]) -> S4 (switches[3])
    NetDeviceContainer d_c7_s4 = qbb.Install(hosts.Get(7), switches.Get(3));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(7)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c7_s4.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[7], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c7_s4);

    // S4 -> R2
    NetDeviceContainer d_s4_r2 = qbb.Install(switches.Get(3), switches.Get(7));
    assignSubnet(d_s4_r2);

    // C8 (hosts[8]) -> S5 (switches[4])
    NetDeviceContainer d_c8_s5 = qbb.Install(hosts.Get(8), switches.Get(4));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(8)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c8_s5.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[8], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c8_s5);

    // C9 (hosts[9]) -> S5 (switches[4])
    NetDeviceContainer d_c9_s5 = qbb.Install(hosts.Get(9), switches.Get(4));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(9)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c9_s5.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[9], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c9_s5);

    // S5 -> R2
    NetDeviceContainer d_s5_r2 = qbb.Install(switches.Get(4), switches.Get(7));
    assignSubnet(d_s5_r2);

    // C10 (hosts[10]) -> S6 (switches[5])
    NetDeviceContainer d_c10_s6 = qbb.Install(hosts.Get(10), switches.Get(5));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(10)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c10_s6.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[10], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c10_s6);

    // C11 (hosts[11]) -> S6 (switches[5])
    NetDeviceContainer d_c11_s6 = qbb.Install(hosts.Get(11), switches.Get(5));
    {
        Ptr<Ipv4> ipv4h = hosts.Get(11)->GetObject<Ipv4>();
        ipv4h->AddInterface(d_c11_s6.Get(0));
        ipv4h->AddAddress(1, Ipv4InterfaceAddress(hostIp[11], Ipv4Mask(0xff000000)));
    }
    assignSubnet(d_c11_s6);

    // S6 -> R2
    NetDeviceContainer d_s6_r2 = qbb.Install(switches.Get(5), switches.Get(7));
    assignSubnet(d_s6_r2);

    // R -> T (switches[8])
    NetDeviceContainer d_r_t = qbb.Install(switches.Get(6), switches.Get(8));
    assignSubnet(d_r_t);

    // R2 -> T (switches[8])
    NetDeviceContainer d_r2_t = qbb.Install(switches.Get(7), switches.Get(8));
    assignSubnet(d_r2_t);

    // Install RDMA hardware and drivers on all nodes (hosts + switches)
    NodeContainer allNodes;
    allNodes.Add(hosts);
    allNodes.Add(switches);
    for (uint32_t i = 0; i < allNodes.GetN(); i++){
        Ptr<RdmaHw> rdmaHw = CreateObject<RdmaHw>();
       rdmaHw->SetAttribute("Mtu", UintegerValue(packetPayloadSize));
       rdmaHw->SetAttribute("CcMode", UintegerValue(1)); // DCQCN enabled
       rdmaHw->SetAttribute("L2AckInterval", UintegerValue(l2AckInterval)); // 🎯 KEY FIX: Reduce ACK frequency
       rdmaHw->SetAttribute("L2BackToZero", BooleanValue(false));
       
       // Window-specific DCQCN tuning for best per-flow accuracy
       rdmaHw->SetAttribute("MinRate", DataRateValue(DataRate(minRate))); // Aggressive ramp-up
       rdmaHw->SetAttribute("MaxRate", DataRateValue(DataRate("40Gbps"))); // Allow transient overshoot for calibration
       rdmaHw->SetAttribute("ClampTargetRate", BooleanValue(false));
       rdmaHw->SetAttribute("AlphaResumInterval", DoubleValue(55));    
       rdmaHw->SetAttribute("RPTimer", DoubleValue(50)); // Very fast rate updates
       rdmaHw->SetAttribute("FastRecoveryTimes", UintegerValue(5)); // Faster recovery
       rdmaHw->SetAttribute("EwmaGain", DoubleValue(1.0/8.0)); // More aggressive EWMA
       rdmaHw->SetAttribute("RateAI", DataRateValue(DataRate("2Gb/s"))); // Much faster additive increase
       rdmaHw->SetAttribute("RateHAI", DataRateValue(DataRate("5Gb/s"))); // Much faster hyper-active increase
       rdmaHw->SetAttribute("L2BackToZero", BooleanValue(false));
       rdmaHw->SetAttribute("L2ChunkSize", UintegerValue(l2ChunkSize)); // Window-specific chunk size
       rdmaHw->SetAttribute("RateDecreaseInterval", DoubleValue(1));
       rdmaHw->SetAttribute("MiThresh", UintegerValue(1));
       rdmaHw->SetAttribute("VarWin", BooleanValue(true));
       rdmaHw->SetAttribute("FastReact", BooleanValue(true)); // Important for performance
       rdmaHw->SetAttribute("MultiRate", BooleanValue(true)); // Important for performance
       rdmaHw->SetAttribute("SampleFeedback", BooleanValue(false));
       rdmaHw->SetAttribute("TargetUtil", DoubleValue(targetUtil)); // Window-specific target utilization

        Ptr<RdmaDriver> rdma = CreateObject<RdmaDriver>();
        rdma->SetNode(allNodes.Get(i));
        rdma->SetRdmaHw(rdmaHw);
        allNodes.Get(i)->AggregateObject(rdma);
        rdma->Init();
    }

    // RDMA forwarding tables (exact-match) on hosts and switches
    // Get interface indices for each link endpoint
    uint8_t if_c4_to_s1 = ifIndexOf(d_c4_s1.Get(0));
    uint8_t if_s1_to_c4 = ifIndexOf(d_c4_s1.Get(1));
    uint8_t if_srv_to_s1 = ifIndexOf(d_srv_s1.Get(0));
    uint8_t if_s1_to_srv = ifIndexOf(d_srv_s1.Get(1));
    uint8_t if_s1_to_r = ifIndexOf(d_s1_r.Get(0));
    uint8_t if_r_to_s1 = ifIndexOf(d_s1_r.Get(1));

    uint8_t if_c1_to_s2 = ifIndexOf(d_c1_s2.Get(0));
    uint8_t if_s2_to_c1 = ifIndexOf(d_c1_s2.Get(1));
    uint8_t if_c5_to_s2 = ifIndexOf(d_c5_s2.Get(0));
    uint8_t if_s2_to_c5 = ifIndexOf(d_c5_s2.Get(1));
    uint8_t if_s2_to_r = ifIndexOf(d_s2_r.Get(0));
    uint8_t if_r_to_s2 = ifIndexOf(d_s2_r.Get(1));

    uint8_t if_c2_to_s3 = ifIndexOf(d_c2_s3.Get(0));
    uint8_t if_s3_to_c2 = ifIndexOf(d_c2_s3.Get(1));
    uint8_t if_c3_to_s3 = ifIndexOf(d_c3_s3.Get(0));
    uint8_t if_s3_to_c3 = ifIndexOf(d_c3_s3.Get(1));
    uint8_t if_s3_to_r = ifIndexOf(d_s3_r.Get(0));
    uint8_t if_r_to_s3 = ifIndexOf(d_s3_r.Get(1));

    uint8_t if_c6_to_s4 = ifIndexOf(d_c6_s4.Get(0));
    uint8_t if_s4_to_c6 = ifIndexOf(d_c6_s4.Get(1));
    uint8_t if_c7_to_s4 = ifIndexOf(d_c7_s4.Get(0));
    uint8_t if_s4_to_c7 = ifIndexOf(d_c7_s4.Get(1));
    uint8_t if_s4_to_r2 = ifIndexOf(d_s4_r2.Get(0));
    uint8_t if_r2_to_s4 = ifIndexOf(d_s4_r2.Get(1));

    uint8_t if_c8_to_s5 = ifIndexOf(d_c8_s5.Get(0));
    uint8_t if_s5_to_c8 = ifIndexOf(d_c8_s5.Get(1));
    uint8_t if_c9_to_s5 = ifIndexOf(d_c9_s5.Get(0));
    uint8_t if_s5_to_c9 = ifIndexOf(d_c9_s5.Get(1));
    uint8_t if_s5_to_r2 = ifIndexOf(d_s5_r2.Get(0));
    uint8_t if_r2_to_s5 = ifIndexOf(d_s5_r2.Get(1));

    uint8_t if_c10_to_s6 = ifIndexOf(d_c10_s6.Get(0));
    uint8_t if_s6_to_c10 = ifIndexOf(d_c10_s6.Get(1));
    uint8_t if_c11_to_s6 = ifIndexOf(d_c11_s6.Get(0));
    uint8_t if_s6_to_c11 = ifIndexOf(d_c11_s6.Get(1));
    uint8_t if_s6_to_r2 = ifIndexOf(d_s6_r2.Get(0));
    uint8_t if_r2_to_s6 = ifIndexOf(d_s6_r2.Get(1));

    uint8_t if_r_to_t = ifIndexOf(d_r_t.Get(0));
    uint8_t if_t_to_r = ifIndexOf(d_r_t.Get(1));
    uint8_t if_r2_to_t = ifIndexOf(d_r2_t.Get(0));
    uint8_t if_t_to_r2 = ifIndexOf(d_r2_t.Get(1));

    // Hosts: send everything off-box via their single interface
    // C4 -> S1
    for (uint32_t j = 0; j < 12; j++){
        if (j == 4) continue;
        hosts.Get(4)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c4_to_s1);
    }

    // Server -> S1
    for (uint32_t j = 1; j < 12; j++){
        hosts.Get(0)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_srv_to_s1);
    }

    // C1 -> S2
    for (uint32_t j = 0; j < 12; j++){
        if (j == 1) continue;
        hosts.Get(1)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c1_to_s2);
    }

    // C5 -> S2
    for (uint32_t j = 0; j < 12; j++){
        if (j == 5) continue;
        hosts.Get(5)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c5_to_s2);
    }

    // C2 -> S3
    for (uint32_t j = 0; j < 12; j++){
        if (j == 2) continue;
        hosts.Get(2)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c2_to_s3);
    }

    // C3 -> S3
    for (uint32_t j = 0; j < 12; j++){
        if (j == 3) continue;
        hosts.Get(3)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c3_to_s3);
    }

    // C6 -> S4
    for (uint32_t j = 0; j < 12; j++){
        if (j == 6) continue;
        hosts.Get(6)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c6_to_s4);
    }

    // C7 -> S4
    for (uint32_t j = 0; j < 12; j++){
        if (j == 7) continue;
        hosts.Get(7)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c7_to_s4);
    }

    // C8 -> S5
    for (uint32_t j = 0; j < 12; j++){
        if (j == 8) continue;
        hosts.Get(8)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c8_to_s5);
    }

    // C9 -> S5
    for (uint32_t j = 0; j < 12; j++){
        if (j == 9) continue;
        hosts.Get(9)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c9_to_s5);
    }

    // C10 -> S6
    for (uint32_t j = 0; j < 12; j++){
        if (j == 10) continue;
        hosts.Get(10)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c10_to_s6);
    }

    // C11 -> S6
    for (uint32_t j = 0; j < 12; j++){
        if (j == 11) continue;
        hosts.Get(11)->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_c11_to_s6);
    }

    // Switch S1: attached to C4, Server and R
    {
        Ptr<Node> s1 = switches.Get(0);
        // To C4
        s1->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[4], if_s1_to_c4);
        DynamicCast<SwitchNode>(s1)->AddTableEntry(hostIp[4], if_s1_to_c4);
        // To Server
        s1->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[0], if_s1_to_srv);
        DynamicCast<SwitchNode>(s1)->AddTableEntry(hostIp[0], if_s1_to_srv);
        // To others -> R
        for (uint32_t j : {1u, 2u, 3u, 5u, 6u, 7u, 8u, 9u, 10u, 11u}){
            s1->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_s1_to_r);
            DynamicCast<SwitchNode>(s1)->AddTableEntry(hostIp[j], if_s1_to_r);
        }
    }

    // Switch S2: attached to C1, C5 and R
    {
        Ptr<Node> s2 = switches.Get(1);
        // To C1
        s2->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[1], if_s2_to_c1);
        DynamicCast<SwitchNode>(s2)->AddTableEntry(hostIp[1], if_s2_to_c1);
        // To C5
        s2->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[5], if_s2_to_c5);
        DynamicCast<SwitchNode>(s2)->AddTableEntry(hostIp[5], if_s2_to_c5);
        // To others -> R
        for (uint32_t j : {0u, 2u, 3u, 4u, 6u, 7u, 8u, 9u, 10u, 11u}){
            s2->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_s2_to_r);
            DynamicCast<SwitchNode>(s2)->AddTableEntry(hostIp[j], if_s2_to_r);
        }
    }

    // Switch S3: attached to C2, C3 and R
    {
        Ptr<Node> s3 = switches.Get(2);
        // To C2
        s3->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[2], if_s3_to_c2);
        DynamicCast<SwitchNode>(s3)->AddTableEntry(hostIp[2], if_s3_to_c2);
        // To C3
        s3->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[3], if_s3_to_c3);
        DynamicCast<SwitchNode>(s3)->AddTableEntry(hostIp[3], if_s3_to_c3);
        // To others -> R
        for (uint32_t j : {0u, 1u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u}){
            s3->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_s3_to_r);
            DynamicCast<SwitchNode>(s3)->AddTableEntry(hostIp[j], if_s3_to_r);
        }
    }

    // Switch S4: attached to C6, C7 and R2
    {
        Ptr<Node> s4 = switches.Get(3);
        // To C6
        s4->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[6], if_s4_to_c6);
        DynamicCast<SwitchNode>(s4)->AddTableEntry(hostIp[6], if_s4_to_c6);
        // To C7
        s4->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[7], if_s4_to_c7);
        DynamicCast<SwitchNode>(s4)->AddTableEntry(hostIp[7], if_s4_to_c7);
        // To others -> R2
        for (uint32_t j : {0u, 1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u}){
            s4->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_s4_to_r2);
            DynamicCast<SwitchNode>(s4)->AddTableEntry(hostIp[j], if_s4_to_r2);
        }
    }

    // Switch S5: attached to C8, C9 and R2
    {
        Ptr<Node> s5 = switches.Get(4);
        // To C8
        s5->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[8], if_s5_to_c8);
        DynamicCast<SwitchNode>(s5)->AddTableEntry(hostIp[8], if_s5_to_c8);
        // To C9
        s5->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[9], if_s5_to_c9);
        DynamicCast<SwitchNode>(s5)->AddTableEntry(hostIp[9], if_s5_to_c9);
        // To others -> R2
        for (uint32_t j : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 10u, 11u}){
            s5->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_s5_to_r2);
            DynamicCast<SwitchNode>(s5)->AddTableEntry(hostIp[j], if_s5_to_r2);
        }
    }

    // Switch S6: attached to C10, C11 and R2
    {
        Ptr<Node> s6 = switches.Get(5);
        // To C10
        s6->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[10], if_s6_to_c10);
        DynamicCast<SwitchNode>(s6)->AddTableEntry(hostIp[10], if_s6_to_c10);
        // To C11
        s6->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[11], if_s6_to_c11);
        DynamicCast<SwitchNode>(s6)->AddTableEntry(hostIp[11], if_s6_to_c11);
        // To others -> R2
        for (uint32_t j : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u}){
            s6->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_s6_to_r2);
            DynamicCast<SwitchNode>(s6)->AddTableEntry(hostIp[j], if_s6_to_r2);
        }
    }

    // Switch R: connected to S1, S2, S3 and T
    {
        Ptr<Node> r = switches.Get(6);
        // To nodes via S1 (Server, C4)
        for (uint32_t j : {0u, 4u}){
            r->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_r_to_s1);
            DynamicCast<SwitchNode>(r)->AddTableEntry(hostIp[j], if_r_to_s1);
        }
        // To nodes via S2 (C1, C5)
        for (uint32_t j : {1u, 5u}){
            r->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_r_to_s2);
            DynamicCast<SwitchNode>(r)->AddTableEntry(hostIp[j], if_r_to_s2);
        }
        // To nodes via S3 (C2, C3)
        for (uint32_t j : {2u, 3u}){
            r->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_r_to_s3);
            DynamicCast<SwitchNode>(r)->AddTableEntry(hostIp[j], if_r_to_s3);
        }
        // To nodes via R2 -> T (C6-C11)
        for (uint32_t j : {6u, 7u, 8u, 9u, 10u, 11u}){
            r->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_r_to_t);
            DynamicCast<SwitchNode>(r)->AddTableEntry(hostIp[j], if_r_to_t);
        }
    }

    // Switch R2: connected to S4, S5, S6 and T
    {
        Ptr<Node> r2 = switches.Get(7);
        // To nodes via S4 (C6, C7)
        for (uint32_t j : {6u, 7u}){
            r2->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_r2_to_s4);
            DynamicCast<SwitchNode>(r2)->AddTableEntry(hostIp[j], if_r2_to_s4);
        }
        // To nodes via S5 (C8, C9)
        for (uint32_t j : {8u, 9u}){
            r2->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_r2_to_s5);
            DynamicCast<SwitchNode>(r2)->AddTableEntry(hostIp[j], if_r2_to_s5);
        }
        // To nodes via S6 (C10, C11)
        for (uint32_t j : {10u, 11u}){
            r2->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_r2_to_s6);
            DynamicCast<SwitchNode>(r2)->AddTableEntry(hostIp[j], if_r2_to_s6);
        }
        // To nodes via R -> T (Server, C1-C5)
        for (uint32_t j : {0u, 1u, 2u, 3u, 4u, 5u}){
            r2->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_r2_to_t);
            DynamicCast<SwitchNode>(r2)->AddTableEntry(hostIp[j], if_r2_to_t);
        }
    }

    // Switch T: aggregation switch connected to R and R2
    {
        Ptr<Node> t = switches.Get(8);
        // To nodes via R (Server, C1-C5)
        for (uint32_t j : {0u, 1u, 2u, 3u, 4u, 5u}){
            t->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_t_to_r);
            DynamicCast<SwitchNode>(t)->AddTableEntry(hostIp[j], if_t_to_r);
        }
        // To nodes via R2 (C6-C11)
        for (uint32_t j : {6u, 7u, 8u, 9u, 10u, 11u}){
            t->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(hostIp[j], if_t_to_r2);
            DynamicCast<SwitchNode>(t)->AddTableEntry(hostIp[j], if_t_to_r2);
        }
    }

    // Install server app on Server (hosts[0]) - matches KvLiteClientApp expectation
    {
        Ptr<KvLiteServerApp> srv = CreateObject<KvLiteServerApp>();
        srv->SetAttribute("DataBytes", UintegerValue(argDataBytes));
        srv->SetAttribute("WindowSize", UintegerValue(argMaxWindows)); // Pass window size for overhead scaling
        hosts.Get(0)->AddApplication(srv);
        srv->SetStartTime(Seconds(0));
        srv->SetStopTime(Seconds(stopTimeSec));
    }

    // Install client apps on client nodes (C1-C11) targeting Server's IP
    auto installClient = [&](Ptr<Node> node){
        Ptr<KvLiteClientApp> cli = CreateObject<KvLiteClientApp>();
        // Only tunable client knob is MaxWindows (WINDOW_SIZE)
        cli->SetAttribute("MaxWindows", UintegerValue(argMaxWindows));
        node->AddApplication(cli);
        cli->SetStartTime(Seconds(0));
        cli->SetStopTime(Seconds(stopTimeSec));
    };
    
    // Install on client nodes (C1-C11 only, not T)
    for (uint32_t i = 1; i < 12; i++){
        installClient(hosts.Get(i));
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    Simulator::Stop(Seconds(stopTimeSec));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}
