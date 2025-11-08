/*
 * Lightweight KV common types (skeleton only)
 */
#pragma once

#include <cstdint>
#include <string>
#include "ns3/core-module.h"
#include "ns3/network-module.h"

namespace ns3 {

// Centralized constants for the simplified KV-lite apps
namespace kvlite {
    static const uint32_t KVL_DEFAULT_SERVER_DPORT = 4000;
    static const uint32_t KVL_PRIORITY_GROUP = 3;
    static const uint32_t KVL_CLIENT_REQUEST_BYTES = 17;
    static const uint64_t KVL_CLIENT_START_DELAY_NS = 0;
    static const uint64_t KVL_CLIENT_SEND_DELAY_NS = 2500; // ns - reduced for faster processing
    static const uint32_t KVL_CLIENT_MAX_REQUESTS = 650;
    static const uint32_t KVL_DEFAULT_MAX_WINDOWS = 1;
    static const uint32_t KVL_HANDSHAKE_REQ_BYTES = 10;
    static const uint32_t KVL_SERVER_SMALL_RESP_BYTES = 41;
    static const uint64_t KVL_SERVER_OVERHEAD_BASE_NS = 500000; // 500μs base overhead
    static const uint64_t KVL_SERVER_OVERHEAD_PER_WINDOW_NS = 1000000; // +1ms per window size
    // This models server-side queuing: overhead = base + (window * per_window)
    // Window 1: 0.5ms + 1*1ms = 1.5ms
    // Window 2: 0.5ms + 2*1ms = 2.5ms  
    // Window 4: 0.5ms + 4*1ms = 4.5ms
    // Matches real testbed UD scaling: 0.63ms → 2.89ms → 4.31ms
    static const uint32_t KVL_CLIENT_BASE_SPORT = 10000;
    static const uint32_t KVL_SERVER_BASE_SPORT = 10001;
    static const uint32_t KVL_SERVER_BASE_IP = 0x0b000001; // 11.0.0.1 (matches node_id_to_ip host octet)
    static const uint32_t KVL_SERVER_NODE_ID = 0; // stable server NodeList ID

    // Distinct port offsets per flow kind to keep reqId identical across flows
    // while avoiding (dip,sport,pg) QP key collisions
    static const uint16_t KVL_PORT_OFF_REQ = 0;
    static const uint16_t KVL_PORT_OFF_HS  = 15000;
    static const uint16_t KVL_PORT_OFF_RESP = 0;
    static const uint16_t KVL_PORT_OFF_DATA = 15000;

    // Request ID policy: either global 0-based or per-client offset
    static const bool KVL_REQID_USE_CLIENT_STRIDE = true;
    static const uint32_t KVL_REQID_CLIENT_STRIDE = 1000000; // base increment per client id
}

enum class KvLiteMsgType : uint8_t {
    REQ = 0,
    RESP = 1,
    HANDSHAKE = 2,
    DATA = 3,
    UNKNOWN = 255
};

struct KvLiteRequest
{
    uint64_t reqId = 0;
    uint32_t clientNodeId = 0;
    uint32_t serverNodeId = 0;
    uint32_t requestBytes = 0;
    uint32_t responseBytes = 0;
    uint64_t sendNs = 0; // scheduled send time
    KvLiteMsgType type = KvLiteMsgType::UNKNOWN;
};

struct KvLiteResponse
{
    uint64_t reqId = 0;
    uint32_t clientNodeId = 0;
    uint32_t serverNodeId = 0;
    uint32_t responseBytes = 0;
    uint64_t completeNs = 0; // delivery time
    KvLiteMsgType type = KvLiteMsgType::UNKNOWN;
};

} // namespace ns3


