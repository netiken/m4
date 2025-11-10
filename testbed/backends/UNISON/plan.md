## Goal

Provide a minimal key–value (KV) application framework (client/server) that lets you program simple client and server logic while everything below is handled — and it DOES use the existing ns-3 network in `scratch/third.cc` to send real packets (segmented by `packet_payload_size`). The application should:

- Accept operations with explicit message sizes and exact send times
- Transmit synthetic payloads over the simulated network (global buffer reused; no real KV data)
- Compute and record application-level FCTs (request send → response delivered)
- Keep the abstraction clean so you only write client/server logic hooks

## Recommended Architecture (Real Traffic, Simple API)

- **Message over RDMA flows:**
  - Each request is a one-shot RDMA flow client→server with size `request_bytes`.
  - On delivery at the server, the server app optionally waits a processing delay, then issues a one-shot response RDMA flow server→client with size `response_bytes`.
  - Payload content is synthetic; we reuse a global static buffer to avoid allocations.
- **Two ns-3 Applications:**
  - `KvLiteClientApp`: reads workload and schedules requests at exact times; logs end-to-end FCTs when the response arrives.
  - `KvLiteServerApp`: on flow delivery, runs server logic hook and triggers the response flow back to the client.
- **Reuse third.cc network:**
  - Segmentation, queues, ECN/PFC, and congestion control all remain active via the existing RDMA path. Packets are actually sent through the simulated switches.

## Components to Add

1. **`KvLiteServerApp` (ns-3 Application)**
   - Attributes: `DefaultResponseBytes`, `ProcessingTimeNs`, `DefaultDport`, optional `PriorityGroup`.
   - Hooks into RDMA delivery: on request flow delivered to this server, call `OnRequest(req)` user hook, then schedule a response flow back to the client.
   - Uses a shared global buffer for payloads (size >= max(request_bytes, response_bytes)).

2. **`KvLiteClientApp` (ns-3 Application)**
   - Loads a workload (events with `t_ns`, `client`, `server`, `op`, `request_bytes`, `response_bytes`).
   - For each event, schedules a request flow at `t_ns` using `RdmaClientHelper` or direct `RdmaDriver` helpers.
   - Records send timestamp; when the matching response is delivered to the client, computes FCT and writes to `KV_TRACE_FILE`.
   - Enforces `MaxInflight` at the application layer.

3. **Flow/port coordination**
   - Reuse `serverAddress` and `portNumder` from `third.cc` to allocate unique ports per pair.
   - Map `req_id` → {client, server, send_time, request_bytes, response_bytes} to correlate response deliveries for FCT.

## Integration Points in the Current Codebase

- `scratch/third.cc`
  - Add `KV_LITE_MODE` (0/1). When `KV_LITE_MODE=1`, bypass `ScheduleFlowInputs()` and use the KV workload instead.
  - Install `KvLiteServerApp` on server nodes and `KvLiteClientApp` on client nodes appearing in the workload.
  - Keep all existing network setup (routing, buffers, ECN/PFC, CC). KV flows traverse the same network and generate packets.
  - Connect RDMA traces: keep existing `QpDelivered`/`QpComplete` tracing, and additionally notify the KV apps to trigger responses and mark completions.

## Detailed Plan

1. **Configuration additions** (parsed in `third.cc`):
   - `KV_LITE_MODE` (uint32): 0=off (default), 1=on
   - `KV_WORKLOAD_FILE` (string): NDJSON/CSV workload
   - `KV_TRACE_FILE` (string): per-op FCT output
   - `KV_MAX_INFLIGHT` (uint32): per-client inflight limit
   - `KV_SERVER_PROC_NS` (uint64): default server processing time per request
   - `KV_DEFAULT_DPORT` (uint32): default server port
   - `KV_PRIORITY_GROUP` (uint32): optional PG for flows

2. **Server bring-up**
   - Install `KvLiteServerApp` on all servers (or a subset). Attributes: `DefaultResponseBytes`, `ProcessingTimeNs`, `DefaultDport`, `PriorityGroup`.
   - Expose hook `OnRequest(const KvRequest& req, KvResponse& rsp)` for custom logic (e.g., response size, processing time). Default: echo with `DefaultResponseBytes`.

3. **Client bring-up**
   - Install `KvLiteClientApp` on clients in the workload. Attributes: `WorkloadFile`, `TraceFile`, `MaxInflight`, `DefaultDport`, `PriorityGroup`.
   - For each event, schedule a request flow at `t_ns` via `RdmaClientHelper` (size=`request_bytes`). Record send time.
   - The server app observes delivery (via RDMA trace or app callback) and triggers a response flow back to the client (size=`response_bytes`).
   - On client response delivery, compute FCT and write to `KV_TRACE_FILE`.

4. **Inflight control**
   - Maintain a per-client counter; if `MaxInflight` reached, queue events and release on completion (upon response delivery).

5. **Tracing**
   - Keep existing `qp_delivered`/`qp_finish` for low-level traces.
   - Add application-level file with lines:
     - `req_id op client server req_bytes rsp_bytes send_ns resp_recv_ns fct_ns status`

6. **Directory layout**
   - `scratch/kv-lite-common.h|cc` (request/response structs, global buffer, correlators)
   - `scratch/kv-lite-server-app.h|cc`
   - `scratch/kv-lite-client-app.h|cc`

## Minimal Changes in `scratch/third.cc`

- Parse `KV_LITE_MODE` and related KV config keys.
- If `KV_LITE_MODE==1`:
  - Skip `ScheduleFlowInputs()`.
  - Install `KvLiteServerApp` on servers and `KvLiteClientApp` on clients in the workload.
  - Connect RDMA traces to notify KV apps on deliveries so the server can trigger responses and the client can close FCTs.

## Example Workload (NDJSON)

```json
{ "t_ns": 1000, "client": 0, "server": 8, "op": "PUT", "request_bytes": 128, "response_bytes": 16 }
{ "t_ns": 2000, "client": 1, "server": 8, "op": "GET", "request_bytes": 32,  "response_bytes": 1024 }
{ "t_ns": 2500, "client": 0, "server": 9, "op": "PUT", "request_bytes": 64,  "response_bytes": 16 }
```

Keys can be UTF-8 strings; values can be supplied as a length (random payload generated by client) or an inline base64 field (optional).

## Message Sizes and Packetization

- Packets are actually sent over the simulated network. Segmentation is handled by the RDMA path using `packet_payload_size`.
- Payload content is synthetic (from a global buffer); only sizes matter for transmission and CC.

## Telemetry Outputs

- `KV_TRACE_FILE` (client): `req_id op client server req_bytes rsp_bytes send_ns complete_ns fct_ns model_fct_ns status`.
- Server logs (optional): request counters, processing time, simple store size.

## Migration Steps

1. Add KV config flags and bypass flow scheduling when `KV_LITE_MODE=1`.
2. Implement `KvLiteServerApp` and `KvLiteClientApp` that use RDMA flows with synthetic payloads.
3. Connect RDMA delivery traces to KV apps; validate a tiny workload and confirm FCT logging.
4. Add inflight limits and server processing delay.
5. Scale up workloads; optional: add PG selection, per-op response sizes via server hook.

## Build & Run (example)

- Build: normal project build (waf/cmake as used by this repo).
- Config example additions:
  - `KV_LITE_MODE 1`
  - `KV_WORKLOAD_FILE ./workloads/kv.ndjson`
  - `KV_TRACE_FILE ./results/kv_ops.txt`
  - `KV_MAX_INFLIGHT 128`
  - `KV_SERVER_PROC_NS 0`
  - `KV_DEFAULT_DPORT 4000`
  - `KV_PRIORITY_GROUP 3`

Run as usual with your existing configuration file; when `KV_MODE=1`, flows are driven entirely by the KV workload.

## Notes

- This design keeps things simple: no packets, no sockets/QPs, deterministic FCTs.
- You can program client/server logic via small hooks and ignore lower layers.
- If you later need realism, you can drop in a real-traffic transport behind the same interfaces.


