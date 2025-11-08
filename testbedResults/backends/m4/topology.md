# Network Topology Description

## Overview
This topology represents a three-tier hierarchical data center network composed of:
- **1 Core Switch**
- **2 Aggregation Switches**
- **6 Top-of-Rack (ToR) Switches**
- **12 Servers**

All links operate at **10 Gbps**.

---

## Device Naming Convention
| Layer | Device Prefix | Range | Example |
|--------|----------------|--------|----------|
| Server | `serv` | `serv0` – `serv11` | `serv3` |
| ToR Switch | `tor` | `tor12` – `tor17` | `tor15` |
| Aggregation Switch | `agg` | `agg18` – `agg19` | `agg18` |
| Core Switch | `core` | `core20` | `core20` |

---

## Core Layer
- **Device:** `core20`
- **Role:** Provides backbone connectivity between aggregation switches.
- **Connections:**
  - 10 Gbps links to both aggregation switches (`agg18`, `agg19`).

---

## Aggregation Layer
- **Devices:** `agg18`, `agg19`
- **Role:** Aggregate traffic from ToR switches and provide uplinks to the core switch.
- **Connections:**
  - Each aggregation switch connects to the core (`core20`) via 10 Gbps.
  - Each aggregation switch connects downstream to three ToR switches (10 Gbps per link).

| Aggregation Switch | Connected ToR Switches |
|--------------------|------------------------|
| `agg18` | `tor12`, `tor13`, `tor14` |
| `agg19` | `tor15`, `tor16`, `tor17` |

---

## Access Layer (ToR Switches)
- **Devices:** `tor12` – `tor17`
- **Role:** Provide server access within each rack.
- **Connections:**
  - Each ToR connects to one aggregation switch.
  - Each ToR connects to **two servers** (10 Gbps per link).

| ToR Switch | Connected Servers |
|-------------|------------------|
| `tor12` | `serv0`, `serv1` |
| `tor13` | `serv2`, `serv3` |
| `tor14` | `serv4`, `serv5` |
| `tor15` | `serv6`, `serv7` |
| `tor16` | `serv8`, `serv9` |
| `tor17` | `serv10`, `serv11` |

---

## Connectivity Summary

| Layer | Devices | Count | Link Speed |
|--------|----------|--------|-------------|
| Core | `core20` | 1 | 10 Gbps |
| Aggregation | `agg18`, `agg19` | 2 | 10 Gbps |
| ToR | `tor12`–`tor17` | 6 | 10 Gbps |
| Servers | `serv0`–`serv11` | 12 | 10 Gbps |

---

## Topology Summary
This is a **two-stage fat-tree** (three-tier) topology designed for balanced bandwidth and redundancy:
- Each ToR connects to an aggregation switch via 10 Gbps.
- Aggregation switches connect to a single core for centralized routing.
- Each server has 10 Gbps uplink capacity through its ToR switch.
- The design ensures symmetrical, high-throughput connectivity across all layers.

---