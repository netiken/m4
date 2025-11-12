# M4 Network Simulation Testbed

Unified testbed for comparing M4 (ML-enhanced), FlowSim (flow-level), and NS3 (packet-level) network simulators against real hardware measurements.

## Available Backends

- **m4**: ML-enhanced simulator (LSTM + GNN)
- **flowsim**: Flow-level event-driven simulator
- **ns3**: Packet-level simulator (UNISON)
- **all**: Run all backends

## Directory Structure

```
eval_test/
├── testbed/     # Real hardware measurements (ground truth)
├── m4/          # M4 simulation results
├── flowsim/     # FlowSim simulation results
└── ns3/         # NS3 simulation results

results/         # Generated plots and accuracy summaries
```

## Common Commands

```bash
# Activate the uv environment, please cd to the root directory of the repository and run:
uv sync
source .venv/bin/activate

# Build specific backend
./build.sh m4|flowsim|ns3|all

# Run with options
python run.py m4|flowsim|ns3|all

# Analyze results
python analyze.py
```