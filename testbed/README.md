# M4 Network Simulation Testbed

Unified testbed for comparing M4 (ML-enhanced), FlowSim (flow-level), and NS3 (packet-level) network simulators against real hardware measurements.

## Quick Start

```bash
# 1. Build backends
./build.sh all

# 2. Run simulations (M4 example with 8 parallel jobs)
python run.py m4 --jobs 8

# 3. Analyze results
python analyze.py
```

Results are saved to `eval_test/{backend}/` and plots to `results/`.

## Available Backends

- **m4**: ML-enhanced simulator (LSTM + GNN)
- **flowsim**: Flow-level event-driven simulator
- **ns3**: Packet-level simulator (UNISON)
- **all**: Run all backends

## Common Commands

```bash
# Build specific backend
./build.sh m4|flowsim|ns3|all

# Run with options
python run.py m4 --jobs 16              # Parallel jobs (default: 32)
python run.py m4 --quick                # Quick test (4 scenarios)

# Analyze specific scenarios
python analyze.py --scenario 250_1      # Single scenario
python analyze.py --quick               # Quick test scenarios only
python analyze.py --no-plots            # Summary stats only
```

## Directory Structure

```
eval_test/
├── testbed/     # Real hardware measurements (ground truth)
├── m4/          # M4 simulation results
├── flowsim/     # FlowSim simulation results
└── ns3/         # NS3 simulation results

results/         # Generated plots and accuracy summaries
```