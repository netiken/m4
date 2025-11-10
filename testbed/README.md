# M4 Network Simulation Testbed

## Setup
```bash
cd /data1/lichenni/m4
uv pip install -e .
```

## Build Network Backends

Build all backends or specific ones:
```bash
# Build all backends (NS3, FlowSim, M4)
./build.sh all

# Build only NS3
./build.sh ns3

# Build only FlowSim
./build.sh flowsim

# Build only M4 (uses .venv for LibTorch)
./build.sh m4
```

## Run Simulations

Run all backends or specific ones with unified runner:
```bash
# Run both NS3 and FlowSim (with 32 parallel jobs by default)
python run.py all

# Run only NS3 (with 32 parallel jobs)
python run.py ns3

# Run only FlowSim (with 32 parallel jobs)
python run.py flowsim

# Run with custom number of parallel jobs
python run.py ns3 --jobs 16
python run.py flowsim --jobs 8
```

Results will be saved to:
- NS3: `eval_test/ns3/`
- FlowSim: `eval_test/flowsim/`
- Testbed (real-world data): `eval_test/testbed/`

## Analyze Results
```bash
python analyze.py
```

This will:
- Analyze all scenarios from `eval_test/`
- Generate plots in `results/`
- Display summary statistics

## Manual Testing (Optional)

### M4
```bash
cd backends/m4
mkdir -p build && cd build
cmake .. && make
./no_flowsim 12 1024008
```

### FlowSim
```bash
cd backends/flowsim
make
./main 1 1024008 12  # window=1, rdma_size=1024008, topology=12
```

### NS3/UNISON
```bash
cd backends/UNISON
./ns3 run "twelve --maxWindows=2 --dataBytes=1024008"
```