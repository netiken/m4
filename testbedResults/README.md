# M4 Network Simulation Testbed

## Setup
```bash
cd /data1/lichenni/m4
uv pip install -e .
```

## Run Network Backends

### M4
```bash
cd backends/m4
python run_sweep.py --topologies 12
```

### FlowSim
```bash
cd backends/flowsim
python run_sweep.py --topologies 12
```

### NS3/UNISON
```bash
cd backends/UNISON
python run_sweep.py --topologies 12
```

## Analyze Results
```bash
python analyze.py
```

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
./main 12 1024008
```

### NS3/UNISON
```bash
cd backends/UNISON
source /data1/lichenni/m4/.venv/bin/activate
./ns3 clean
./ns3 configure --build-profile=optimized --enable-mtp
./ns3 build
./ns3 run "twelve --maxWindows=2 --dataBytes=1024008"
```