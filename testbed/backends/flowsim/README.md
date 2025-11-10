# FlowSim Testbed Sweeps

FlowSim is the C++ baseline we compare against the real testbed. This directory contains the simulator source, helper scripts, and the sweep outputs that `new_scenarios/analyzev2.py` consumes.

## Requirements
- GCC/Clang toolchain with C++17 support
- `make`
- Python 3.9+ (only for the helper scripts)

## Local Quickstart
```bash
cd flowsim
make -j                   # build ./main
python3 run_sweep.py      # regenerate sweeps_4/…
```

`run_sweep.py` rebuilds the simulator, runs the hard-coded window/RDMA combinations, and writes results to `sweeps_4/<scenario>/`. Adjust `WINDOW_SIZES`, `RDMA_SIZES`, or add CLI flags as needed for different evaluation sets.

Once the sweeps finish:

```bash
cd ../new_scenarios
python3 analyzev2.py --local-base expirements_4
# or explicitly:
python3 analyzev2.py --source flowsim=../flowsim/sweeps_4:flowsim_output.txt
```

## Common Commands
- Rebuild only: `make clean && make -j`
- Single run without sweeping: `./main <window_size> <rdma_bytes>`
- Inspect latest logs: `less sweeps_4/<scenario>/stderr.txt`

## Directory Notes
- `sweeps_4/` – FlowSim outputs in the same naming scheme as the real scenarios (e.g. `1000_2/flowsim_output.txt`).
- `run_sweep.py` – Local automation with no hard-coded absolute paths.
- `flowsim_output.txt` format matches the `[ud]/[rdma]` lines expected by the analyzer.
