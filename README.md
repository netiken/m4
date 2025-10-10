# **m4: A Learned Flow-level Network Simulator**

This repository provides scripts and instructions to replicate the experiments from our paper, [*m4: A Learned Flow-level Network Simulator.*](https://arxiv.org/pdf/2503.01770) It includes all necessary tools to reproduce the experimental results documented in Sections 5.2 to 5.6 of the paper.

## **Contents**

- [Repository Structure](#repository-structure)
- [Quick Reproduction](#quick-reproduction)
- [Setup and Installation](#setup-and-installation)
- [Running Experiments from Scratch](#running-experiments-from-scratch)
  - [Section 5.2: Testbed Integration](#section-52-testbed-integration)
  - [Section 5.3: SimAI Integration](#section-53-simai-integration-experiments)
  - [Sections 5.4-5.6: m4 Evaluation](#sections-54-56-m4-evaluation-experiments)
- [Training Your Own Model](#training-your-own-model)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## **Repository Structure**
```
├── checkpoints/                    # Pre-trained model checkpoints
├── config/                         # Configuration files for training and testing m4
├── figs/                          # Generated figures and plots from experiments
├── High-Precision-Congestion-Control/ # HPCC repository for data generation
├── inference/                     # C++ inference engine for m4
├── parsimon-eval/                 # Scripts to reproduce m4 experiments and comparisons
├── results/                       # Experimental results and outputs
├── results_train/                 # Training results and outputs
├── testbed/                       # Testbed integration with ns-3, flowSim, and m4 backends
│   ├── eval_train/                # Evaluation training data
│   ├── results_train/             # Training results and outputs
│   └──  run_m4_post.py             # Post-processing script for testbed data
├── SimAI/                         # SimAI integration with UNISON, flowSim, and m4 backends
│   ├── astra-sim-alibabacloud/    # Core simulation framework
│   │   ├── astra-sim/             # AstraSim system layer
│   │   │   ├── network_frontend/  # Network backend implementations
│   │   │   │   ├── ns3/           # UNISON (ns-3) packet-level simulator
│   │   │   │   ├── flowsim/       # flowSim analytical simulator
│   │   │   │   └── m4/            # m4 ML-based simulator
│   │   │   └── system/            # System components (routing, collective ops)
│   │   ├── extern/                # ns-3 source code
│   │   └── build.sh               # Build script for all backends
│   ├── example/                   # Example workloads and topologies
│   │   ├── gray_failures/         # 105 pre-generated gray failure topology files
│   │   │   └── gray_topo_N{2-16}_R{4-10}.txt  # Topology files for N degraded GPUs, R reduction factor
│   │   ├── microAllReduce.txt     # AllReduce collective workload
│   │   └── SimAI.conf             # ns-3 configuration
│   ├── scripts/                   # Build and run scripts
│   ├── results_gray_failures/     # Pre-computed gray failure results (315 simulations)
│   │   └── n_{N}_r_{R}_{backend}/ # Individual scenario results (ns3/flowsim/m4)
│   ├── gray_failure_run_sweep.py  # Gray failure sweep runner
│   ├── gray_failure_plot_results.py # Generate evaluation plots (6 figures)
│   └── gray_failure_topo_viz.py   # Topology visualization tool
├── util/                          # Utility functions for m4, including data loaders and ML model implementations
├── main_train.py                  # Main script for training and testing m4
└── plot_results.ipynb            # Jupyter notebook for visualizing results
```

---

## **Quick Reproduction**

To quickly reproduce the results in the paper, follow these steps:

**1. Clone the repository and initialize submodules:**
```bash
git clone https://github.com/netiken/m4.git
cd m4
git submodule update --init --recursive
```

**2. Set up Python environment:**

- **Install uv** (a fast Python package manager): Follow the installation guide at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

- **Set up Python environment:**
   ```bash
   uv sync
   source .venv/bin/activate  # Activate the virtual environment!
   ```

**3. Reproduce paper results:**
- **Section 5.3** (SimAI Integration): Check pre-computed results in `SimAI/results_examples/` and run the notebook `gray_failure_plot_results.ipynb` to generate paper figures
- **Sections 5.4-5.6** (m4 Evaluation): Run the notebook `plot_results.ipynb` to generate paper figures

---

## **Setup and Installation**

1. Always activate the python environment before running any commands:
   ```bash
   uv sync
   source .venv/bin/activate  # Activate the virtual environment!
   ```
   
2. Install Rust and Cargo:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup install nightly
   rustup default nightly
   ```

3. Install gcc-9:
   ```bash
   sudo apt-get install gcc-9 g++-9
   ```

4. Set up ns-3 (for training dataset with packet traces) and UNISON (for fast simulation) for data generation:
   ```bash
   cd High-Precision-Congestion-Control/UNISON-for-ns-3
   ./configure.sh
   ./ns3 run 'scratch/third mix/config_test.txt'
   cd ../ns-3.39
   ./configure.sh
   ./ns3 run 'scratch/third mix/config_test.txt'
   ```

---

## **Running Experiments from Scratch**

This section shows how to reproduce the experimental results from the paper using pre-trained models. The pre-trained checkpoints are available in the `checkpoints/` directory.

### **Section 5.2: Testbed Integration**

TODO for Om: add the instructions to run the testbed experiments.
<!-- The `testbed/` directory contains the code to run the experiments on the testbed.

Run the data processing script:
```bash
python testbed/run_m4_post.py
```

Run the training and testing scripts:
```bash
# training
python main_train.py --train_config=./config/train_config_testbed.yaml
# testing
python main_train.py --test_config=./config/test_config_testbed.yaml
``` -->

### **Section 5.3: SimAI Integration Experiments**

The `SimAI/` directory contains an integrated evaluation framework with three network simulation backends: **UNISON (ns-3)** , **flowSim** , and **m4** .

#### Build Backends

Build all three backends (requires GCC-9):

```bash
cd SimAI
./scripts/build.sh -c ns3      # Build UNISON (ns-3) backend
./scripts/build.sh -c flowsim  # Build flowSim backend
./scripts/build.sh -c m4       # Build m4 backend (requires CUDA)
```

#### Gray Failure Evaluation

We evaluate all three backends under **gray failure** conditions—scenarios where network components experience partial performance degradation rather than complete failures. This mimics real-world datacenter issues like cable aging, thermal throttling, or partial switch failures.

**Gray Failure Topologies:**

The repository includes **105 pre-generated topologies** in `example/gray_failures/` covering a comprehensive parameter sweep:
- **N ∈ {2, 3, ..., 16}**: Number of degraded GPUs (6%-50% of 32-GPU cluster)
- **R ∈ {4, 5, ..., 10}**: Bandwidth reduction factor (degraded links operate at 1/R capacity, i.e., 75%-90% bandwidth loss)

**Run Gray Failure Sweep:**

Note: Pre-computed results for all 315 simulations (3 backends × 105 scenarios) are available in `results_gray_failures/`. Running the sweep script will overwrite the pre-computed results.

```bash
# Run all scenarios for a specific backend
python gray_failure_run_sweep.py ns3      # UNISON (packet-level ground truth)
python gray_failure_run_sweep.py flowsim  # flowSim (analytical)
python gray_failure_run_sweep.py m4       # m4 (ML-based, uses GPU auto-detection)

# Run a single scenario (N=8 degraded GPUs, R=4 bandwidth reduction)
python gray_failure_run_sweep.py m4 --n 8 --r 4
```

**Visualize Results:**

Generate all evaluation plots (CDFs, runtime comparison, MAE analysis, scatter plots):
```bash
python gray_failure_plot_results.py
```

This produces 6 figures in the `SimAI/` directory:
- `gray_failure_errors.png` — CDF of error magnitudes
- `gray_failure_signed_errors.png` — CDF of signed errors (showing bias)
- `gray_failure_runtimes.png` — Runtime comparison across backends
- `gray_failure_mae_by_n.png` — Mean error vs. number of degraded GPUs
- `gray_failure_mae_by_r.png` — Mean error vs. bandwidth reduction factor
- `gray_failure_scatter_n8.png` — Completion time analysis for N=8

**Visualize Network Topology:**

Generate a visualization of the 32-GPU datacenter topology structure:
```bash
python gray_failure_topo_viz.py
```

This produces `simai_topo_groups.png` showing the hierarchical network topology with NVSwitch and rail switch layers.

---

### **Sections 5.4-5.6: m4 Evaluation Experiments**

Reproduce m4's accuracy evaluation across diverse network scenarios using pre-trained models.

TODO for Anton: add the instructions to run the flowSim and m4.

#### Quick Test (Small Scale)

For a quick test with reduced dataset size:

```bash
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./demo --mixes spec/0.mix.json --nr-flows 2000 --enable-train ns3
cargo run --release -- --root=./demo --mixes spec/0.mix.json --nr-flows 2000 --enable-train mlsys
```

Results will be saved in the `demo/` directory.

**Flags:**
- `--enable-train` — Use training-specific ns-3 version (`ns-3.39`) for packet traces
- `--enable-app` — Synchronize flow start times for application completion scenarios (Appendix 1)

#### Full Evaluation

**Option A: Use Demo Data (Recommended for Quick Start)**
We provide pre-generated demo data in the `parsimon-eval/expts/fig_8/eval_test_demo` directory.

**Option B: Generate Full Dataset**
Or you can generate the complete dataset yourself:

**For Section 5.4 (Large-scale evaluation):**
```bash
cd parsimon-eval/expts/fig_7
cargo run --release -- --root=./data --mixes spec/eval_test.mix.json ns3
cargo run --release -- --root=./data --mixes spec/eval_test.mix_large.json ns3
cargo run --release -- --root=./data --mixes spec/eval_test.mix.json mlsys
cargo run --release -- --root=./data --mixes spec/eval_test.mix_large.json mlsys
```
Results will be saved in the `data` directory.

**For Section 5.5 (Flow-level evaluation):**
```bash
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json --nr-flows 20000 ns3
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json --nr-flows 20000 mlsys
```
Results will be saved in the `eval_test` directory.

**For Appendix 1 (Application completion time):**
```bash
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./eval_app --mixes spec/eval_app.mix.json --nr-flows 20000 --enable-app ns3
cargo run --release -- --root=./eval_app --mixes spec/eval_app.mix.json --nr-flows 20000 --enable-app mlsys
```
Results will be saved in the `eval_app` directory.

#### Visualize Results
After completing the data generation and inference steps above, create the paper figures in the notebook `plot_results.ipynb`.

---


## **Training Your Own Model**

This section shows how to train and test your own m4 model from scratch. Follow these steps in order:

### **Step 1: Prepare Training Data**

**Option A: Use Demo Data (Recommended for Quick Start)**
We provide pre-generated demo training data in the `parsimon-eval/expts/fig_8/eval_train_demo` directory.

**Option B: Generate Full Training Dataset** 
Or you can generate the complete training dataset yourself:
   ```bash
   cd parsimon-eval/expts/fig_8
   cargo run --release -- --root={dir_to_data} --mixes={config_for_sim_scenarios} --enable-train ns3
   ```
   Example:
   ```bash
   cargo run --release -- --root=./eval_train --mixes spec/eval_train.mix.json --nr-flows 2000 --enable-train ns3
   ```

### **Step 2: Train the Model**
Train the neural network using the generated or demo training data:
   - Ensure you are in the correct Python environment.
   - Modify `config/train_config.yaml` if needed.
   - Run:
     ```bash
     cd m4
     uv run python main_train.py --train_config={path_to_config_file} --mode=train --dir_input={dir_to_save_data} --dir_output={dir_to_save_ckpts} --note={note}
     ```
   Example:
   ```bash
   # train on demo data
   uv run python main_train.py
   # train on the simulation data used in the paper
   uv run python main_train.py --train_config=./config/train_config.yaml --mode=train --dir_input=./parsimon-eval/expts/fig_8/eval_train --dir_output=./results_train --note m4
   ```

   Note: You can also use tensorboard to visualize the training process:
   ```bash
   uv run tensorboard --logdir ./results_train/ --port 8009 --bind_all
   ```
   Then, you can open the tensorboard in your browser following the instructions in the terminal.

### **Step 3: Test the Model**
Validate your trained model using the training data to check performance:
   - Ensure you are in the correct Python environment.
   - Modify `config/test_config.yaml` if needed.
   - Run:
     ```bash
     cd m4
     uv run python main_train.py --mode=test --test_config={path_to_config_file} --dir_input={dir_to_save_data} --dir_output={dir_to_save_results} --note={note}
     ```
   Example:
   ```bash
   # test on the demo data
   uv run python main_train.py --mode=test
   # validate on the simulation data used in the paper
   uv run python main_train.py --mode=test --test_config=./config/test_config.yaml --dir_input=./parsimon-eval/expts/fig_8/eval_train --dir_output=./results_train --note m4
   ```
---

## **Citation**
If you find our work useful, please cite our paper:
```bibtex
@inproceedings{m4,
    author = {Li, Chenning and Zabreyko, Anton and Nasr-Esfahany, Arash and Zhao, Kevin and Goyal, Prateesh and Alizadeh, Mohammad and Anderson, Thomas},
    title = {m4: A Learned Flow-level Network Simulator},
    year = {2025},
}
```

---

## **Acknowledgments**
We extend special thanks to Kevin Zhao and Thomas Anderson for their insights in the NSDI'23 paper *Scalable Tail Latency Estimation for Data Center Networks.* Their source code is available in [Parsimon](https://github.com/netiken/parsimon).

---

## **Contact**
For further inquiries, reach out to **Chenning Li** at:  
📧 **lichenni@mit.edu**
