# **m4: A Learned Flow-level Network Simulator**

This repository provides scripts and instructions to replicate the experiments from our paper, [*m4: A Learned Flow-level Network Simulator.*](https://arxiv.org/pdf/2503.01770) It includes all necessary tools to reproduce the experimental results documented in Sections 5.2 and 5.6 of the paper.

## **Contents**

- [Quick Reproduction](#quick-reproduction)
- [Setup and Installation](#setup-and-installation)
- [Running Experiments from Scratch](#running-experiments-from-scratch)
- [Training Your Own Model](#training-your-own-model)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## **Quick Reproduction**
To quickly reproduce the results in the paper, follow these steps:

1. Clone the repository and initialize submodules:
   ```bash
   git clone https://github.com/netiken/m4.git
   cd m4
   git submodule update --init --recursive
   ```

2. **Install uv** (a fast Python package manager): Follow the installation guide at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

3. Set up the python environment:
   ```bash
   uv sync
   source .venv/bin/activate
   ```
4. Please run the notebook `plot_results.ipynb` to generate the paper figures from Sections 5.2 and 5.6.

---

## **Setup and Installation**

### **Install Dependencies**

1. Set up Python environment:
   ```bash
   uv sync
   source .venv/bin/activate  # Activate the virtual environment
   ```
   
   **Note**: You can either activate the environment as shown above, or use `uv run <command>` to run commands directly (e.g., `uv run python main_train.py`).

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

This section shows how to reproduce the experimental results from the paper using pre-trained models. The pre-trained checkpoints for the full m4 pipeline are available in the `checkpoints` directory. You can use them directly or train your own model (see [Training Your Own Model](#training-your-own-model)).


### **Replicating Paper Results**

This section shows how to reproduce the experimental results from the paper. You can either use our provided demo data or generate the full dataset yourself.

**For Quick Test (Small Scale):**
```bash
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./demo --mixes spec/0.mix.json --nr-flows 2000 --enable-train ns3
cargo run --release -- --root=./demo --mixes spec/0.mix.json --nr-flows 2000 --enable-train mlsys
```
Results will be saved in the `demo` directory.

**Note**: 
- Use the `--enable-train` flag for commands that need the training-specific ns-3 version (`ns-3.39`) to generate training datasets with packet traces.
- Use the `--enable-app` flag for application completion time scenarios (Appendix 1) to synchronize flow start times.


#### **Step 1: Generate Test Data**

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

#### **Step 2: Run Inference and Generate Results**  
TODO: add the instructions to run the test.

#### **Step 3: Visualize Results**
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

## **Integrating m4 into SimAI**

The `SimAI/` directory contains an integrated evaluation framework comparing three network simulation backends:

| Backend | Description | Accuracy | Speed | Use Case |
|---------|-------------|----------|-------|----------|
| **ns-3** | Packet-level simulator with full RDMA/DCTCP/PFC modeling | Highest (ground truth) | Slowest | Validation & accuracy benchmark |
| **flowSim** | Analytical simulator using max-min fairness | Medium | Fastest | Quick iteration & prototyping |
| **m4** | ML-based simulator (LSTM+GNN+MLP) with bottleneck correction | Medium-High | Fast | Production workloads & realistic scenarios |

### Quick Start

**1. Build all backends** (requires GCC-9):
```bash
cd SimAI
./scripts/build.sh -c ns3      # Build NS-3 backend
./scripts/build.sh -c flowsim  # Build FlowSim backend
./scripts/build.sh -c m4       # Build M4 backend
```

**2. Run a sweep experiment**:
```bash
./run_sweep.sh <backend> <N> <M>
```

**Parameters:**
- `backend`: Network simulator to use (`ns3`, `flowsim`, or `m4`)
- `N`: Number of GPUs with bottleneck links (out of 32 total GPUs)
- `M`: Bandwidth throttling ratio — each throttled GPU gets `400 Gbps / M` bandwidth

**Example Scenarios:**
```bash
# Scenario 1: Heavy bottleneck - 16 GPUs @ 50 Gbps, 16 GPUs @ 400 Gbps
./run_sweep.sh ns3 16 8

# Scenario 2: Medium bottleneck - 4 GPUs @ 200 Gbps, 28 GPUs @ 400 Gbps
./run_sweep.sh flowsim 4 2

# Scenario 3: Light bottleneck - 8 GPUs @ 100 Gbps, 24 GPUs @ 400 Gbps
./run_sweep.sh m4 8 4
```

3. Results are saved to `SimAI/results/<backend>_<N>_<M>/`, we provide the demo results in the `results_examples` directory

### Output Files

Each simulation generates an `EndToEnd.csv` file with workload-level performance metrics:

| Backend | Output Location | What's Measured |
|---------|-----------------|-----------------|
| **ns-3** | `results/ns3_<N>_<M>/EndToEnd.csv` | Packet-level accurate workload completion time with full congestion control simulation |
| **flowSim** | `results/flowsim_<N>_<M>/EndToEnd.csv` | Analytical workload completion time using max-min fair bandwidth sharing |
| **m4** | `results/m4_<N>_<M>/EndToEnd.csv` | ML-predicted workload completion time with bottleneck-aware correction |

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
├── SimAI/                         # SimAI integration with ns-3, FlowSim, and m4 backends
│   ├── astra-sim-alibabacloud/    # Core simulation framework
│   │   ├── astra-sim/             # AstraSim system layer
│   │   │   ├── network_frontend/  # Network backend implementations
│   │   │   │   ├── ns3/           # NS-3 packet-level simulator (ground truth)
│   │   │   │   ├── flowsim/       # FlowSim analytical simulator
│   │   │   │   └── m4/            # M4 ML-based simulator
│   │   │   └── system/            # System components (routing, collective ops)
│   │   ├── extern/                # NS-3 source code
│   │   ├── inputs/                # Configuration files and topologies
│   │   └── build.sh               # Build script for all backends
│   ├── example/                   # Example workloads and topologies
│   │   └── sweep/                 # Sweep experiment configurations
│   ├── scripts/                   # Build and run scripts
│   ├── results/                   # Simulation results (we provide the demo results in the `results_examples` directory)
│   └── run_sweep.sh               # Sweep experiment runner
├── util/                          # Utility functions for m4, including data loaders and ML model implementations
├── main_train.py                  # Main script for training and testing m4
└── plot_results.ipynb            # Jupyter notebook for visualizing results
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
