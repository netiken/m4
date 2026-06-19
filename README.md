# **m4: A Learned Flow-level Network Backend**

This repository contains the public artifact for our paper, [*m4: A Learned Flow-level Network Backend*](https://arxiv.org/pdf/2503.01770). `m4` is a learned online flow-level backend for distributed-application simulators: it accepts flow requests, maintains state over a flow-link graph, and returns completion-time callbacks much faster than packet-level simulation while improving accuracy over analytical flow-level baselines.

The public release includes the core training and inference code, pretrained checkpoints, standalone simulation artifacts, and the SimAI integration. The RDMA hardware testbed artifact and raw testbed results are intentionally not included in this repository.

## **Contents**

- [Repository Structure](#repository-structure)
- [Artifact Scope](#artifact-scope)
- [Quick Reproduction](#quick-reproduction)
- [Setup and Installation](#setup-and-installation)
- [Running Experiments from Scratch](#running-experiments-from-scratch)
  - [SimAI Integration](#simai-integration)
  - [Standalone m4 Evaluation](#standalone-m4-evaluation)
  - [RDMA Testbed Results](#rdma-testbed-results)
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
├── SimAI/                         # SimAI integration with UNISON, flowSim, and m4 backends
│   ├── astra-sim-alibabacloud/    # Core simulation framework
│   │   ├── astra-sim/             # AstraSim system layer
│   │   │   ├── network_frontend/  # Network backend implementations
│   │   │   │   ├── ns3/           # UNISON (ns-3) packet-level simulator
│   │   │   │   ├── flowsim/       # flowSim analytical simulator
│   │   │   │   └── m4/            # m4 learned backend
│   │   │   └── system/            # System components (routing, collective ops)
│   │   ├── extern/                # ns-3 source code
│   │   └── build.sh               # Build script for all backends
│   ├── example/                   # Example workloads and topologies
│   │   ├── gray_failures/         # 105 pre-generated gray failure topology files
│   │   │   └── gray_topo_N{2-16}_R{4-10}.txt  # N degraded edge links, R reduction factor
│   │   ├── microAllReduce.txt     # AllReduce collective workload
│   │   └── SimAI.conf             # ns-3 configuration
│   ├── scripts/                   # Build and run scripts
│   ├── gray_failure_run_sweep.py  # Gray failure sweep runner
│   ├── gray_failure_plot_results.py # Generate gray-failure evaluation plots
│   └── gray_failure_topo_viz.py   # Topology visualization tool
├── util/                          # Utility functions for m4, including data loaders and ML model implementations
├── main_train.py                  # Main script for training and testing m4
└── plot_results.ipynb            # Jupyter notebook for visualizing results
```

---

## **Artifact Scope**

| Paper component | Public release status |
| --- | --- |
| Core learned backend, model code, checkpoints, and standalone simulation artifacts | Included |
| Standalone packet-simulator-labeled evaluation and ablations | Public artifacts and scripts are included through `results/`, `results_train/`, `parsimon-eval/`, and `plot_results.ipynb` |
| SimAI gray-failure integration | Included; topologies and runner scripts are provided, while generated sweep outputs are produced locally |
| RDMA hardware testbed integration and raw hardware results | Not included in this public release |

The paper also reports an RDMA hardware evaluation, but those hardware traces and testbed scripts are not part of the GitHub artifact.

---

## **Quick Reproduction**

To quickly inspect the public artifact and reproduce the included results:

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

**3. Use the included artifacts:**
- **Standalone simulation and ablations:** open `plot_results.ipynb`, which reads the included result files under `results/` and `results_train/`.
- **SimAI gray-failure evaluation:** build the desired SimAI backends, run `SimAI/gray_failure_run_sweep.py`, then run `SimAI/gray_failure_plot_results.py`.
- **RDMA testbed evaluation:** reported in the paper but intentionally excluded from this public repository.

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

This section describes the public experiment workflows that are included in this repository. The pre-trained checkpoints are available in the `checkpoints/` directory.

### **SimAI Integration**

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

The SimAI workflow evaluates gray-failure scenarios where network components experience partial performance degradation rather than complete failures.

**Gray Failure Topologies:**

The repository includes **105 pre-generated topologies** in `example/gray_failures/` covering a comprehensive parameter sweep:
- **N ∈ {2, 3, ..., 16}**: Number of degraded GPU-facing edge links (6%-50% of 32-GPU cluster)
- **R ∈ {4, 5, ..., 10}**: Bandwidth reduction factor (degraded links operate at 1/R capacity, i.e., 75%-90% bandwidth loss)

**Run Gray Failure Sweep:**

Generated sweep outputs are written under `SimAI/results_gray_failures/`. That directory is intentionally not part of the public GitHub release, so run the sweep before plotting if you start from a fresh clone.

```bash
# From the SimAI/ directory
# Run all scenarios for a specific backend
python gray_failure_run_sweep.py ns3      # UNISON (packet-level ground truth)
python gray_failure_run_sweep.py flowsim  # flowSim (analytical)
python gray_failure_run_sweep.py m4       # m4 (learned backend, uses GPU auto-detection)

# Run a single scenario (N=8 degraded edge links, R=4 bandwidth reduction)
python gray_failure_run_sweep.py m4 --n 8 --r 4
```

**Visualize Results:**

Generate all evaluation plots (CDFs, runtime comparison, MAE analysis, scatter plots):
```bash
python gray_failure_plot_results.py
```

This produces gray-failure evaluation plots in the `SimAI/` directory, including:
- `gray_failure_errors.png` — CDF of error magnitudes
- `gray_failure_signed_errors.png` — CDF of signed errors (showing bias)
- `gray_failure_runtimes.png` — Runtime comparison across backends
- `gray_failure_mae_by_n.png` — Mean error vs. number of degraded edge links
- `gray_failure_mae_by_r.png` — Mean error vs. bandwidth reduction factor
- `gray_failure_scatter_n8.png` — Completion time analysis for N=8

**Visualize Network Topology:**

Generate a visualization of the 32-GPU datacenter topology structure:
```bash
python gray_failure_topo_viz.py
```

This produces `simai_topo_groups.png` showing the hierarchical network topology with NVSwitch and rail switch layers.

### **Standalone m4 Evaluation**

Reproduce `m4`'s standalone accuracy, scaling, sensitivity, and ablation evaluation across diverse network scenarios using the included checkpoints and result files.

For the fastest path, run `plot_results.ipynb` from the repository root. The notebook reads the public result artifacts under `results/` and `results_train/` and regenerates the main standalone evaluation plots.

To regenerate simulation outputs from scratch, use the experiment drivers under `parsimon-eval/`:

- **Large-scale evaluation:**
```bash
# From the repository root
cd parsimon-eval/expts/fig_7
cargo run --release -- --root=./data --mixes spec/eval_test.mix.json ns3
cargo run --release -- --root=./data --mixes spec/eval_test.mix_large.json ns3
cargo run --release -- --root=./data --mixes spec/eval_test.mix.json mlsys
cargo run --release -- --root=./data --mixes spec/eval_test.mix_large.json mlsys
```
Results are saved in the `data` directory.

- **Flow-level evaluation:**
```bash
# From the repository root
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json --nr-flows 20000 ns3
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json --nr-flows 20000 mlsys
```
Results are saved in the `eval_test` directory.

- **Application completion-time evaluation:**
```bash
# From the repository root
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./eval_app --mixes spec/eval_app.mix.json --nr-flows 20000 --enable-app ns3
cargo run --release -- --root=./eval_app --mixes spec/eval_app.mix.json --nr-flows 20000 --enable-app mlsys
```
Results are saved in the `eval_app` directory.

#### Visualize Results
After completing the data generation and inference steps above, use `plot_results.ipynb` to inspect and regenerate the standalone plots from the available result files.

---

### **RDMA Testbed Results**

The paper also evaluates a checkpoint fine-tuned on a 4-host RDMA deployment and tested on 12-host hardware. That part of the evaluation depends on hardware traces and testbed integration code that are not included in this public GitHub release.

The public repository therefore does not contain a `testbed/` workflow.

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
@misc{m4,
    author = {Li, Chenning and Zabreyko, Anton and Nasr-Esfahany, Arash and Zhao, Kevin and Goyal, Prateesh and Alizadeh, Mohammad and Anderson, Thomas},
    title = {m4: A Learned Flow-level Network Backend},
    year = {2026},
    eprint = {2503.01770},
    archivePrefix = {arXiv},
}
```

---

## **Acknowledgments**
We extend special thanks to Kevin Zhao and Thomas Anderson for their insights in the NSDI'23 paper *Scalable Tail Latency Estimation for Data Center Networks.* Their source code is available in [Parsimon](https://github.com/netiken/parsimon).

---

## **Contact**
For further inquiries, reach out to **Chenning Li** at:  
**lichenni@mit.edu**
