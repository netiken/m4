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

2. Set up the environment and run the evaluation script to replicate results from Sections 5.2 and 5.6:
   ```bash
   uv sync
   source .venv/bin/activate
   jupyter notebook plot_results.ipynb
   ```

---

## **Setup and Installation**

### **Install Dependencies**
1. **Install uv** (a fast Python package manager): Follow the installation guide at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

2. Set up Python environment:
   ```bash
   uv sync
   source .venv/bin/activate  # Activate the virtual environment
   ```
   
   **Note**: You can either activate the environment as shown above, or use `uv run <command>` to run commands directly (e.g., `uv run python main_train.py`).

3. Install Rust and Cargo:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup install nightly
   rustup default nightly
   ```

4. Install gcc-9:
   ```bash
   sudo apt-get install gcc-9 g++-9
   ```

5. Set up ns-3 for data generation:
   ```bash
   cd High-Precision-Congestion-Control/UNISON-for-ns-3
   ./configure.sh
   ./ns3 run 'scratch/third mix/config_test.txt'
   ```

---

## **Running Experiments from Scratch**

The pre-trained checkpoints for the full m4 pipeline are available in the `checkpoints` directory. You can use them directly or train your own model (see [Training Your Own Model](#training-your-own-model)).


### **Replicating Paper Results**

#### **Generate Test Data**

We provide the demo data in the `parsimon-eval/expts/fig_8/eval_test_demo` directory.

Or you can generate the data yourself.

##### **Section 5.4**
```bash
cd parsimon-eval/expts/fig_7
cargo run --release -- --root=./data --mixes spec/eval_test.mix.json ns3
cargo run --release -- --root=./data --mixes spec/eval_test.mix_large.json ns3
cargo run --release -- --root=./data --mixes spec/eval_test.mix.json mlsys
cargo run --release -- --root=./data --mixes spec/eval_test.mix_large.json mlsys
```
Then, you can find the results in the `data` directory.

#### **Section 5.5**
```bash
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json --nr-flows 20000 ns3
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json --nr-flows 20000 mlsys
```
Then, you can find the results in the `eval_test` directory.

#### **Appendix 1**
```bash
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./eval_app --mixes spec/eval_app.mix.json --nr-flows 20000 ns3
cargo run --release -- --root=./eval_app --mixes spec/eval_app.mix.json --nr-flows 20000 mlsys
```
Then, you can find the results in the `eval_app` directory.

#### **Run test**
TODO: add the instructions to run the test.

---


## **Training Your Own Model**

To train a new model, follow these steps:

1. **Generate training data**:
   We provide the demo data in the `parsimon-eval/expts/fig_8/eval_train_demo` directory.
   Or you can generate the data yourself.
   ```bash
   cd parsimon-eval/expts/fig_8
   cargo run --release -- --root={dir_to_data} --mixes={config_for_sim_scenarios} ns3
   ```
   Example:
   ```bash
   cargo run --release -- --root=./eval_train --mixes spec/eval_train.mix.json --nr-flows 2000 ns3
   ```

2. **Train the model**:
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

3. **Test the model**:
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
