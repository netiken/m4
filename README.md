# **m4: A Learned Flow-level Network Simulator**

This repository provides scripts and instructions to replicate the experiments from our paper, *m4: A Learned Flow-level Network Simulator.* It includes all necessary tools to reproduce the experimental results documented in Sections 5.2 and 5.3 of the paper.

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

2. Run the evaluation script to replicate results from Sections 5.2 and 5.4:
   ```bash
   jupyter notebook plot_eval.ipynb
   ```

---

## **Setup and Installation**

Ensure you have the following installed:
- **Python 3**
- **Rust & Cargo**
- **gcc-9**

### **Install Dependencies**
1. Set up Python environment:
   ```bash
   conda env create -f environment.yml
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

4. Set up ns-3 for data generation:
   ```bash
   cd High-Precision-Congestion-Control/ns-3.39
   ./configure
   ```

---

## **Running Experiments from Scratch**

The pre-trained checkpoints for the full m4 pipeline are available in the `XXX` directory. You can use them directly or train your own model (see [Training Your Own Model](#training-your-own-model)).

### **Replicating Paper Results**
#### **Section 5.2**
```bash
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json ns3
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json mlsys
```
Then, visualize the results using:
```bash
jupyter notebook plot_eval.ipynb
```

#### **Section 5.3**
```bash
cd parsimon-eval/expts/fig_7
cargo run --release -- --root=./data --mix spec/0.mix.json ns3
cargo run --release -- --root=./data --mix spec/1.mix.json ns3
cargo run --release -- --root=./data --mix spec/2.mix.json ns3

cargo run --release -- --root=./data --mix spec/0.mix.json mlsys
cargo run --release -- --root=./data --mix spec/1.mix.json mlsys
cargo run --release -- --root=./data --mix spec/2.mix.json mlsys
```
Then, visualize the results using:
```bash
jupyter notebook plot_eval.ipynb
```

#### **Section 5.4**
```bash
cd parsimon-eval/expts/fig_8
cargo run --release -- --root=./eval_app --mixes spec/eval_app.mix.json ns3
cargo run --release -- --root=./eval_app --mixes spec/eval_app.mix.json mlsys
```
Then, visualize the results using:
```bash
jupyter notebook plot_eval.ipynb
```

---

## **Training Your Own Model**

To train a new model, follow these steps:

1. **Generate training data**:
   ```bash
   cd parsimon-eval/expts/fig_8
   cargo run --release -- --root={dir_to_data} --mixes={config_for_sim_scenarios} ns3
   ```
   Example:
   ```bash
   cargo run --release -- --root=./eval_train --mixes spec/eval_train.mix.json ns3
   ```

2. **Train the model**:
   - Ensure you are in the correct Python environment.
   - Modify `config/train_config_lstm_topo.yaml` if needed.
   - Run:
     ```bash
     cd m4
     python main_train.py --train_config={path_to_config_file} --mode=train --dir_input={dir_to_save_data} --dir_output={dir_to_save_ckpts} --note={note}
     ```
   Example:
   ```bash
   python main_train.py --train_config=./config/train_config_lstm_topo.yaml --mode=train --dir_input=./parsimon-eval/expts/fig_8/eval_train --dir_output=/data2/lichenni/output_perflow --note m4
   ```

---

## **Repository Structure**
```
├── config                          # Configuration files for training and testing m4
├── High-Precision-Congestion-Control # HPCC repository for data generation
├── parsimon-eval                   # Scripts to reproduce m4 experiments and comparisons
├── util                             # Utility functions for m4, including data loaders and ML model implementations
└── main_train.py                    # Main script for training and testing m4
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