# **m4: A Learned Flow-level Network Simulator**

This repository provides scripts and instructions to replicate the experiments from our paper, [*m4: A Learned Flow-level Network Simulator.*](https://arxiv.org/pdf/2503.01770) It includes all necessary tools to reproduce the experimental results documented in Sections 5.2 and 5.3 of the paper.

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
   conda activate m4
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
   cd High-Precision-Congestion-Control/UNISON-for-ns-3
   ./configure
   ```
   A quick test for the installation:
   ```bash
   # in High-Precision-Congestion-Control/UNISON-for-ns-3
   ./ns3 run 'scratch/third mix/config_test.txt'
   ```

---

## **Running Experiments from Scratch**

### **Replicating Table 4**
```bash
cd parsimon-eval/expts/fig_7
cargo run --release -- --root=./data_large --mixes spec/7.mix.json ns3
cargo run --release -- --root=./data_large --mixes spec/8.mix.json ns3
cargo run --release -- --root=./data_large --mixes spec/9.mix.json ns3
```
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
