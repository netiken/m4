# m4: A Learned Flow-level Network Simulator

This GitHub repository houses the scripts and guidance needed to replicate the experiments presented in our paper, "m4: A Learned Flow-level Network Simulator". It offers all necessary tools to reproduce the experimental results documented in sections 5.2 and 5.3 of our study.

## Contents

- [Quick Reproduction](#quick-reproduction)
- [From Scratch](#from-scratch)
- [Train your own model](#train-your-own-model)
- [Repository Structure](#repository-structure)
- [Citation Information](#citation-information)
- [Acknowledgments](#acknowledgments)
- [Getting in Touch](#getting-in-touch)

First, clone the repository and install the necessary dependencies. To install m3, execute: 
```bash
git clone https://github.com/netiken/m4.git
cd m4
# Initialize the submodules, including parsimon-eval and HPCC
git submodule update --init --recursive
```
## Quick Reproduction
The following steps provide a quick guide to reproduce the results in the paper.

1. To replicate paper results in Section 5.2 and 5.4, run the notebook `plot_eval.ipynb`.

## From Scratch
1. Ensure you have installed: Python 3, Rust, Cargo, and gcc-9. Use `environment.yml` conda environment files for Python setup, and follow additional instructions for other packages.

```bash
conda env create -f environment.yml
```

```bash
# Install Rust and Cargo, https://www.rust-lang.org/tools/install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# setup the path and check the installation using rustc --version. Then switch to nightly version
rustup install nightly
rustup default nightly
```
```bash
# Install gcc-9
sudo apt-get install gcc-9 g++-9
```

2. For setting up the ns-3 for data generation, follow the instructions below:

```bash
cd High-Precision-Congestion-Control/ns-3.39
./configure
```

3. The checkpotins for the end-to-end m4 pipeline are available in the `ckpts` directory. You can use them directly for the following steps. Please refer to the section [Train your own model](#train-your-own-model) for training the model from scratch.

3. To replicate paper results in Section 5.2, run the following in the `parsimon-eval/expts/fig_8` directory:

```bash
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json ns3
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json mlsys
```

Then reproduce the results in the script `plot_eval.ipynb`.

4. To replicate paper results in Section 5.3, run the following in the `parsimon-eval/expts/fig_7` directory:

```bash
cargo run --release -- --root=./data --mix spec/0.mix.json ns3
cargo run --release -- --root=./data --mix spec/1.mix.json ns3
cargo run --release -- --root=./data --mix spec/2.mix.json ns3
```

Then reproduce the results in the script `plot_eval.ipynb`.

5. To replicate paper results in Section 5.4, run the following in the `parsimon-eval/expts/fig_8` directory:

```bash
cargo run --release -- --root=./eval_test_app --mixes spec/eval_test_app.mix.json ns3
cargo run --release -- --root=./eval_test_app --mixes spec/eval_test_app.mix.json mlsys
```

Then reproduce the results in the script `plot_eval.ipynb`.

# Train your own model

* Please use the demo data in `data` in the main directory to test the training process.

1. To generate data for training and testing your own model, run:

```bash
cd ./parsimon-eval/expts/fig_8
cargo run --release -- --root={dir_to_data} --mixes={config_for_sim_scenarios} ns3

e.g., 
cargo run --release -- --root=./eval_train --mixes spec/eval_train.mix.json ns3
```

2. For training the model, ensure you're using the Python 3 environment and configure settings in `config/train_config_lstm_topo.yaml`. Then execute:

```bash
cd m4
python main_path.py --train_config={path_to_config_file} --mode=train --dir_input={dir_to_save_data} --dir_output={dir_to_save_ckpts} --note={note}

e.g., 
python main_train.py --train_config=./config/train_config_lstm_topo.yaml --mode=train --dir_input=./parsimon-eval/expts/fig_8/eval_train --dir_output=/data2/lichenni/output_perflow --note m4
```
Also, change the configurations for the dataset or model for your specific use case.

# Repository Structure

```bash
├── config         # Configuration files for training and testing m4
├── High-Precision-Congestion-Control   # HPCC repository for data generation
├── parsimon-eval  # Scripts to reproduce m4 experiments and comparisons
├── util           # Utility functions for m4, including data loader and ML model implementations
└── main_train.py   # Main script for training and testing m4
```

# Citation Information
If our work assists in your research, kindly cite our paper as follows:
```bibtex
@inproceedings{m4,
    author = {Li, Chenning and Zabreyko, Anton and Nasr-Esfahany, Arash and Zhao, Kevin and Goyal, Prateesh and Alizadeh, Mohammad and Anderson, Thomas},
    title = {m4: A Learned Flow-level Network Simulator},
    year = {2025},
}
```

# Acknowledgments

Special thanks to Kevin Zhao and Thomas Anderson for their insights shared in the NSDI'23 paper [Scalable Tail Latency Estimation for Data Center Networks](https://www.usenix.org/conference/nsdi23/presentation/zhao-kevin). The source codes can be found in [Parsimon](https://github.com/netiken/parsimon).

# Getting in Touch
For further inquiries, reach out to Chenning Li at lichenni@mit.edu

