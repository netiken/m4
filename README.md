# m4: Modeling Flow-level Network Dynamics from Data

This GitHub repository houses the scripts and guidance needed to replicate the experiments presented in our paper, "m4: Modeling Flow-level Network Dynamics from Data". It offers all necessary tools to reproduce the experimental results documented in sections 5.2, 5.3, and 5.4 of our study.

## Contents

- [m4: Modeling Flow-level Network Dynamics from Data](#m4-precise-estimation-of-flow-level-performance-via-machine-learning)
  - [Contents](#contents)
  - [Setup Instructions](#setup-instructions)
- [Repository Structure](#repository-structure)
- [Citation Information](#citation-information)
- [Acknowledgments](#acknowledgments)
- [Getting in Touch](#getting-in-touch)

## Setup Instructions

Before you begin, ensure you have installed: Python 3, Rust, Cargo, gcc-9, and gcc-5. Use `environment.yml` conda environment files for Python setup, and follow additional instructions for other packages.

```bash
conda env create -f environment.yml
```

1. To install m4, execute: 
```bash
git clone https://github.com/netiken/m4.git
cd m4
```

2. To initialize the submodules, including parsimon, parsimon-eval, and HPCC:

```bash
git submodule update --init --recursive
```

4. For setting up the ns-3 for data generation, follow the detailed instructions in `parsimon/backends/High-Precision-Congestion-Control/simulation/README.md`:

```bash
cd High-Precision-Congestion-Control/ns-3.39
./configure
```

5. For training the model, ensure you're using the Python 3 environment and configure settings in `config/train_config_path.yaml`. Then execute:

```bash
cd m4
python main_path.py --train_config=./config/train_config_path.yaml --mode=train --dir_input={dir_to_save_data} --dir_output={dir_to_save_ckpts}

e.g., 
python main_path.py --train_config=./config/train_config_path.yaml --mode=train --dir_input=/data1/lichenni/m4/parsimon/backends/High-Precision-Congestion-Control/gen_path/data --dir_output=/data1/lichenni/m4/ckpts
```
Also, change the configurations for the dataset or model for your specific use case.

6. To create checkpoints for the end-to-end m4 pipeline:
```bash
cd m4
python gen_ckpt.py --dir_output={dir_to_save_ckpts}

e.g., 
python gen_ckpt.py --dir_output=/data1/lichenni/m4/ckpts
```
Note the checkpoints will be saved in the `ckpts` directory, one is for the Llama-2 model and the other is for the 2-layer MLP model.

7. To replicate paper results in Section 5.2, run the following in the `parsimon-eval/expts/fig_8` directory:

```bash
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json ns3
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json mlsys
```

Then reproduce the results in the script `plot_m4.ipynb`.

8. To replicate paper results in Section 5.3, run the following in the `parsimon-eval/expts/fig_7` directory:

```bash
cargo run --release -- --root=./data --mix spec/0.mix.json ns3
cargo run --release -- --root=./data --mix spec/1.mix.json ns3
cargo run --release -- --root=./data --mix spec/2.mix.json ns3
```

Then reproduce the results in the script `plot_m4.ipynb`.

9. To replicate paper results in Section 5.4, run the following in the `parsimon-eval/expts/fig_8` directory:

```bash
cargo run --release -- --root=./eval_test_app --mixes spec/eval_test_app.mix.json ns3
cargo run --release -- --root=./eval_test_app --mixes spec/eval_test_app.mix.json mlsys
```

Then reproduce the results in the script `plot_m4.ipynb`.

# Repository Structure

```bash
├── ckpts          # Checkpoints of Llama-2 and 2-layer MLP used in m4
├── clibs          # C libraries for running the path-level simulation in m4
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
    author = {Li, Chenning and Nasr-Esfahany, Arash and Zhao, Kevin and Noorbakhsh, Kimia and Goyal, Prateesh and Alizadeh, Mohammad and Anderson, Thomas},
    title = {m4: Modeling Flow-level Network Dynamics from Data},
    year = {2024},
}
```

# Acknowledgments

Special thanks to Kevin Zhao and Thomas Anderson for their insights shared in the NSDI'23 paper [Scalable Tail Latency Estimation for Data Center Networks](https://www.usenix.org/conference/nsdi23/presentation/zhao-kevin). The source codes can be found in [Parsimon](https://github.com/netiken/parsimon).

# Getting in Touch
For further inquiries, reach out to Chenning Li at lichenni@mit.edu

