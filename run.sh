watch -c nvidia-htop.py -c

git fetch origin main
git fetch origin arjunvb
git fetch origin tracker
git merge origin/arjunvb

git submodule update --init --recursive
git submodule update --recursive
git rm --cached xxx
git submodule status --recursive

ps aux | head -1; ps aux | grep ^lichenni| sort -rnk 4 | more

tensorboard --logdir /data2/lichenni/output_perflow/ --port 8009 --bind_all

git add -A . ; git commit -m "wait for res"; git push

>/dev/null

time run ../ckpts/model_llama.bin ../ckpts/model_mlp.bin ../ckpts/data_lr10Gbps_7 -b 10 -e 576 -n 7 -t 1 -f 30 -k 18000 -p 1 -c 0 -x 30 

# train
CUDA_VISIBLE_DEVICES=0,1 python main_train.py --train_config=./config/train_config_lstm_debug.yaml --mode=train --dir_input=/data2/lichenni/perflow_link --dir_output=/data2/lichenni/output_perflow --note debug

python main_train.py --train_config=./config/train_config_lstm_link.yaml --mode=train --dir_input=/data2/lichenni/perflow_link_size --dir_output=/data2/lichenni/output_perflow --note link_test

python main_train.py --train_config=./config/train_config_lstm_link.yaml --mode=train --dir_input=/data2/lichenni/perflow_link_size --dir_output=/data2/lichenni/output_perflow --note link_100m_lstm_best_rtt_remainsize

# path
CUDA_VISIBLE_DEVICES=0,1 python main_train.py --train_config=./config/train_config_lstm_path.yaml --mode=train --dir_input=/data2/lichenni/perflow_path_size --dir_output=/data2/lichenni/output_perflow --note path_512

CUDA_VISIBLE_DEVICES=0,1 python main_train.py --train_config=./config/train_config_lstm_path.yaml --mode=train --dir_input=/data2/lichenni/perflow_path --dir_output=/data2/lichenni/output_perflow --note path_1000000_gat

# topo
python main_train.py --train_config=./config/train_config_lstm_topo_debug.yaml --mode=train --dir_input=/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/eval_train_test --dir_output=/data2/lichenni/output_perflow --note debug

python main_train.py --train_config=./config/train_config_lstm_topo.yaml --mode=train --dir_input=/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data --dir_output=/data2/lichenni/output_perflow --note topo_gnn

python main_train.py --train_config=./config/train_config_lstm_topo_empirical.yaml --mode=train --dir_input=/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data_empirical --dir_output=/data2/lichenni/output_perflow --note topo_512_flowsim_input_empirical_remainsize

python main_train.py --train_config=./config/train_config_lstm_topo.yaml --mode=train --dir_input=/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/eval_train --dir_output=/data2/lichenni/output_perflow --note m4_queuegt

python main_train.py --test_config=./config/test_config_lstm_topo.yaml --mode=test --version_id 0 --dir_input=/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data --dir_output=/data2/lichenni/output_perflow --test_on_train --note=topo_256_flowsim

python main_train.py --test_config=./config/test_config_lstm_topo.yaml --mode=test --version_id 0 --dir_input=/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data --dir_output=/data2/lichenni/output_perflow --test_on_manual --note=topo_256_flowsim

python main_train.py --test_config=./config/test_config_lstm_topo.yaml --mode=test --version_id 0 --dir_input=/data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data_empirical --dir_output=/data2/lichenni/output_perflow --test_on_empirical --note=topo_256_flowsim

# test
python main_train.py --test_config=./config/test_config_lstm_link.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_link_size --dir_output=/data2/lichenni/output_perflow --test_on_train --note=link_flowsim_input_remainsize

python main_train.py --test_config=./config/test_config_lstm_link.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_link_size --dir_output=/data2/lichenni/output_perflow --test_on_manual --note=link_flowsim_input_remainsize

python main_train.py --test_config=./config/test_config_lstm_link.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_link_size_empirical --dir_output=/data2/lichenni/output_perflow --test_on_empirical --note=link_flowsim_input_remainsize

# path scenarios
python main_train.py --test_config=./config/test_config_lstm_path.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_path_size --dir_output=/data2/lichenni/output_perflow --test_on_train --note=path_flowsim_input_remainsize

python main_train.py --test_config=./config/test_config_lstm_path.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_path_size --dir_output=/data2/lichenni/output_perflow --test_on_manual --note=path_flowsim_input_remainsize

python main_train.py --test_config=./config/test_config_lstm_path.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_path_size_empirical --dir_output=/data2/lichenni/output_perflow --test_on_empirical --note=path_flowsim_input_remainsize

cargo run --release -- --root=./data_test --mixes spec/motivation.mix.json mlsys-test
cargo run --release -- --root=./data_test --mixes spec/motivation.mix.json ns3-config

cargo run --release -- --root=./data_test --mixes spec/test.mix.json mlsys-test
cargo run --release -- --root=./data_test --mixes spec/test.mix.json ns3-config

time python main_inference_link.py

https://github.com/kwzhao/High-Precision-Congestion-Control/compare/8ded7c2ec5dae18a72c53c268eee0a70df5a2964...01fdae82351980dba89a34b33d9680e2eae44855

https://github.com/liecn/per-flow-sim/compare/c6a4a7aeb191a7b924be19d87d2c79d7384f4a06...617b5e4907f1422d04783bdc2178f106aab78ce7

ulimit -n 65536


https://github.com/kwzhao/High-Precision-Congestion-Control/compare/13958423c9b7e666b8b51bdb889816ec3f52d79a...8f1f1becb0de9d3add3176e55f73dec032aa123b



# gen sampled topology

# when updating the sampled topology, remember to update the tracing node ids in the config file projects/per-flow-sim/High-Precision-Congestion-Control/ns-3.39/mix_m3/trace.txt

## step-0:
cd /data1/lichenni/projects/per-flow-sim/High-Precision-Congestion-Control/traffic_gen
python traffic_gen_synthetic_dist.py --output /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/distributions/synthetic

## step-1: 
cd /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/spec
dhall-to-json --file cluster_1_to_1_eval_train.dhall > cluster_1_to_1_eval_train.json
dhall-to-json --file cluster_2_to_1_eval_train.dhall > cluster_2_to_1_eval_train.json
dhall-to-json --file cluster_4_to_1_eval_train.dhall > cluster_4_to_1_eval_train.json

dhall-to-json --file cluster_1_to_1_eval.dhall > cluster_1_to_1_eval.json
dhall-to-json --file cluster_2_to_1_eval.dhall > cluster_2_to_1_eval.json
dhall-to-json --file cluster_4_to_1_eval.dhall > cluster_4_to_1_eval.json

dhall-to-json --file cluster_1_to_1_eval_large.dhall > cluster_1_to_1_eval_large.json
dhall-to-json --file cluster_2_to_1_eval_large.dhall > cluster_2_to_1_eval_large.json
dhall-to-json --file cluster_4_to_1_eval_large.dhall > cluster_4_to_1_eval_large.json

## step-2: 
cd /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/src/bin
cargo run --bin contiguousify -- /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/spec/cluster_4_to_1_eval_train.json

cargo run --bin contiguousify -- /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/spec/cluster_1_to_1_eval.json

cargo run --bin contiguousify -- /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/spec/cluster_1_to_1_eval_large.json

# gen spatial matrix for fat-tree topology
cd /data1/lichenni/projects/parsimon-eval-all/workload/src
cargo run --bin downsample -- --spatial /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_c.json --nr-pods 2 --nr-tors-per-pod 4 --out /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_c_2_4.json

cargo run --bin downsample -- --spatial /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_c.json --nr-pods 4 --nr-tors-per-pod 16 --out /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_c_large.json

cargo run --bin downsample -- --spatial /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_b.json --nr-pods 4 --nr-tors-per-pod 16 --out /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_b_large.json

cargo run --bin downsample -- --spatial /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_a.json --nr-pods 4 --nr-tors-per-pod 16 --out /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_a_large.json

cargo run --bin downsample -- --spatial /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_c.json --nr-pods 8 --nr-tors-per-pod 16 --out /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_c_8_16.json

cargo run --bin downsample -- --spatial /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_b.json --nr-pods 8 --nr-tors-per-pod 16 --out /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_b_8_16.json

cargo run --bin downsample -- --spatial /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_a.json --nr-pods 4 --nr-tors-per-pod 16 --out /data1/lichenni/projects/per-flow-sim/parsimon-eval/workload/spatials/cluster_a_large.json

# gen configs
cd /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/spec/
python gen_mix_space.py

cd /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/src/

cargo run --bin gen_mixes -- --input ../spec/dctcp_empirical.mixspace.json --count 100 --output ../spec/dctcp_empirical.mix.json

cargo run --bin gen_mixes -- --input ../spec/dctcp_sync.mixspace.json --count 2000 --output ../spec/dctcp_sync.mix.json

cargo run --bin gen_mixes -- --input ../spec/dctcp_eval.mixspace.json --count 100 --output ../spec/dctcp_eval.mix.json

cargo run --bin gen_mixes -- --input ../spec/dctcp_eval_large.mixspace.json --count 100 --output ../spec/dctcp_eval_large.mix.json

# run exps fig 8
cd /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8
cargo run --release -- --root=./data --mixes spec/dctcp_sync.mix.json ns3
cargo run --release -- --root=./data_empirical --mixes spec/dctcp_empirical.mix.json ns3
cargo run --release -- --root=./data_eval --mixes spec/dctcp_eval.mix.json ns3
cargo run --release -- --root=./data_eval_large --mixes spec/dctcp_eval_large.mix.json ns3
cargo run --release -- --root=./data_test_config --mixes spec/test_config.mix.json ns3
cargo run --release -- --root=./eval_train --mixes spec/eval_train.mix.json ns3
cargo run --release -- --root=./eval_train_test --mixes spec/eval_train_test.mix.json ns3
cargo run --release -- --root=./eval_test --mixes spec/eval_test.mix.json ns3
cargo run --release -- --root=./eval_test --mixes spec/0.mix.json ns3

cargo run --release -- --root=./data_test_config --mixes spec/0.mix.json ns3
cargo run --release -- --root=./test --mixes spec/0.mix.json ns3

# run exps fig 7
cd /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_7
cargo run --release -- --root=./data --mix spec/0.mix.json ns3
cargo run --release -- --root=./data --mix spec/1.mix.json ns3
cargo run --release -- --root=./data --mix spec/2.mix.json ns3

# git large file
git filter-branch --tree-filter 'rm -f plot_simulation.ipynb' HEAD

python run_m4.py --root /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data_test_config/3/ns3 --base_rtt 14400 --topo topology --trace flows --bw 10 --bfsz 20 --fwin 10000 --shard_cc 0 --random_seed 0 --enable_pfc 1 --cc dctcp --param_1 20 --param_2 0 --enable_tr 0 --enable_debug 0 --max_inflight_flows 0 > /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data_test_config/3/ns3/log_sim.txt

python run_m4_post.py --shard 0 -p topology_flows --output_dir /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data/0/ns3 --cc dctcp --shard_cc 0 --enable_tr 0 --max_inflight_flows 0 > /data1/lichenni/projects/per-flow-sim/parsimon-eval/expts/fig_8/data/0/ns3/log_post.txt 2>&1

# run inference for Anton

# the config file is in per-flow-sim/config/test_config_lstm_topo_cplusplus.yaml
## 1. run python inference
cd /data1/lichenni/projects/per-flow-sim
python main_inference_topo_cplusplus.py

## 2. run c++ inference
cd /data1/lichenni/projects/per-flow-sim/inference/build
cmake ..
make

cd /data1/lichenni/projects/per-flow-sim/inference/python
python main_inference.py