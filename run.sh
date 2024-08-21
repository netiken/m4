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
python main_train.py --train_config=./config/train_config_lstm_link.yaml --mode=train --dir_input=/data2/lichenni/perflow_link --dir_output=/data2/lichenni/output_perflow --note link_200000_abstime_gnn

CUDA_VISIBLE_DEVICES=0,1 python main_train.py --train_config=./config/train_config_lstm_path.yaml --mode=train --dir_input=/data2/lichenni/perflow_path --dir_output=/data2/lichenni/output_perflow --note path_1000000

CUDA_VISIBLE_DEVICES=0,1 python main_train.py --train_config=./config/train_config_lstm_path.yaml --mode=train --dir_input=/data2/lichenni/perflow_path --dir_output=/data2/lichenni/output_perflow --note path_1000000_gnn

# test
python main_train.py --test_config=./config/test_config_lstm_link.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_link_empirical --dir_output=/data2/lichenni/output_perflow --test_on_empirical --note=link_200000_abstime

python main_train.py --test_config=./config/test_config_lstm_link.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_link --dir_output=/data2/lichenni/output_perflow --note=link_200000_abstime --test_on_train

python main_train.py --test_config=./config/test_config_lstm_path.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_path_empirical --dir_output=/data2/lichenni/output_perflow --test_on_empirical --note=path_1000000

python main_train.py --test_config=./config/test_config_lstm_path.yaml --mode=test --version_id 0 --dir_input=/data2/lichenni/perflow_path --dir_output=/data2/lichenni/output_perflow --note=path_1000000 --test_on_train

cargo run --release -- --root=./data_test --mixes spec/motivation.mix.json mlsys-test
cargo run --release -- --root=./data_test --mixes spec/motivation.mix.json ns3-config

cargo run --release -- --root=./data_test --mixes spec/test.mix.json mlsys-test
cargo run --release -- --root=./data_test --mixes spec/test.mix.json ns3-config

time python main_inference_link.py

https://github.com/kwzhao/High-Precision-Congestion-Control/compare/8ded7c2ec5dae18a72c53c268eee0a70df5a2964...01fdae82351980dba89a34b33d9680e2eae44855