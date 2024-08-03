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

git add -A . ; git commit -m "new model"; git push

>/dev/null

time run ../ckpts/model_llama.bin ../ckpts/model_mlp.bin ../ckpts/data_lr10Gbps_7 -b 10 -e 576 -n 7 -t 1 -f 30 -k 18000 -p 1 -c 0 -x 30 


python main_train.py --train_config=./config/train_config_lstm.yaml --mode=train --dir_input=/data2/lichenni/path_perflow_link --dir_output=/data2/lichenni/output_perflow --note fct_lstm_balanced_gcn

python main_train.py --train_config=./config/train_config_transformer.yaml --mode=train --dir_input=/data2/lichenni/path_perflow_1k --dir_output=/data2/lichenni/output_perflow --note fct_transformer_noncausal_b

python main_train.py --test_config=./config/test_config_lstm.yaml --mode=test --note=fct_lstm_final_flow --version_id 0 --dir_input=/data2/lichenni/path_perflow_link_empirical --dir_output=/data2/lichenni/output_perflow --test_on_empirical

python main_train.py --test_config=./config/test_config_lstm.yaml --mode=test --note=fct_lstm_final_flow --version_id 0 --dir_input=/data2/lichenni/path_perflow_link --dir_output=/data2/lichenni/output_perflow --test_on_train

cargo run --release -- --root=./data_test --mixes spec/motivation.mix.json mlsys-test
cargo run --release -- --root=./data_test --mixes spec/motivation.mix.json ns3-config

cargo run --release -- --root=./data_test --mixes spec/test.mix.json mlsys-test
cargo run --release -- --root=./data_test --mixes spec/test.mix.json ns3-config

