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

tensorboard --logdir /data2/lichenni/output_cc --port 8009 --bind_all

git add -A . ; git commit -m "For now, it lacks the functions to parse transformer texts. Additionally, it requires traces from the real run, which are then parsed via the PyTorch converter."; git push