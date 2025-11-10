sudo sysctl -w vm.nr_hugepages=2048
sudo sysctl -w kernel.shmmax=$((16*1024*1024*1024*1024))
sudo sysctl -w kernel.shmall=$((16*1024*1024*1024*1024/4096))
