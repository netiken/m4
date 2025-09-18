#!/usr/bin/env bash
source $(dirname $0)/../scripts/utils.sh
source $(dirname $0)/../scripts/mlx_env.sh
export HRD_REGISTRY_IP="specialnode.RDMA.fawn.apt.emulab.net"

drop_shm

num_server_threads=1

blue "Reset server QP registry"
sudo pkill memcached
memcached -l 0.0.0.0 1>/dev/null 2>/dev/null &
sleep 1

blue "Starting $num_server_threads server threads"

flags="
	--num_server_threads $num_server_threads \
	--dual_port 0 \
	--is_client 0 \
	--size 32 \
	--postlist 16
"

# Check for non-gdb mode
if [ "$#" -eq 0 ]; then
  sudo -E numactl --cpunodebind=0 --membind=0 ../build/ss-echo $flags
fi

# Check for gdb mode
if [ "$#" -eq 1 ]; then
  sudo -E gdb -ex run --args ../build/ss-echo $flags
fi
