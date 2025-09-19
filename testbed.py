#!/usr/bin/env python3
import subprocess
import shutil
import pathlib
import time
import datetime as dt
import os
import sys
from typing import List


WINDOW_SIZES: List[int] = [2] #1,2,16
RDMA_SIZES: List[int] = [
    #102408, 204808, 256008, 307208, 409608,
    #512008, 665608, 768008, 921608, 
    #1024008#, # 
    1048584,
]
RDMA_TITLES = [
    #"100", "200", "250", "300", "400", "500", "650", "750", "900", 
    #"1000" #, 
    "10000"
]
RDMA_TITLES = {rdma: title for rdma, title in zip(RDMA_SIZES, RDMA_TITLES)}


def run_cmd(cmd: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> int:
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
        return proc.returncode


def ensure_built(src_dir: pathlib.Path, binary: pathlib.Path) -> None:
    # Always run make to be safe; adjust if you want to skip when binary is fresh
    print("[build] Running make...")
    rc = run_cmd(["make", "-j"], cwd=src_dir, stdout_path=src_dir / "build_stdout.txt", stderr_path=src_dir / "build_stderr.txt")
    if rc != 0:
        print(f"[build] make failed with code {rc}. See build_stderr.txt")
        sys.exit(rc)
    if not binary.exists():
        print(f"[build] Binary not found at {binary}")
        sys.exit(1)


def clean_logs(src_dir: pathlib.Path) -> None:
    # Remove leftover logs from previous runs to avoid mixing
    for pat in ["client_*.log", "server.log", "flows.txt", "flowsim_output.txt"]:
        for p in src_dir.glob(pat):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def collect_outputs(src_dir: pathlib.Path, dest_dir: pathlib.Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Copy known outputs if present
    for name in ["flows.txt", "server.log", "flowsim_output.txt"]:
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, dest_dir / p.name)
    for p in src_dir.glob("client_*.log"):
        shutil.copy2(p, dest_dir / p.name)


def main() -> None:
    # flowsim_dir = pathlib.Path("/home/ubuntu/flowsim/flowsim")
    flowsim_dir = pathlib.Path("/data1/lichenni/projects/om/m4/inference")
    binary = flowsim_dir / "build/no_flowsim"
    #os.system("rm -rf " + str(flowsim_dir / "sweeps"))
    #os.system("mkdir -p " + str(flowsim_dir / "sweeps"))
    results_root = flowsim_dir / "sweeps" 
    results_root.mkdir(parents=True, exist_ok=True)

    # ensure_built(flowsim_dir, binary)

    total = len(WINDOW_SIZES) * len(RDMA_SIZES)
    i = 0
    for ws in WINDOW_SIZES:
        for rdma in RDMA_SIZES:
            i += 1
            rdma_name = rdma
            run_tag = f"{RDMA_TITLES[rdma_name]}_{ws}"
            run_dir = results_root / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"[run {i}/{total}] {run_tag}")

            # Clean any previous outputs to avoid mixing
            clean_logs(flowsim_dir)

            # Execute the binary with HERD-mode args: argv[1]=window_size, argv[2]=resp_rdma_bytes
            stdout_path = run_dir / "stdout.txt"
            stderr_path = run_dir / "stderr.txt"
            rc = run_cmd([str(binary), str(ws), str(rdma)], cwd=flowsim_dir, stdout_path=stdout_path, stderr_path=stderr_path)
            if rc != 0:
                print(f"  -> non-zero exit {rc}; see {stderr_path}")

            # Collect run artifacts
            collect_outputs(flowsim_dir, run_dir)

            # Small pause to keep things tidy
            time.sleep(0.1)

    print(f"[done] Results under {results_root}")


if __name__ == "__main__":
    main()


