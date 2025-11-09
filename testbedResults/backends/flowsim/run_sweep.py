#!/usr/bin/env python3
import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List


# Sweep dimensions for the HERD workload.
WINDOW_SIZES: List[int] = [1, 2, 4]  # Focus on these three as requested
RDMA_TITLES_BASE: Dict[int, str] = {
    102408: "100",
    256008: "250",
    307208: "300",
    409608: "400",
    512008: "500",
    665608: "650",
    768008: "750",
    921608: "900",
    1024008: "1000",
}
RDMA_SIZES: List[int] = list(RDMA_TITLES_BASE.keys())


# Repository root for FlowSim (script lives inside this directory).
ROOT_DIR = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments that control which topologies to sweep."""
    parser = argparse.ArgumentParser(description="Run FlowSim sweeps across window/RDMA/topology combos.")
    parser.add_argument(
        "--topologies",
        type=int,
        nargs="+",
        default=[12],
        choices=[1, 4, 12],
        help="Topology sizes to include (1=single link, 4=multi-client, 12=tree). Default: 1.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> int:
    """Spawn a subprocess and stream stdout/stderr into files."""
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
        return proc.returncode


def ensure_built(src_dir: pathlib.Path, binary: pathlib.Path) -> None:
    """Run make every time so the FlowSim binary stays up to date."""
    print("[build] Running make...")
    rc = run_cmd(["make", "-j"], cwd=src_dir, stdout_path=src_dir / "build_stdout.txt", stderr_path=src_dir / "build_stderr.txt")
    if rc != 0:
        print(f"[build] make failed with code {rc}. See build_stderr.txt")
        sys.exit(rc)
    if not binary.exists():
        print(f"[build] Binary not found at {binary}")
        sys.exit(1)


def clean_logs(src_dir: pathlib.Path) -> None:
    """Remove prior FlowSim artefacts so each run starts fresh."""
    for pattern in ["client_*.log", "server.log", "flows.txt", "flowsim_output.txt"]:
        for path in src_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def collect_outputs(src_dir: pathlib.Path, dest_dir: pathlib.Path) -> None:
    """Copy FlowSim outputs from a temp workspace into the sweep directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ["flows.txt", "server.log", "flowsim_output.txt"]:
        candidate = src_dir / name
        if candidate.exists():
            shutil.copy2(candidate, dest_dir / candidate.name)
    for log in src_dir.glob("client_*.log"):
        shutil.copy2(log, dest_dir / log.name)


def results_dir_for_topology(base: pathlib.Path, topo: int) -> pathlib.Path:
    """Return the sweep directory name that matches downstream tooling."""
    if topo == 12:
        return base / "sweeps_12"
    raise ValueError(f"Unsupported topology {topo}")


def main() -> None:
    args = parse_args()

    flowsim_dir = ROOT_DIR
    binary = flowsim_dir / "main"

    # Clean output directories for the requested topologies.
    results_roots = {}
    for topo in args.topologies:
        root = results_dir_for_topology(flowsim_dir, topo)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        results_roots[topo] = root

    ensure_built(flowsim_dir, binary)

    total_runs = len(args.topologies) * len(WINDOW_SIZES) * len(RDMA_SIZES)
    run_idx = 0

    for topo in args.topologies:
        results_root = results_roots[topo]
        for window in WINDOW_SIZES:
            for rdma in RDMA_SIZES:
                run_idx += 1
                title = RDMA_TITLES_BASE.get(rdma, str(rdma))
                run_tag = f"{title}_{window}"
                run_dir = results_root / run_tag
                run_dir.mkdir(parents=True, exist_ok=True)

                print(f"[run {run_idx}/{total_runs}] {run_tag}")

                # Run inside a disposable workspace so the repository stays clean.
                with tempfile.TemporaryDirectory(prefix="flowsim_run_") as tmp_str:
                    tmp_dir = pathlib.Path(tmp_str)
                    clean_logs(tmp_dir)

                    stdout_path = run_dir / "stdout.txt"
                    stderr_path = run_dir / "stderr.txt"
                    rc = run_cmd(
                        [str(binary.resolve()), str(window), str(rdma), str(topo)],
                        cwd=tmp_dir,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
                    if rc != 0:
                        print(f"  -> non-zero exit {rc}; check {stderr_path}")

                    collect_outputs(tmp_dir, run_dir)

                # Short delay to avoid hammering the system with back-to-back runs.
                time.sleep(0.1)

    print("[done] Results saved to:")
    for topo, root in results_roots.items():
        print(f"  topo {topo}: {root}")


if __name__ == "__main__":
    main()
