#!/usr/bin/env python3
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

# Sweep dimensions shared with the FlowSim harness.
WINDOW_SIZES: List[int] = [1, 2, 4, 8, 12, 16]
RDMA_TITLES_BASE: Dict[int, str] = {
    102408: "100",
    204808: "200",
    256008: "250",
    307208: "300",
    409608: "400",
    512008: "500",
    665608: "650",
    768008: "750",
    921608: "900",
    1024008: "1000",
    1048584: "10000",
}
RDMA_SIZES: List[int] = list(RDMA_TITLES_BASE.keys())

# Repository root for M4 (script lives inside this directory).
ROOT_DIR = pathlib.Path(__file__).resolve().parent

def cmake_path() -> str:
    """Return the absolute path to the cmake executable, or exit with guidance."""
    cmake = shutil.which("cmake")
    if cmake is None:
        print("[build] Could not find 'cmake' in PATH. Please install or load cmake before running sweeps.", file=sys.stderr)
        sys.exit(1)
    return cmake


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments that control which topologies to sweep."""
    parser = argparse.ArgumentParser(description="Run M4 sweeps across window/RDMA/topology combos.")
    parser.add_argument(
        "--topologies",
        type=int,
        nargs="+",
        default=[4],
        choices=[1, 4, 12],
        help="Topology sizes to include (1=single link, 4=multi-client, 12=tree). Default: 4.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path, env: Optional[Dict[str, str]] = None) -> int:
    """Spawn a subprocess and stream stdout/stderr into files."""
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True, env=env)
        return proc.returncode


def locate_nvcc(existing_env: Dict[str, str]) -> Optional[pathlib.Path]:
    """Find the nvcc compiler if it is available."""
    nvcc = shutil.which("nvcc", path=existing_env.get("PATH"))
    if nvcc:
        return pathlib.Path(nvcc)

    for key in ("CUDA_HOME", "CUDA_PATH"):
        cuda_root = existing_env.get(key)
        if cuda_root:
            candidate = pathlib.Path(cuda_root) / "bin" / "nvcc"
            if candidate.exists():
                return candidate

    fallback = pathlib.Path("/usr/local/cuda/bin/nvcc")
    if fallback.exists():
        return fallback
    return None


def ensure_built(project_root: pathlib.Path) -> pathlib.Path:
    """Configure and build the no_flowsim binary via CMake."""
    cmake = cmake_path()
    build_dir = project_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    stdout = build_dir / "build_stdout.txt"
    stderr = build_dir / "build_stderr.txt"

    build_env = os.environ.copy()
    nvcc_path = build_env.get("CUDACXX")
    if not nvcc_path:
        nvcc = locate_nvcc(build_env)
        if nvcc is None:
            print("[build] Could not locate 'nvcc'. Install the CUDA toolkit or set the CUDACXX environment variable.", file=sys.stderr)
            sys.exit(1)
        build_env["CUDACXX"] = str(nvcc)
        nvcc_dir = str(nvcc.parent)
        path_entries = build_env.get("PATH", "").split(os.pathsep)
        if nvcc_dir not in path_entries:
            build_env["PATH"] = os.pathsep.join([nvcc_dir] + [p for p in path_entries if p])

    # Ensure the CMake cache matches this checkout; reconfigure if needed.
    cache_file = build_dir / "CMakeCache.txt"
    needs_configure = True
    if cache_file.exists():
        try:
            for line in cache_file.open("r"):
                if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                    cache_root = pathlib.Path(line.split("=", 1)[1].strip()).resolve()
                    if cache_root == project_root.resolve():
                        needs_configure = False
                    break
        except OSError:
            needs_configure = True

    if needs_configure:
        print("[build] Configuring M4 via CMake...")
        cfg_stdout = build_dir / "configure_stdout.txt"
        cfg_stderr = build_dir / "configure_stderr.txt"
        rc_cfg = run_cmd(
            [cmake, str(project_root.resolve())],
            cwd=build_dir,
            stdout_path=cfg_stdout,
            stderr_path=cfg_stderr,
            env=build_env,
        )
        if rc_cfg != 0:
            print(f"[build] CMake configure failed with code {rc_cfg}. See {cfg_stderr}")
            sys.exit(rc_cfg)

    print("[build] Building no_flowsim via CMake...")
    rc = run_cmd(
        [cmake, "--build", ".", "--target", "no_flowsim", f"-j{os.cpu_count() or 1}"],
        cwd=build_dir,
        stdout_path=stdout,
        stderr_path=stderr,
        env=build_env,
    )
    if rc != 0:
        print(f"[build] CMake build failed with code {rc}. See {stderr}")
        sys.exit(rc)

    binary = build_dir / "no_flowsim"
    if not binary.exists():
        print(f"[build] Binary not found at {binary}")
        sys.exit(1)

    print(f"[build] Using binary: {binary}")
    return binary


def clean_logs(src_dir: pathlib.Path) -> None:
    """Remove leftover logs from previous runs to avoid mixing outputs."""
    for pattern in ["client_*.log", "server.log", "flows.txt", "flowsim_output.txt"]:
        for path in src_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def collect_outputs(src_dir: pathlib.Path, dest_dir: pathlib.Path) -> None:
    """Copy well-known artefacts from the source directory into the run directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ["flows.txt", "server.log", "flowsim_output.txt"]:
        candidate = src_dir / name
        if candidate.exists():
            shutil.copy2(candidate, dest_dir / candidate.name)
    for log in src_dir.glob("client_*.log"):
        shutil.copy2(log, dest_dir / log.name)


def results_dir_for_topology(base: pathlib.Path, topo: int) -> pathlib.Path:
    """Return the sweep directory name that matches downstream tooling."""
    if topo == 1:
        return base / "sweeps"
    if topo == 4:
        return base / "sweeps_4"
    if topo == 12:
        return base / "sweeps_12"
    raise ValueError(f"Unsupported topology {topo}")


def main() -> None:
    args = parse_args()

    project_root = ROOT_DIR

    # Clean output directories for the requested topologies.
    results_roots: Dict[int, pathlib.Path] = {}
    for topo in args.topologies:
        root = results_dir_for_topology(project_root, topo)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        results_roots[topo] = root

    binary = ensure_built(project_root)

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

                print(f"[run {run_idx}/{total_runs}] topo {topo} -> {run_tag}")

                clean_logs(project_root)

                stdout_path = run_dir / "stdout.txt"
                stderr_path = run_dir / "stderr.txt"
                rc = run_cmd(
                    [str(binary.resolve()), str(window), str(rdma), str(topo)],
                    cwd=project_root,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                if rc != 0:
                    print(f"  -> non-zero exit {rc}; check {stderr_path}")

                collect_outputs(project_root, run_dir)
                time.sleep(0.1)

    print("[done] Results saved to:")
    for topo, root in results_roots.items():
        print(f"  topo {topo}: {root}")


if __name__ == "__main__":
    main()
