#!/usr/bin/env python3
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Dict, List

# Sweep dimensions shared with the FlowSim harness.
WINDOW_SIZES: List[int] = [1, 2, 4] # 8, 12, 16
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


# Repository root for UNISON (script lives inside this directory).
ROOT_DIR = pathlib.Path(__file__).resolve().parent

# Mapping between topology choice and the corresponding scratch target/binary suffix.
TOPOLOGY_TARGETS: Dict[int, Dict[str, str]] = {
    1: {"target": "scratch_single", "suffix": "single"},
    4: {"target": "scratch_four", "suffix": "four"},
    12: {"target": "scratch_twelve", "suffix": "twelve"},
}


def cmake_path() -> str:
    """Return the absolute path to the cmake executable, or exit with guidance."""
    cmake = shutil.which("cmake")
    if cmake:
        return cmake

    # Fallback: look for a portable CMake under the build directory
    portable_candidates = sorted((ROOT_DIR / "cmake-cache").glob(".cmake/cmake-*/bin/cmake"))
    if portable_candidates:
        return str(portable_candidates[0].resolve())

    print(
        "[build] Could not find 'cmake' in PATH and no portable CMake found under 'cmake-cache/.cmake'.\n"
        "        Install CMake (e.g., 'sudo apt-get install cmake') or download a portable binary into 'UNISON/cmake-cache/.cmake'.",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments that control which topologies to sweep."""
    parser = argparse.ArgumentParser(description="Run UNISON sweeps across window/RDMA/topology combos.")
    parser.add_argument(
        "--topologies",
        type=int,
        nargs="+",
        default=[4],
        choices=sorted(TOPOLOGY_TARGETS.keys()),
        help="Topology sizes to include (1=single link, 4=multi-client, 12=tree). Default: 4.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> int:
    """Spawn a subprocess and stream stdout/stderr into files."""
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
        return proc.returncode


def ensure_built(ns3_root: pathlib.Path, topology: int) -> pathlib.Path:
    """Build the ns-3 scratch binary for the requested topology and return its path."""
    if topology not in TOPOLOGY_TARGETS:
        raise ValueError(f"Unsupported topology {topology}")

    cmake = cmake_path()
    target_info = TOPOLOGY_TARGETS[topology]
    build_dir = ns3_root / "cmake-cache"
    build_dir.mkdir(parents=True, exist_ok=True)

    stdout = build_dir / f"build_{target_info['suffix']}_stdout.txt"
    stderr = build_dir / f"build_{target_info['suffix']}_stderr.txt"

    # Ensure the CMake cache matches this checkout; reconfigure if needed.
    cache_file = build_dir / "CMakeCache.txt"
    needs_configure = True
    if cache_file.exists():
        try:
            cache_lines = cache_file.read_text().splitlines()
            for line in cache_lines:
                if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                    cache_root = pathlib.Path(line.split("=", 1)[1].strip()).resolve()
                    if cache_root == ns3_root.resolve():
                        needs_configure = False
                    # don't break; we may also want to check options below
            # Ensure required options are set
            for line in cache_lines:
                if line.startswith("NS3_MTP:"):
                    if not line.strip().endswith("=ON"):
                        needs_configure = True
                    break
            for line in cache_lines:
                if line.startswith("NS3_WARNINGS_AS_ERRORS:"):
                    if not line.strip().endswith("=OFF"):
                        needs_configure = True
                    break
        except OSError:
            needs_configure = True

    # If we need to configure because the cache points to a different tree,
    # clean up the stale CMake state so cmake doesn't error out.
    if needs_configure and cache_file.exists():
        for entry in ("CMakeCache.txt", "CMakeFiles", "cmake_install.cmake", "Makefile", "compile_commands.json"):
            path = build_dir / entry
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except OSError:
                pass

        # Also drop any generated wrapper headers pointing to the old tree
        include_dir = ns3_root / "build" / "include"
        try:
            if include_dir.exists():
                shutil.rmtree(include_dir)
        except OSError:
            pass

    if needs_configure:
        print("[build] Configuring ns-3 via CMake...")
        cfg_stdout = build_dir / "configure_stdout.txt"
        cfg_stderr = build_dir / "configure_stderr.txt"
        rc_cfg = run_cmd(
            [
                cmake,
                "-DNS3_MTP=ON",
                "-DNS3_WARNINGS_AS_ERRORS=OFF",
                "-DCMAKE_BUILD_TYPE=default",
                str(ns3_root.resolve()),
            ],
            cwd=build_dir,
            stdout_path=cfg_stdout,
            stderr_path=cfg_stderr,
        )
        if rc_cfg != 0:
            print(f"[build] CMake configure failed with code {rc_cfg}. See {cfg_stderr}")
            sys.exit(rc_cfg)
        needs_configure = False

    print(f"[build] Building ns-3 {target_info['target']} via CMake...")
    rc = run_cmd(
        [cmake, "--build", ".", "--target", target_info["target"], f"-j{os.cpu_count() or 1}"],
        cwd=build_dir,
        stdout_path=stdout,
        stderr_path=stderr,
    )
    if rc != 0:
        print(f"[build] CMake build failed with code {rc}. See {stderr}")
        sys.exit(rc)

    bin_dir = ns3_root / "build" / "scratch"
    candidates = sorted(bin_dir.glob(f"ns3.*-{target_info['suffix']}-default"))
    if not candidates:
        print(f"[build] Could not find binary for topology {topology} under {bin_dir}")
        sys.exit(1)
    binary = candidates[0]
    print(f"[build] Using binary: {binary}")
    return binary


def clean_logs(_root: pathlib.Path) -> None:
    """Placeholder for log cleanup; UNISON scratch binaries only emit stdout/stderr."""
    return


def collect_outputs(_root: pathlib.Path, _dest_dir: pathlib.Path) -> None:
    """Placeholder for copying extra artefacts; no additional outputs today."""
    return


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

    ns3_root = ROOT_DIR

    # Clean output directories for the requested topologies.
    results_roots: Dict[int, pathlib.Path] = {}
    for topo in args.topologies:
        root = results_dir_for_topology(ns3_root, topo)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        results_roots[topo] = root

    # Build binaries up front so we fail fast if something goes wrong.
    binaries: Dict[int, pathlib.Path] = {}
    for topo in args.topologies:
        binaries[topo] = ensure_built(ns3_root, topo)

    total_runs = len(args.topologies) * len(WINDOW_SIZES) * len(RDMA_SIZES)
    run_idx = 0

    for topo in args.topologies:
        binary = binaries[topo]
        results_root = results_roots[topo]
        for window in WINDOW_SIZES:
            for rdma in RDMA_SIZES:
                run_idx += 1
                title = RDMA_TITLES_BASE.get(rdma, str(rdma))
                run_tag = f"{title}_{window}"
                run_dir = results_root / run_tag
                run_dir.mkdir(parents=True, exist_ok=True)

                print(f"[run {run_idx}/{total_runs}] topo {topo} -> {run_tag}")

                clean_logs(ns3_root)

                stdout_path = run_dir / "stdout.txt"
                stderr_path = run_dir / "stderr.txt"
                rc = run_cmd(
                    [str(binary.resolve()), f"--maxWindows={window}", f"--dataBytes={rdma}"],
                    cwd=ns3_root,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                if rc != 0:
                    print(f"  -> non-zero exit {rc}; check {stderr_path}")

                collect_outputs(ns3_root, run_dir)

                time.sleep(0.1)

    print("[done] Results saved to:")
    for topo, root in results_roots.items():
        print(f"  topo {topo}: {root}")


if __name__ == "__main__":
    main()
