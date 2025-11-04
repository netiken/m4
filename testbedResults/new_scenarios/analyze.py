import re
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from itertools import chain

import os
import matplotlib.pyplot as plt
import numpy as np

Key = Tuple[int, str]

line_re = re.compile(r"\[(ud|rdma)\] client=(\d+) id=\d+(?:-\d+)? dur_ns=(\d+)")
ts_ns_re = re.compile(r"ts_ns=(\d+)")
grouped_ts_re = re.compile(r"t=(\d+)\s+ns")

TRIM_SAMPLES = 500

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Optional scaling for local ground-truth durations
# - LOCAL_UD_SCALE and LOCAL_RDMA_SCALE apply to per-request UD/RDMA comparisons
# - LOCAL_E2E_SCALE applies to end-to-end duration comparisons
LOCAL_SCALES = {
    "ud": float(os.getenv("LOCAL_UD_SCALE", "1")),
    "rdma": float(os.getenv("LOCAL_RDMA_SCALE", "1")),
}
LOCAL_E2E_SCALE = float(os.getenv("LOCAL_E2E_SCALE", "1"))


@dataclass(frozen=True)
class Source:
    name: str
    base: Path
    filenames: Sequence[str]
    rdma_scale: float = 1.0
    fallback_subdirs: Sequence[str] = ()

    def candidate_dirs(self, sweep_dir: str) -> List[Path]:
        dirs: List[Path] = []
        if sweep_dir:
            dirs.append(self.base / sweep_dir)
        for sub in self.fallback_subdirs:
            candidate = self.base / sub
            if candidate not in dirs:
                dirs.append(candidate)
        if not dirs:
            dirs.append(self.base / sweep_dir if sweep_dir else self.base)
        return dirs

    def resolve_file(self, sweep_dir: str, scenario: str) -> Path:
        for directory in self.candidate_dirs(sweep_dir):
            for filename in self.filenames:
                candidate = directory / scenario / filename
                if candidate.exists():
                    return candidate
        dirs = self.candidate_dirs(sweep_dir)
        return dirs[0] / scenario / self.filenames[0]


SOURCES: Sequence[Source] = (
    Source(
        name="flowsim",
        base=REPO_ROOT / "flowsim",
        filenames=("flowsim_output.txt",),
        fallback_subdirs=("sweeps",),
        rdma_scale=10.0 #Multiply everything by 10 to handle units
    ),
    Source(
        name="ns3",
        base=REPO_ROOT / "UNISON",
        filenames=("ns3_output.txt",),
        fallback_subdirs=("sweeps",),
        rdma_scale=10.0,
    ),
    Source(
        name="m4",
        base=REPO_ROOT / "m4",
        filenames=("m4_outputv2.txt", "m4_output.txt", "ns3_output.txt"),
        rdma_scale=10
    ),
)

TOPOLOGIES: Sequence[Tuple[str, Path, str]] = (
#    ("sweeps_4", SCRIPT_DIR / "expirements_4", "4"),
    ("sweeps_12", SCRIPT_DIR / "expirements_12", "12"),
)


def trim_series(values: Sequence[int], trim: int = TRIM_SAMPLES) -> List[int]:
    data = list(values)
    if not data or trim <= 0:
        return data
    if len(data) <= trim:
        return []
    if len(data) <= trim * 2:
        return data[trim:]
    return data[trim:-trim]


def parse_results(file_path: Path) -> Dict[Key, List[int]]:
    results: Dict[Key, List[int]] = {}
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                match = line_re.match(raw_line.strip())
                if not match:
                    continue
                op_type, client, duration = match.groups()
                key = (int(client), op_type)
                results.setdefault(key, []).append(int(duration))
    except FileNotFoundError:
        return {}
    return dict(sorted(results.items(), key=lambda item: item[0]))


def _extract_duration_from_event_logs(lines: Iterable[str]) -> Optional[Tuple[int, int, int]]:
    first_start_ns: Optional[int] = None
    last_end_ns: Optional[int] = None
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None

    for raw_line in lines:
        if "ts_ns=" not in raw_line:
            continue

        match = ts_ns_re.search(raw_line)
        if not match:
            continue

        timestamp = int(match.group(1))
        if min_ts is None or timestamp < min_ts:
            min_ts = timestamp
        if max_ts is None or timestamp > max_ts:
            max_ts = timestamp

        line = raw_line.strip()
        if "event=req_send" in line:
            if first_start_ns is None or timestamp < first_start_ns:
                first_start_ns = timestamp
        if "event=resp_rdma_read" in line or "event=resp_recv_ud" in line:
            if last_end_ns is None or timestamp > last_end_ns:
                last_end_ns = timestamp

    start_ns = first_start_ns if first_start_ns is not None else min_ts
    end_ns = last_end_ns if last_end_ns is not None else max_ts

    if start_ns is None or end_ns is None or end_ns < start_ns:
        return None

    return start_ns, end_ns, end_ns - start_ns


def _extract_duration_from_grouped_flows(grouped_path: Path) -> Optional[Tuple[int, int, int]]:
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None
    first_start_ns: Optional[int] = None
    last_end_ns: Optional[int] = None

    with grouped_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            match = grouped_ts_re.search(raw_line)
            if not match:
                continue

            timestamp = int(match.group(1))
            if min_ts is None or timestamp < min_ts:
                min_ts = timestamp
            if max_ts is None or timestamp > max_ts:
                max_ts = timestamp

            if "[client req_send]" in raw_line:
                if first_start_ns is None or timestamp < first_start_ns:
                    first_start_ns = timestamp
            if "[client rdma_recv]" in raw_line:
                if last_end_ns is None or timestamp > last_end_ns:
                    last_end_ns = timestamp

    start_ns = first_start_ns if first_start_ns is not None else min_ts
    end_ns = last_end_ns if last_end_ns is not None else max_ts

    if start_ns is None or end_ns is None or end_ns < start_ns:
        return None

    return start_ns, end_ns, end_ns - start_ns


def _compute_trimmed_range_from_boundaries(
    bounds: List[Tuple[int, int]], trim: int
) -> Optional[Tuple[int, int, int]]:
    if not bounds:
        return None
    bounds_sorted = sorted(bounds, key=lambda b: (b[0], b[1]))
    n = len(bounds_sorted)
    # Apply similar logic to trim_series but to boundaries
    if n > trim * 2:
        first_idx = trim
        last_idx = n - trim - 1
    elif n > trim:
        first_idx = trim
        last_idx = n - 1
    else:
        first_idx = 0
        last_idx = n - 1

    start_ns = bounds_sorted[first_idx][0]
    end_ns = bounds_sorted[last_idx][1]
    if end_ns < start_ns:
        return None
    return start_ns, end_ns, end_ns - start_ns


def _extract_flow_boundaries_from_flows_debug(flows_debug_path: Path) -> List[Tuple[int, int]]:
    """Return list of (start_ts_ns, end_ts_ns) per flow from flows_debug.txt.

    - Start timestamp is the first req_send ts, falling back to minimum ts_ns in the flow.
    - End timestamp is the last of resp_rdma_read/resp_recv_ud ts, falling back to max ts_ns in the flow.
    """
    bounds: List[Tuple[int, int]] = []
    if not flows_debug_path.exists():
        return bounds

    current_start: Optional[int] = None
    current_end: Optional[int] = None
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None

    def flush() -> None:
        nonlocal current_start, current_end, min_ts, max_ts
        if min_ts is None and max_ts is None:
            current_start = None
            current_end = None
            return
        start = current_start if current_start is not None else min_ts
        end = current_end if current_end is not None else max_ts
        if start is not None and end is not None and end >= start:
            bounds.append((start, end))
        current_start = None
        current_end = None
        min_ts = None
        max_ts = None

    with flows_debug_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("flow_id="):
                # Close previous flow block
                flush()
                continue
            if "ts_ns=" in line:
                match = ts_ns_re.search(line)
                if match:
                    ts = int(match.group(1))
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
                    if max_ts is None or ts > max_ts:
                        max_ts = ts
                if "event=req_send" in line:
                    if match:
                        ts = int(match.group(1))
                        if current_start is None or ts < current_start:
                            current_start = ts
                if "event=resp_rdma_read" in line or "event=resp_recv_ud" in line:
                    if match:
                        ts = int(match.group(1))
                        if current_end is None or ts > current_end:
                            current_end = ts
        # Flush last block
        flush()

    return bounds


def _extract_flow_boundaries_from_grouped(grouped_path: Path) -> List[Tuple[int, int]]:
    """Return list of (start_ts_ns, end_ts_ns) per flow from grouped_flows.txt."""
    bounds: List[Tuple[int, int]] = []
    if not grouped_path.exists():
        return bounds

    current_start: Optional[int] = None
    current_end: Optional[int] = None

    with grouped_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("### reqId="):
                # Close previous flow
                if current_start is not None and current_end is not None and current_end >= current_start:
                    bounds.append((current_start, current_end))
                current_start = None
                current_end = None
                continue
            match = grouped_ts_re.search(line)
            if not match:
                continue
            ts = int(match.group(1))
            if "[client req_send]" in line:
                if current_start is None or ts < current_start:
                    current_start = ts
            if "[client rdma_recv]" in line or "[client resp_recv]" in line:
                if current_end is None or ts > current_end:
                    current_end = ts

    if current_start is not None and current_end is not None and current_end >= current_start:
        bounds.append((current_start, current_end))
    return bounds


def compute_end_to_end_duration(scenario_dir: Path, trim_flows: int = TRIM_SAMPLES) -> Optional[Tuple[int, int, int]]:
    flows_debug = scenario_dir / "flows_debug.txt"
    if flows_debug.exists():
        # Prefer trimmed-per-flow computation if possible
        bounds = _extract_flow_boundaries_from_flows_debug(flows_debug)
        trimmed = _compute_trimmed_range_from_boundaries(bounds, trim_flows)
        if trimmed is not None:
            return trimmed
        # Fallback: simple min/max across all events
        with flows_debug.open("r", encoding="utf-8") as handle:
            result = _extract_duration_from_event_logs(handle)
        if result:
            return result

    grouped = scenario_dir / "grouped_flows.txt"
    if grouped.exists():
        # Prefer trimmed-per-flow computation if possible
        bounds = _extract_flow_boundaries_from_grouped(grouped)
        trimmed = _compute_trimmed_range_from_boundaries(bounds, trim_flows)
        if trimmed is not None:
            return trimmed
        # Fallback: simple min/max across all events
        result = _extract_duration_from_grouped_flows(grouped)
        if result:
            return result

    client_logs = sorted(scenario_dir.glob("client*.log"))
    if client_logs:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(path.open("r", encoding="utf-8"))
                for path in client_logs
            ]
            result = _extract_duration_from_event_logs(chain.from_iterable(handles))
        if result:
            return result

    return None


def compare_with_source(
    local_trimmed: Dict[Key, List[int]],
    remote_file: Path,
    *,
    rdma_scale: float,
    source_name: str,
) -> Optional[Tuple[np.ndarray, Dict[Key, Dict[str, List[float]]]]]:
    if not remote_file.exists():
        print(f"    • {source_name}: missing file {remote_file}")
        return None

    remote_results = parse_results(remote_file)
    if not remote_results:
        print(f"    • {source_name}: no parseable data in {remote_file}")
        return None

    remote_trimmed = {key: trim_series(vals) for key, vals in remote_results.items()}
    missing_keys = sorted(set(local_trimmed.keys()) - set(remote_trimmed.keys()))
    if missing_keys:
        preview = ", ".join(f"{key[0]}:{key[1]}" for key in missing_keys[:3])
        print(
            f"    • {source_name}: missing {len(missing_keys)} keys in {remote_file.name} "
            f"(e.g. {preview})"
        )

    errors: List[float] = []
    actual_by_key: Dict[Key, Dict[str, List[float]]] = {}

    for key, local_vals in local_trimmed.items():
        remote_vals = remote_trimmed.get(key)
        if not remote_vals:
            continue

        count = min(len(local_vals), len(remote_vals))
        if count == 0:
            continue

        local_arr = np.array(local_vals[:count], dtype=float)
        remote_arr = np.array(remote_vals[:count], dtype=float)
        if key[1] == "rdma" and rdma_scale != 1.0:
            remote_arr *= rdma_scale

        # Apply optional local GT scaling (per op-type)
        local_scale = LOCAL_SCALES.get(key[1], 1.0)
        if local_scale != 1.0:
            local_arr *= local_scale

        mask = local_arr != 0
        if not np.any(mask):
            continue

        local_arr = local_arr[mask]
        remote_arr = remote_arr[mask]
        if local_arr.size == 0:
            continue

        errs = np.abs(local_arr - remote_arr) / local_arr
        errors.extend(errs.tolist())
        actual_by_key[key] = {
            "local": local_arr.tolist(),
            "remote": remote_arr.tolist(),
        }

    if not errors:
        print(f"    • {source_name}: no overlapping samples in {remote_file}")
        return None

    return np.array(errors), actual_by_key


def plot_scenario_cdfs(
    scenario_data: Dict[str, Dict[str, Dict]],
    output_dir: Path,
    topology_label: str,
) -> None:
    for scenario, data in scenario_data.items():
        if not data["relative_errors"]:
            continue

        plt.figure(figsize=(10, 6))

        for source_name, errors in data["relative_errors"].items():
            if len(errors) == 0:
                continue
            arr = np.sort(np.array(errors))
            y = np.linspace(0, 1, len(arr), endpoint=False)
            plt.step(arr, y, where="post", label=source_name, linewidth=2)

        plt.xlabel("Relative Error")
        plt.xscale("log")
        plt.xlim(1e-2, 10)
        plt.ylabel("CDF")
        plt.title(f"CDF of Relative Errors - Scenario: {scenario} (topology {topology_label})")
        plt.grid(True, linestyle="--", alpha=0.6)
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()
        plt.tight_layout()

        safe_scenario = scenario.replace("/", "_").replace("\\", "_")
        plt.savefig(output_dir / f"cdf_{safe_scenario}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_box_whiskers(
    scenario_data: Dict[str, Dict[str, Dict]],
    output_dir: Path,
) -> None:
    key_data = defaultdict(lambda: defaultdict(list))

    for data in scenario_data.values():
        for source_name, actual_vals in data["actual_values"].items():
            if not actual_vals:
                continue
            for key, values in actual_vals.items():
                if isinstance(values, dict):
                    remote_values = values.get("remote", [])
                    if remote_values:
                        key_data[key][source_name].extend(remote_values)
                else:
                    key_data[key][source_name].extend(values)

        local_vals = data.get("local_values", {})
        for key, values in local_vals.items():
            if values:
                key_data[key]["local"].extend(values)

    if not key_data:
        print("No data available for box plots")
        return

    keys = list(key_data.keys())
    n_keys = len(keys)
    n_cols = min(3, n_keys)
    n_rows = (n_keys + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for idx, (key, source_map) in enumerate(key_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        plot_data = []
        labels = []
        for source_name, values in source_map.items():
            if values:
                plot_data.append(values)
                labels.append(source_name)

        if plot_data:
            ax.boxplot(plot_data, tick_labels=labels)
            ax.set_title(f"Key: {key}")
            ax.set_ylabel("Duration (ns)")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    for idx in range(n_keys, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "box_whiskers_all_keys.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_scenario_box_whiskers(
    scenario_data: Dict[str, Dict[str, Dict]],
    output_dir: Path,
    topology_label: str,
) -> None:
    for scenario, data in scenario_data.items():
        if not data["actual_values"] or not data["local_values"]:
            continue

        first_remote = next((vals for vals in data["actual_values"].values() if vals), None)
        if not first_remote:
            continue

        keys = sorted(first_remote.keys())
        x_values = sorted({key[0] for key in keys})
        y_values = sorted({key[1] for key in keys})

        n_cols = len(x_values)
        n_rows = len(y_values)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for key in keys:
            x_idx = x_values.index(key[0])
            y_idx = y_values.index(key[1])
            ax = axes[y_idx][x_idx]

            plot_data = []
            labels = []

            for source_name, actual_vals in data["actual_values"].items():
                if not actual_vals or key not in actual_vals:
                    continue
                values = actual_vals[key]
                if isinstance(values, dict):
                    remote_values = values.get("remote", [])
                    if remote_values:
                        plot_data.append(remote_values)
                        labels.append(source_name)

            if key in data["local_values"]:
                local_values = data["local_values"][key]
                if local_values:
                    plot_data.append(local_values)
                    labels.append("local")

            if plot_data:
                bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True)
                colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow"]
                for patch, color in zip(bp["boxes"], colors * ((len(bp["boxes"]) + len(colors) - 1) // len(colors))):
                    patch.set_facecolor(color)

                ax.set_title(f"Key: {key}")
                ax.set_ylabel("Duration (ns)")
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_visible(False)

        for i, y_val in enumerate(y_values):
            axes[i][0].set_ylabel(f"Y={y_val}", fontsize=12, fontweight="bold")

        for j, x_val in enumerate(x_values):
            axes[0][j].set_title(f"X={x_val}", fontsize=12, fontweight="bold")

        plt.suptitle(f"Box Plots for Scenario: {scenario} (topology {topology_label})", fontsize=16, fontweight="bold")
        plt.tight_layout()
        safe_scenario = scenario.replace("/", "_").replace("\\", "_")
        plt.savefig(output_dir / f"box_{safe_scenario}.png", dpi=300, bbox_inches="tight")
        plt.close()


def write_scenario_csvs(
    scenario_data: Dict[str, Dict[str, Dict]],
    output_dir: Path,
) -> None:
    base_csv_dir = output_dir / "sweepcsv"
    base_csv_dir.mkdir(parents=True, exist_ok=True)

    for scenario, data in scenario_data.items():
        if not data["actual_values"]:
            continue

        remote_columns: List[str] = []
        for source_name in data["actual_values"].keys():
            if source_name not in remote_columns:
                remote_columns.append(source_name)

        keys = set()
        for actual_vals in data["actual_values"].values():
            if isinstance(actual_vals, dict):
                keys.update(actual_vals.keys())
        if "local_values" in data and isinstance(data["local_values"], dict):
            keys.update(data["local_values"].keys())
        keys = sorted(keys)

        safe_scenario = scenario.replace("/", "_").replace("\\", "_")
        scenario_dir = base_csv_dir / safe_scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        for key in keys:
            local_vals = data["local_values"].get(key, [])

            filename_to_values = {name: [] for name in remote_columns}
            for source_name, kv in data["actual_values"].items():
                if not kv or key not in kv:
                    continue
                values = kv[key]
                if isinstance(values, dict):
                    values = values.get("remote", [])
                filename_to_values[source_name] = values

            column_order = ["local"] + remote_columns
            columns = [local_vals] + [filename_to_values[name] for name in remote_columns]
            max_len = max((len(col) for col in columns), default=0)

            safe_key = f"key_{key[0]}_{key[1]}"
            out_path = scenario_dir / f"{safe_key}.csv"
            with out_path.open("w", encoding="utf-8") as handle:
                handle.write(",".join(column_order) + "\n")
                for i in range(max_len):
                    row_vals = []
                    for col in columns:
                        row_vals.append(str(col[i]) if i < len(col) else "")
                    handle.write(",".join(row_vals) + "\n")


def process_topology(sweep_dir: str, local_base: Path, topology_label: str) -> None:
    if not local_base.is_dir():
        print(f"[skip] local base not found: {local_base}")
        return

    output_dir = SCRIPT_DIR / "sweepfigures" / sweep_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Processing topology {topology_label} ({sweep_dir}) =====")

    vals_per_source: Dict[str, List[float]] = {source.name: [] for source in SOURCES}
    e2e_errors_per_source: Dict[str, List[float]] = {source.name: [] for source in SOURCES}
    scenario_data: Dict[str, Dict[str, Dict]] = {}

    for scenario_path in sorted(local_base.iterdir()):
        if not scenario_path.is_dir():
            continue

        scenario = scenario_path.name
        local_file = scenario_path / "real_world.txt"
        if not local_file.exists():
            print(f"\n=== Scenario: {scenario} ===")
            print(f"    • Missing local file {local_file}")
            continue

        local_results = parse_results(local_file)
        if not local_results:
            print(f"\n=== Scenario: {scenario} ===")
            print("    • No local measurements parsed")
            continue

        local_trimmed = {key: trim_series(vals) for key, vals in local_results.items()}
        scenario_data[scenario] = {
            "relative_errors": {source.name: [] for source in SOURCES},
            "actual_values": {source.name: {} for source in SOURCES},
            "local_values": {},
        }

        print(f"\n=== Scenario: {scenario} ===")

        # Only compute/print E2E for scenarios ending with "_1"
        compute_e2e = True #scenario.endswith("_1")
        duration_info = None
        if compute_e2e:
            duration_info = compute_end_to_end_duration(scenario_path)
            if duration_info:
                start_ns, end_ns, duration_ns = duration_info
                # Scale local E2E if requested
                scaled_duration_ns = duration_ns * LOCAL_E2E_SCALE
                duration_ms = scaled_duration_ns / 1_000_000
                print(
                    f"    • End-to-end duration (trimmed): {scaled_duration_ns:.0f} ns "
                    f"({duration_ms:.3f} ms)"
                )
            else:
                print("    • End-to-end duration: unavailable")

        for source in SOURCES:
            remote_file = source.resolve_file(sweep_dir, scenario)
            if remote_file.exists() and compute_e2e:
                remote_duration = compute_end_to_end_duration(remote_file.parent)
                if remote_duration:
                    r_start_ns, r_end_ns, r_duration_ns = remote_duration
                    r_duration_ms = r_duration_ns / 1_000_000
                    print(
                        f"    • {source.name}: End-to-end duration (trimmed) "
                        f"{r_duration_ns} ns ({r_duration_ms:.3f} ms)"
                    )
                    if duration_info:
                        # Compute relative end-to-end error vs local
                        if scaled_duration_ns != 0:
                            e2e_err = abs(r_duration_ns - scaled_duration_ns) / scaled_duration_ns
                            e2e_errors_per_source[source.name].append(e2e_err)
                else:
                    print(f"    • {source.name}: End-to-end duration unavailable")

            outcome = compare_with_source(
                local_trimmed,
                remote_file,
                rdma_scale=source.rdma_scale,
                source_name=source.name,
            )
            if outcome is None:
                continue

            errors, actual = outcome
            scenario_data[scenario]["relative_errors"][source.name] = errors
            scenario_data[scenario]["actual_values"][source.name] = actual
            if not scenario_data[scenario]["local_values"]:
                scenario_data[scenario]["local_values"] = {
                    key: values["local"] for key, values in actual.items()
                }

            vals_per_source[source.name].extend(errors.tolist())
            print(
                f"    • {source.name}: median error {np.median(errors):.6f}, "
                f"mean error {np.mean(errors):.6f}, samples {len(errors)}"
            )

    print("\n=== Overall Results ===")
    for source_name, vals in vals_per_source.items():
        if vals:
            arr = np.array(vals)
            print(
                f"{source_name} → median error: {np.median(arr):.6f}, "
                f"mean error: {np.mean(arr):.6f}, samples: {len(arr)}"
            )
        else:
            print(f"{source_name} → no valid results")

    print("\n=== End-to-End Error Summary ===")
    for source_name, vals in e2e_errors_per_source.items():
        if vals:
            arr = np.array(vals)
            print(
                f"{source_name} → median E2E error: {np.median(arr):.6f}, "
                f"mean E2E error: {np.mean(arr):.6f}, samples: {len(arr)}"
            )
        else:
            print(f"{source_name} → no E2E comparisons")

    plt.figure(figsize=(8, 6))
    for source_name, vals in vals_per_source.items():
        if not vals:
            continue
        arr = np.sort(np.array(vals))
        y = np.linspace(0, 1, len(arr), endpoint=False)
        plt.step(arr, y, where="post", label=source_name)

    plt.xlabel("Relative Error")
    plt.xlim(0, 5)
    plt.ylabel("CDF")
    plt.title(f"CDF of Relative Errors Across Sources (topology {topology_label})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "relative_errors.png")
    plt.close()

    all_relative_errors: List[float] = []
    for source_name, vals in vals_per_source.items():
        if vals:
            all_relative_errors.extend(vals)
            safe_name = source_name.replace(" ", "_")
            np.save(
                output_dir / f"relative_errors_{safe_name}_{topology_label}.npy",
                np.array(vals),
            )
            print(
                f"Saved {len(vals)} relative error samples for {source_name} "
                f"to relative_errors_{safe_name}_{topology_label}.npy"
            )

    if all_relative_errors:
        np.save(
            output_dir / f"relative_errors_data_{topology_label}.npy",
            np.array(all_relative_errors),
        )
        print(
            f"Saved {len(all_relative_errors)} combined relative error samples "
            f"to relative_errors_data_{topology_label}.npy"
        )

    print("\n=== Generating Additional Plots ===")
    #plot_scenario_cdfs(scenario_data, output_dir, topology_label)
    plot_box_whiskers(scenario_data, output_dir)
    #plot_scenario_box_whiskers(scenario_data, output_dir, topology_label)
    #write_scenario_csvs(scenario_data, output_dir)
    print(f"Plots saved to {output_dir}/")


def main() -> None:
    processed_any = False
    for sweep_dir, local_base, label in TOPOLOGIES:
        if local_base.is_dir():
            process_topology(sweep_dir, local_base, label)
            processed_any = True
        else:
            print(f"[info] Skipping topology {label}; missing {local_base}")

    if not processed_any:
        print("No experiment directories found. Nothing to do.")


if __name__ == "__main__":
    main()
