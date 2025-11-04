import argparse
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

Key = Tuple[int, str]

line_re = re.compile(r"\[(ud|rdma)\] client=(\d+) id=\d+(?:-\d+)? dur_ns=(\d+)")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LOCAL_BASE_CANDIDATES = ("expirements_4", "expirements_12", "expirements")


@dataclass
class DataSource:
    name: str
    root: Path
    filename: str
    rdma_scale: float = 1.0

    def resolve_file(self, scenario: str) -> Path:
        return self.root / scenario / self.filename


@dataclass
class ComparisonOutcome:
    errors: np.ndarray
    per_key: Dict[Key, Dict[str, List[float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local measurements against simulator outputs and generate plots."
    )
    parser.add_argument(
        "--local-base",
        type=str,
        help="Directory that contains real_world.txt files (default: auto-detect). "
             "Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Additional data source in the form name=path:filename[:rdma_scale]. "
            "Example: flowsim=/path/to/sweeps:flowsim_output.txt"
        ),
    )
    parser.add_argument(
        "--no-default-sources",
        action="store_true",
        help="Disable auto-detected data sources (flowsim, ns3).",
    )
    parser.add_argument(
        "--trim",
        type=int,
        default=20,
        help="Number of samples to drop from both the start and end of each series.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Where to write plots and CSVs (default: new_scenarios/sweepfigures).",
    )
    return parser.parse_args()


def detect_default_local_base() -> Path:
    for candidate in DEFAULT_LOCAL_BASE_CANDIDATES:
        path = SCRIPT_DIR / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "Could not auto-detect a local base directory. "
        "Please create one of "
        f"{', '.join(DEFAULT_LOCAL_BASE_CANDIDATES)} or pass --local-base."
    )


def resolve_relative(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def build_default_sources(local_base: Path) -> List[DataSource]:
    suffix_map = {
        "_4": "sweeps_4",
        "_12": "sweeps_12",
    }
    remote_subdir = "sweeps"
    for suffix, sweeps_dir in suffix_map.items():
        if local_base.name.endswith(suffix):
            remote_subdir = sweeps_dir
            break

    defaults: List[DataSource] = []

    flowsim_root = REPO_ROOT / "flowsim" / remote_subdir
    if flowsim_root.is_dir():
        defaults.append(DataSource("flowsim", flowsim_root, "flowsim_output.txt"))

    ns3_root = REPO_ROOT / "UNISON" / remote_subdir
    if ns3_root.is_dir():
        defaults.append(DataSource("ns3", ns3_root, "ns3_output.txt", rdma_scale=2.0))

    return defaults


def parse_extra_sources(specs: Sequence[str]) -> List[DataSource]:
    extras: List[DataSource] = []
    for spec in specs:
        try:
            name, rest = spec.split("=", 1)
            path_part, filename, *scale_part = rest.split(":")
        except ValueError as exc:
            raise SystemExit(
                f"Invalid --source '{spec}'. Expected format name=path:filename[:rdma_scale]."
            ) from exc

        if not name:
            raise SystemExit(f"Invalid --source '{spec}': name cannot be empty.")

        root = resolve_relative(path_part)
        rdma_scale = 1.0
        if scale_part:
            try:
                rdma_scale = float(scale_part[0])
            except ValueError as exc:  # pragma: no cover - defensive
                raise SystemExit(
                    f"Invalid rdma_scale '{scale_part[0]}' in --source '{spec}'."
                ) from exc

        extras.append(DataSource(name=name, root=root, filename=filename, rdma_scale=rdma_scale))
    return extras


def merge_sources(defaults: Sequence[DataSource], extras: Sequence[DataSource]) -> List[DataSource]:
    merged: "OrderedDict[str, DataSource]" = OrderedDict()
    for source in defaults:
        merged[source.name] = source
    for source in extras:
        merged[source.name] = source
    return list(merged.values())


def trim_series(values: Sequence[int], trim: int) -> List[int]:
    series = list(values)
    if trim <= 0:
        return series
    if len(series) <= trim:
        return []
    if len(series) <= trim * 2:
        return series[trim:]
    return series[trim:-trim]


def parse_results(file_path: Path) -> Dict[Key, List[int]]:
    results: Dict[Key, List[int]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            match = line_re.match(raw_line.strip())
            if not match:
                continue
            op_type, client_id, duration = match.groups()
            key = (int(client_id), op_type)
            results.setdefault(key, []).append(int(duration))
    return dict(sorted(results.items(), key=lambda item: item[0]))


def compare_with_source(
    local_trimmed: Dict[Key, List[int]],
    remote_file: Path,
    *,
    trim: int,
    rdma_scale: float,
) -> Optional[ComparisonOutcome]:
    if not remote_file.exists():
        return None

    remote_results = parse_results(remote_file)
    if not remote_results:
        return None

    missing_keys = sorted(set(local_trimmed.keys()) - set(remote_results.keys()))
    if missing_keys:
        preview = ", ".join(f"{key[0]}:{key[1]}" for key in missing_keys[:3])
        print(
            f"    • missing {len(missing_keys)} keys in {remote_file.name} "
            f"(e.g. {preview})"
        )

    arrays: List[np.ndarray] = []
    per_key: Dict[Key, Dict[str, List[float]]] = {}
    for key, local_vals in local_trimmed.items():
        if key not in remote_results:
            continue

        remote_vals = trim_series(remote_results[key], trim)
        if not local_vals or not remote_vals:
            continue

        length = min(len(local_vals), len(remote_vals))
        if length == 0:
            continue

        local_arr = np.asarray(local_vals[:length], dtype=np.float64)
        remote_arr = np.asarray(remote_vals[:length], dtype=np.float64)

        if key[1] == "rdma" and rdma_scale != 1.0:
            remote_arr *= rdma_scale

        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = np.where(local_arr == 0, np.nan, local_arr)
            rel_err = np.abs(local_arr - remote_arr) / denominator

        rel_err = rel_err[np.isfinite(rel_err)]
        if rel_err.size == 0:
            continue

        arrays.append(rel_err)
        per_key[key] = {
            "local": local_arr.tolist(),
            "remote": remote_arr.tolist(),
        }

    if not arrays:
        return None

    return ComparisonOutcome(errors=np.concatenate(arrays), per_key=per_key)


def process_scenarios(
    local_base: Path,
    sources: Sequence[DataSource],
    trim: int,
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, List[float]]]:
    scenario_data: Dict[str, Dict[str, Dict]] = {}
    aggregated: Dict[str, List[float]] = {source.name: [] for source in sources}

    scenario_dirs = sorted(path for path in local_base.iterdir() if path.is_dir())
    if not scenario_dirs:
        print(f"No scenarios found in {local_base}")
        return scenario_data, aggregated

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        local_file = scenario_dir / "real_world.txt"
        if not local_file.exists():
            print(f"\n=== Scenario: {scenario_name} ===")
            print(f"  - Skipping: missing {local_file.name}")
            continue

        local_results = parse_results(local_file)
        if not local_results:
            print(f"\n=== Scenario: {scenario_name} ===")
            print(f"  - Skipping: no parsable lines in {local_file.name}")
            continue

        local_trimmed = {key: trim_series(values, trim) for key, values in local_results.items()}
        scenario_record = {
            "relative_errors": {source.name: [] for source in sources},
            "actual_values": {source.name: {} for source in sources},
            "local_values": local_trimmed,
        }
        scenario_data[scenario_name] = scenario_record

        print(f"\n=== Scenario: {scenario_name} ===")
        for source in sources:
            remote_file = source.resolve_file(scenario_name)
            outcome = compare_with_source(
                local_trimmed,
                remote_file,
                trim=trim,
                rdma_scale=source.rdma_scale,
            )
            if outcome is None:
                print(f"  - {source.name}: no valid comparison (checked {remote_file})")
                continue

            aggregated[source.name].extend(outcome.errors.tolist())
            scenario_record["relative_errors"][source.name] = outcome.errors.tolist()
            scenario_record["actual_values"][source.name] = outcome.per_key

            print(
                f"  - {source.name}: median={np.median(outcome.errors):.6f}, "
                f"mean={np.mean(outcome.errors):.6f}, samples={outcome.errors.size}"
            )

    return scenario_data, aggregated


def summarize_source_stats(aggregated: Dict[str, List[float]]) -> None:
    print("\n=== Overall Results ===")
    for name, values in aggregated.items():
        if not values:
            print(f"{name}: no valid results")
            continue
        arr = np.asarray(values, dtype=np.float64)
        print(
            f"{name}: median={np.median(arr):.6f}, "
            f"mean={np.mean(arr):.6f}, samples={arr.size}"
        )


def plot_relative_error_cdf(
    aggregated: Dict[str, List[float]],
    sources: Sequence[DataSource],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    have_data = False

    for source in sources:
        values = aggregated.get(source.name, [])
        if not values:
            continue

        arr = np.sort(np.asarray(values, dtype=np.float64))
        if arr.size == 0:
            continue

        y = np.linspace(0, 1, arr.size, endpoint=False)
        ax.step(arr, y, where="post", label=source.name)
        np.save(output_dir / f"relative_errors_{source.name}.npy", arr)
        have_data = True

    if not have_data:
        plt.close(fig)
        print("No relative error data available for overall CDF plot")
        return

    ax.set_xlabel("Relative Error")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Relative Errors Across Sources")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(output_dir / "relative_errors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    combined = [value for values in aggregated.values() for value in values]
    if combined:
        np.save(
            output_dir / "relative_errors_combined.npy",
            np.asarray(combined, dtype=np.float64),
        )


def plot_scenario_cdfs(scenario_data: Dict[str, Dict], output_dir: Path) -> None:
    for scenario_name, data in sorted(scenario_data.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        have_data = False

        for source_name, errors in data["relative_errors"].items():
            if not errors:
                continue
            arr = np.sort(np.asarray(errors, dtype=np.float64))
            if arr.size == 0:
                continue

            y = np.linspace(0, 1, arr.size, endpoint=False)
            ax.step(arr, y, where="post", label=source_name, linewidth=2)
            have_data = True

        if not have_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Relative Error")
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 10)
        ax.set_ylabel("CDF")
        ax.set_title(f"CDF of Relative Errors — Scenario: {scenario_name}")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()

        safe_name = scenario_name.replace("/", "_").replace("\\", "_")
        fig.savefig(output_dir / f"cdf_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_box_whiskers(
    scenario_data: Dict[str, Dict],
    sources: Sequence[DataSource],
    output_dir: Path,
) -> None:
    key_data: Dict[Key, Dict[str, List[float]]] = {}

    for data in scenario_data.values():
        for key, values in data["local_values"].items():
            if values:
                key_data.setdefault(key, {}).setdefault("local", []).extend(values)

        for source_name, per_key in data["actual_values"].items():
            for key, value_dict in per_key.items():
                remote_vals = value_dict.get("remote", [])
                if remote_vals:
                    key_data.setdefault(key, {}).setdefault(source_name, []).extend(remote_vals)

    if not key_data:
        print("No data available for aggregated box-and-whisker plot")
        return

    keys = sorted(key_data.keys())
    n_cols = min(3, len(keys))
    n_rows = (len(keys) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes_array = np.atleast_2d(axes)

    for idx, key in enumerate(keys):
        row, col = divmod(idx, n_cols)
        ax = axes_array[row, col]
        labels: List[str] = []
        plot_data: List[List[float]] = []

        for label in ["local"] + [source.name for source in sources]:
            values = key_data[key].get(label)
            if values:
                labels.append(label)
                plot_data.append(values)

        if not plot_data:
            ax.set_visible(False)
            continue

        ax.boxplot(plot_data, tick_labels=labels)
        ax.set_title(f"client={key[0]} op={key[1]}")
        ax.set_ylabel("Duration (ns)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    total_axes = n_rows * n_cols
    for idx in range(len(keys), total_axes):
        row, col = divmod(idx, n_cols)
        axes_array[row, col].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "box_whiskers_all_keys.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_box_whiskers(
    scenario_data: Dict[str, Dict],
    sources: Sequence[DataSource],
    output_dir: Path,
) -> None:
    for scenario_name, data in sorted(scenario_data.items()):
        keys = set(data["local_values"].keys())
        for per_key in data["actual_values"].values():
            keys.update(per_key.keys())

        if not keys:
            continue

        keys = sorted(keys)
        x_values = sorted({key[0] for key in keys})
        y_values = sorted({key[1] for key in keys})
        n_cols = max(1, len(x_values))
        n_rows = max(1, len(y_values))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes_array = np.atleast_2d(axes)

        for ax in axes_array.flat:
            ax.set_visible(False)

        for key in keys:
            row = y_values.index(key[1])
            col = x_values.index(key[0])
            ax = axes_array[row, col]

            plot_data: List[List[float]] = []
            labels: List[str] = []

            for source in sources:
                remote_vals = (
                    data["actual_values"].get(source.name, {}).get(key, {}).get("remote", [])
                )
                if remote_vals:
                    plot_data.append(remote_vals)
                    labels.append(source.name)

            local_vals = data["local_values"].get(key, [])
            if local_vals:
                plot_data.append(local_vals)
                labels.append("local")

            if not plot_data:
                continue

            ax.set_visible(True)
            bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)

            ax.set_title(f"client={key[0]} op={key[1]}")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

        for col_idx, client in enumerate(x_values):
            ax = axes_array[0, col_idx]
            if ax.get_visible():
                ax.set_title(f"client={client}", fontsize=12, fontweight="bold")

        for row_idx, op in enumerate(y_values):
            ax = axes_array[row_idx, 0]
            if ax.get_visible():
                ax.set_ylabel(f"op={op}", fontsize=12, fontweight="bold")

        safe_name = scenario_name.replace("/", "_").replace("\\", "_")
        fig.suptitle(f"Per-key Distributions — {scenario_name}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f"box_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_scenario_csvs(
    scenario_data: Dict[str, Dict],
    sources: Sequence[DataSource],
    output_dir: Path,
) -> None:
    base_csv_dir = output_dir / "sweepcsv"
    base_csv_dir.mkdir(parents=True, exist_ok=True)

    remote_order = [source.name for source in sources]

    for scenario_name, data in sorted(scenario_data.items()):
        keys = set(data["local_values"].keys())
        for per_key in data["actual_values"].values():
            keys.update(per_key.keys())

        if not keys:
            continue

        safe_name = scenario_name.replace("/", "_").replace("\\", "_")
        scenario_dir = base_csv_dir / safe_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        for key in sorted(keys):
            local_vals = data["local_values"].get(key, [])
            remote_map = {
                source.name: data["actual_values"].get(source.name, {}).get(key, {}).get("remote", [])
                for source in sources
            }

            column_order = ["local"] + remote_order
            columns = [local_vals] + [remote_map[name] for name in remote_order]
            max_len = max((len(col) for col in columns), default=0)

            target = scenario_dir / f"key_{key[0]}_{key[1]}.csv"
            with target.open("w", encoding="utf-8") as handle:
                handle.write(",".join(column_order) + "\n")
                for idx in range(max_len):
                    row = []
                    for col in columns:
                        row.append(str(col[idx]) if idx < len(col) else "")
                    handle.write(",".join(row) + "\n")


def main() -> None:
    args = parse_args()

    try:
        local_base = resolve_relative(args.local_base) if args.local_base else detect_default_local_base()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = resolve_relative(args.output_dir) if args.output_dir else SCRIPT_DIR / "sweepfigures"
    if args.trim < 0:
        raise SystemExit("--trim must be non-negative.")

    default_sources = [] if args.no_default_sources else build_default_sources(local_base)
    extra_sources = parse_extra_sources(args.source)
    sources = merge_sources(default_sources, extra_sources)

    if not sources:
        raise SystemExit("No data sources configured. Add --source entries or use the defaults.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Local base: {local_base}")
    for source in sources:
        print(
            f"Configured source '{source.name}': {source.root} / {source.filename} "
            f"(rdma_scale={source.rdma_scale})"
        )

    scenario_data, aggregated = process_scenarios(local_base, sources, args.trim)
    summarize_source_stats(aggregated)

    print("\n=== Generating plots & tables ===")
    plot_relative_error_cdf(aggregated, sources, output_dir)
    plot_scenario_cdfs(scenario_data, output_dir)
    plot_box_whiskers(scenario_data, sources, output_dir)
    plot_scenario_box_whiskers(scenario_data, sources, output_dir)
    write_scenario_csvs(scenario_data, sources, output_dir)

    print(f"Artifacts saved under {output_dir}")


if __name__ == "__main__":
    main()
