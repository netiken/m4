#!/usr/bin/env python3
"""
Normalize ts_ns and start_ns across each scenario directory so timestamps are relative to the
first line of c1/log_0.txt in that scenario.

Input layout (under the workspace root):
- <scenario>/c1/log_*.txt
- <scenario>/server/worker_log_*.txt
- <scenario>/server/dist_sample.txt (ignored)

Output:
- Write normalized files alongside originals with suffix ".normalized.txt".
- Preserve line order and all fields, only rewrite ts_ns/start_ns values.

Rules:
- Base timestamp is taken from the first line of <scenario>/c1/log_0.txt.
  - Prefer ts_ns if present; otherwise use start_ns.
  - If neither exists, the scenario is skipped with a warning.
- Any ts_ns/start_ns = X becomes X - base + 1 (so the first line becomes 1).
- Lines without these fields are preserved verbatim.
"""

import os
import re
import sys
from typing import Dict, Optional, Tuple


WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))


FIELD_PATTERNS = {
    "ts_ns": re.compile(r"\bts_ns=(\d+)\b"),
    "start_ns": re.compile(r"\bstart_ns=(\d+)\b"),
}


def find_scenario_dirs(root_dir: str) -> Dict[str, str]:
    """Return mapping of scenario name -> absolute path. A scenario is any directory with subdirs c1 and server."""
    scenarios: Dict[str, str] = {}
    try:
        for entry in os.listdir(root_dir):
            abs_path = os.path.join(root_dir, entry)
            if not os.path.isdir(abs_path):
                continue
            c1_dir = os.path.join(abs_path, "c1")
            server_dir = os.path.join(abs_path, "server")
            if os.path.isdir(c1_dir) and os.path.isdir(server_dir):
                scenarios[entry] = abs_path
    except FileNotFoundError:
        pass
    return scenarios


def collect_files_for_scenario(scenario_dir: str) -> Tuple[list, list]:
    """Return (client_files, server_files) lists, absolute paths, excluding dist_sample.txt.

    Clients: any subdir starting with 'c' (e.g., c1, c2, ...), include all *.txt.
    Servers: files under server/ except dist_sample.txt.
    """
    client_files = []
    server_files = []
    # Collect client files across c*/
    try:
        for entry in sorted(os.listdir(scenario_dir)):
            client_dir = os.path.join(scenario_dir, entry)
            if not os.path.isdir(client_dir) or not entry.startswith("c"):
                continue
            for name in sorted(os.listdir(client_dir)):
                if not name.endswith(".txt"):
                    continue
                if name.endswith(".normalized.txt"):
                    continue
                client_files.append(os.path.join(client_dir, name))
    except FileNotFoundError:
        pass

    # Collect server files
    server_dir = os.path.join(scenario_dir, "server")
    if os.path.isdir(server_dir):
        for name in sorted(os.listdir(server_dir)):
            if not name.endswith(".txt"):
                continue
            if name == "dist_sample.txt":
                continue
            if name.endswith(".normalized.txt"):
                continue
            server_files.append(os.path.join(server_dir, name))
    return client_files, server_files


def get_base_from_clients(scenario_dir: str) -> Optional[int]:
    """Return a base timestamp across all clients: min of first-line ts_ns/start_ns among c*/log_*.txt."""
    best: Optional[int] = None
    try:
        for entry in sorted(os.listdir(scenario_dir)):
            client_dir = os.path.join(scenario_dir, entry)
            if not os.path.isdir(client_dir) or not entry.startswith("c"):
                continue
            for name in sorted(os.listdir(client_dir)):
                if not name.endswith(".txt") or name.endswith(".normalized.txt"):
                    continue
                path = os.path.join(client_dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        first_line = f.readline()
                        if not first_line:
                            continue
                        m_ts = FIELD_PATTERNS["ts_ns"].search(first_line)
                        m_start = FIELD_PATTERNS["start_ns"].search(first_line)
                        cand = int(m_ts.group(1)) if m_ts else (int(m_start.group(1)) if m_start else None)
                        if cand is not None and (best is None or cand < best):
                            best = cand
                except FileNotFoundError:
                    continue
    except FileNotFoundError:
        return None
    return best


def rewrite_line(line: str, base: int) -> str:
    """Rewrite ts_ns/start_ns by subtracting base and adding 1."""
    def _replace(match: re.Match) -> str:
        original_val = int(match.group(1))
        normalized = original_val - base + 1
        return match.group(0).replace(match.group(1), str(normalized))

    for field, patt in FIELD_PATTERNS.items():
        # Replace all occurrences if any
        line = patt.sub(_replace, line)
    return line


def normalize_files(files: list, base: int) -> None:
    for path in files:
        out_path = f"{path[:-4]}.normalized.txt" if path.endswith(".txt") else f"{path}.normalized"
        try:
            with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    fout.write(rewrite_line(line, base))
        except FileNotFoundError:
            continue


def main() -> int:
    root = WORKSPACE_ROOT
    scenarios = find_scenario_dirs(root)
    if not scenarios:
        print("No scenario directories found.")
        return 1

    any_done = False
    for name, scenario_dir in sorted(scenarios.items()):
        client_files, server_files = collect_files_for_scenario(scenario_dir)
        all_files = client_files + server_files
        if not all_files:
            continue
        base = get_base_from_clients(scenario_dir)
        if base is None:
            print(f"Scenario {name}: WARNING: could not determine base from clients; skipping")
            continue
        normalize_files(all_files, base)
        any_done = True
        print(f"Scenario {name}: base={base} -> wrote normalized copies for {len(all_files)} files")

    if not any_done:
        print("Nothing to do.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


