#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def load_fct_ns(path):
    fct = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                #if parts[-1] == "s2c":
                    fct_ns = int(parts[-2])  # last column = fct_ns
                    print(fct_ns)
                    fct.append(fct_ns)
            except ValueError:
                continue
    print(f"Loaded {len(fct)} FCT samples")
    return np.array(fct, dtype=np.float64)

def main():
    ap = argparse.ArgumentParser(description="Plot CDF of flow completion time (ms) from flows.txt")
    ap.add_argument("input", help="Path to flows.txt")
    ap.add_argument("--out", default="fct_cdf.png", help="Output image filename (default: fct_cdf.png)")
    ap.add_argument("--title", default="FCT CDF", help="Plot title")
    args = ap.parse_args()

    fct_ns = load_fct_ns(args.input)
    if fct_ns.size == 0:
        print("No valid FCT samples found.")
        return

    fct_ms = fct_ns  # ns -> ms
    fct_ms.sort()
    y = np.arange(1, fct_ms.size + 1) / fct_ms.size

    plt.figure(figsize=(6,4))
    plt.plot(fct_ms, y, lw=2)
    plt.xlabel("Flow completion time (ns)")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:,.0f}"))
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ylabel("CDF")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()