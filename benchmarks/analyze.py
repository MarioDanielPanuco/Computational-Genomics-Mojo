#!/usr/bin/env python3
"""
analyze.py — plot benchmark trends from benchmarks/results/*.jsonl

Usage:
    python benchmarks/analyze.py                    # all hosts, all benches
    python benchmarks/analyze.py --host m1-mac      # filter by host
    python benchmarks/analyze.py --bench kmer       # filter bench name substring
    python benchmarks/analyze.py --list             # list available benchmarks

Requires: pandas, matplotlib
    pip install pandas matplotlib
"""
import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_records(host_filter=None, bench_filter=None):
    records = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if host_filter and rec.get("host") != host_filter:
                    continue
                if bench_filter and bench_filter not in rec.get("bench", ""):
                    continue
                records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark trends")
    parser.add_argument("--host", help="Filter by host name")
    parser.add_argument("--bench", help="Filter bench name substring")
    parser.add_argument("--list", action="store_true", help="List available benchmarks and exit")
    parser.add_argument("--metric", default="throughput_gbs",
                        choices=["throughput_gbs", "latency_ms"],
                        help="Metric to plot (default: throughput_gbs)")
    parser.add_argument("--output", help="Save plot to file instead of showing")
    args = parser.parse_args()

    records = load_records(host_filter=args.host, bench_filter=args.bench)

    if not records:
        print("No records found.", file=sys.stderr)
        print(f"  results dir: {RESULTS_DIR}", file=sys.stderr)
        print("  Run ./benchmarks/record.sh to generate data.", file=sys.stderr)
        sys.exit(1)

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Install dependencies: pip install pandas matplotlib", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"].fillna("00:00"))

    if args.list:
        print("Available benchmarks:")
        for bench in sorted(df["bench"].unique()):
            hosts = ", ".join(sorted(df[df["bench"] == bench]["host"].unique()))
            print(f"  {bench}  (hosts: {hosts})")
        return

    metric = args.metric
    metric_label = "Throughput (GB/s or GElems/s)" if metric == "throughput_gbs" else "Latency (ms)"

    benches = sorted(df["bench"].unique())
    hosts = sorted(df["host"].unique())
    n = len(benches)

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f"Benchmark trends — {metric_label}", fontsize=14, fontweight="bold")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, bench in enumerate(benches):
        ax = axes[idx // cols][idx % cols]
        bdf = df[df["bench"] == bench].sort_values("datetime")

        for hi, host in enumerate(hosts):
            hdf = bdf[bdf["host"] == host]
            if hdf.empty:
                continue
            color = colors[hi % len(colors)]
            ax.plot(hdf["datetime"], hdf[metric], marker="o", label=host,
                    color=color, linewidth=1.5, markersize=4)
            # Annotate last point with commit
            last = hdf.iloc[-1]
            ax.annotate(last.get("commit", "")[:7],
                        xy=(last["datetime"], last[metric]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=color)

        ax.set_title(bench, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        if hi > 0 or len(hosts) > 1:
            ax.legend(fontsize=7)

    # Hide unused subplot panels
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()

    # Also print a summary table
    print(f"\n{'Benchmark':<35} {'Host':<12} {'Runs':>4}  {metric_label}")
    print("-" * 70)
    for bench in benches:
        for host in hosts:
            sub = df[(df["bench"] == bench) & (df["host"] == host)]
            if sub.empty:
                continue
            latest = sub.sort_values("datetime").iloc[-1]
            val = latest[metric]
            nruns = len(sub)
            print(f"{bench:<35} {host:<12} {nruns:>4}  {val:.4f}")


if __name__ == "__main__":
    main()
