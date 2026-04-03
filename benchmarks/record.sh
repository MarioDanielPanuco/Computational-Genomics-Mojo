#!/usr/bin/env bash
# record.sh — run all benchmarks and append results to benchmarks/results/
#
# Usage:
#   ./benchmarks/record.sh                 # auto-detect host
#   BENCH_HOST=desktop ./benchmarks/record.sh  # override host label
#   BENCH_HOST=gpu-box ./benchmarks/record.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"

# ---- Identity ---------------------------------------------------------------
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M)
HOST="${BENCH_HOST:-$(hostname -s 2>/dev/null || echo "unknown")}"
COMMIT=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
OUTFILE="$RESULTS_DIR/${DATE}_${HOST}.jsonl"

echo "Recording benchmarks → $OUTFILE"
echo "  host=$HOST  commit=$COMMIT  date=$DATE  time=$TIME"

# ---- Parser -----------------------------------------------------------------
# Parses one Bench.dump_report() markdown table line into a JSON record.
# Input format (after "Running <bench_name>"):
#   | bench_name | met (ms) | iters | throughput_label |
#   | ---------- | -------- | ----- | --------------- |
#   | value      | 10.43    | 100   | 0.144           |
#
# Arguments: $1=bench_name, $2=device, raw table on stdin
parse_bench_output() {
    local bench_name="$1"
    local device="$2"
    # Extract the data row (third line of the markdown table)
    local data_line
    data_line=$(grep -v '^$' | tail -1)
    # Parse pipe-separated fields: | name | met_ms | iters | throughput |
    local met_ms iters throughput
    met_ms=$(echo "$data_line" | awk -F'|' '{gsub(/ /,"",$3); print $3}')
    iters=$(echo "$data_line" | awk -F'|' '{gsub(/ /,"",$4); print $4}')
    throughput=$(echo "$data_line" | awk -F'|' '{gsub(/ /,"",$5); print $5}')

    printf '{"date":"%s","time":"%s","host":"%s","commit":"%s","bench":"%s","device":"%s","throughput_gbs":%s,"latency_ms":%s,"iters":%s}\n' \
        "$DATE" "$TIME" "$HOST" "$COMMIT" \
        "$bench_name" "$device" "$throughput" "$met_ms" "$iters"
}

# ---- Run benchmarks ---------------------------------------------------------
run_bench_file() {
    local mojo_file="$1"
    local device="$2"
    echo ""
    echo "=== $(basename "$mojo_file") ==="

    # Run and capture output, tee to terminal
    local raw
    raw=$(cd "$PROJECT_ROOT" && pixi run mojo run "$mojo_file" 2>/dev/null)
    echo "$raw"

    # Parse each benchmark block (triggered by "Running <name>" line)
    local current_bench=""
    local table_lines=""
    local in_table=0

    while IFS= read -r line; do
        if [[ "$line" =~ ^Running\ (.+)$ ]]; then
            current_bench="${BASH_REMATCH[1]}"
            table_lines=""
            in_table=0
        elif [[ "$line" =~ ^\|.*\|.*\| ]] && [[ -n "$current_bench" ]]; then
            table_lines+="$line"$'\n'
            in_table=1
        elif [[ $in_table -eq 1 ]] && [[ -z "$line" || ! "$line" =~ ^\| ]]; then
            # End of table — parse it
            if [[ -n "$table_lines" ]]; then
                record=$(echo "$table_lines" | parse_bench_output "$current_bench" "$device")
                if [[ -n "$record" ]]; then
                    echo "$record" >> "$OUTFILE"
                fi
            fi
            table_lines=""
            in_table=0
            current_bench=""
        fi
    done <<< "$raw"

    # Handle table at end of output with no trailing blank line
    if [[ $in_table -eq 1 && -n "$table_lines" && -n "$current_bench" ]]; then
        record=$(echo "$table_lines" | parse_bench_output "$current_bench" "$device")
        if [[ -n "$record" ]]; then
            echo "$record" >> "$OUTFILE"
        fi
    fi
}

# Determine device label (gpu if has_accelerator returns true, else cpu)
# For now we tag all runs as "cpu" on M1; override with BENCH_DEVICE env var
DEVICE="${BENCH_DEVICE:-cpu}"

run_bench_file "benchmarks/bench_kmer.mojo" "$DEVICE"
run_bench_file "benchmarks/bench_alignment.mojo" "$DEVICE"
run_bench_file "benchmarks/bench_memory.mojo" "$DEVICE"

echo ""
echo "Done. Records written to: $OUTFILE"
echo "Lines appended: $(grep -c '' "$OUTFILE" 2>/dev/null || echo 0)"
