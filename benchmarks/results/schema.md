# Benchmark Results Schema

Each run appends one JSON object per benchmark to a `.jsonl` file named `YYYY-MM-DD_<host>.jsonl`.

## Fields

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `date` | string (ISO 8601) | `"2026-04-01"` | Date of the run |
| `time` | string (HH:MM) | `"14:32"` | Local time of the run |
| `host` | string | `"m1-mac"` | Machine identifier (set via `BENCH_HOST` env var or `hostname`) |
| `commit` | string | `"abc1234"` | Short git commit hash |
| `bench` | string | `"kmer_k21_n10000"` | Benchmark name from `Bench.dump_report()` |
| `device` | string | `"cpu"` or `"gpu"` | Compute device |
| `throughput_gbs` | float | `0.144` | Throughput in GB/s (DataMovement) or GElems/s (elements) |
| `latency_ms` | float | `10.43` | Mean iteration latency in milliseconds |
| `iters` | int | `100` | Number of benchmark iterations |

## Example Record

```json
{"date":"2026-04-01","time":"14:32","host":"m1-mac","commit":"abc1234","bench":"kmer_k21_n10000","device":"cpu","throughput_gbs":0.144,"latency_ms":10.43,"iters":100}
```

## File Naming

`benchmarks/results/YYYY-MM-DD_<host>.jsonl`

Multiple runs on the same day and host append to the same file. Each line is one benchmark entry. This lets you track regressions across commits on the same machine.
