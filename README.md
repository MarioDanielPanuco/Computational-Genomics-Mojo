# Computational Genomics

A performance-oriented genomics library written in [Mojo](https://www.modular.com/mojo), built around a compact 2-bit DNA representation and matching CPU reference implementations and GPU kernels for the core primitives of genomic signal processing: k-mer analysis, pairwise alignment, and sliding-window sequence statistics.

> **Status:** early / experimental (`v0.1.0`). The CPU paths are implemented and unit-tested against real genomes; GPU kernels are written but **untested on hardware** (no NVIDIA/AMD device in CI). Built on **MAX nightly** Mojo, whose syntax changes frequently — see [Mojo syntax notes](#a-note-on-mojo-syntax).

## Why this exists

Genomic workloads are dominated by a handful of tight inner loops over billions of nucleotides. This project explores how far a modern systems language with first-class SIMD and GPU support can push those loops, while keeping a readable CPU reference alongside every accelerated kernel so results can be validated. It doubles as a study in data-parallel data structures (structure-of-arrays batching, bit-packed encodings) and MLIR-backed benchmarking.

## Core ideas

### 2-bit packed DNA

DNA is stored two bits per base, 32 bases per `UInt64`, left-aligned (most-significant pair = leftmost base):

```
A = 0b00   C = 0b01   G = 0b10   T = 0b11
```

- **Reverse complement** is a bitwise NOT followed by a 5-round 2-bit-granularity bit-reversal — no per-base branching.
- **Ambiguous bases (`N`, gaps, IUPAC codes)** are stored as `A` in the packed word and flagged in a **parallel `NMask` bitmask** (one bit per position). Callers check `NMask` to skip windows containing ambiguity, so the hot path never needs a fifth symbol.

### Structure-of-arrays batches

`SequenceBatch` concatenates every sequence's packed words into shared `List` buffers with per-sequence `offsets` and `lengths`. This SoA layout is what makes the GPU path viable: assign one thread-block per sequence and it reads a contiguous, coalesced run of words. `SequenceView` is a per-sequence slice (it copies its words) used by the CPU code paths.

### Compile-time k-mers

`Kmer[k]` carries `k` as a compile-time parameter (`1 ≤ k ≤ 32`) and stores the k-mer left-aligned in a `UInt64`. It supports `roll(base)` for O(1) sliding, `canonical()` (= `min(kmer, reverse_complement)`) for strand-agnostic indexing, and a MurmurHash3 finalizer for hashing.

## Layout

```
genomics/
├── core/                  # foundational types
│   ├── dna.mojo           # 2-bit encoding, NMask, complement / reverse-complement
│   ├── sequence.mojo      # SequenceBatch (SoA) + SequenceView
│   └── kmer.mojo          # Kmer[k]: roll, canonical, hash
├── cpu/                   # reference implementations (validate the GPU kernels)
│   ├── kmer_cpu.mojo      # extract_kmers[k], kmer_frequencies[k]
│   ├── align_cpu.mojo     # banded Smith-Waterman, Needleman-Wunsch, gap-affine WFA
│   └── sliding_window.mojo# GC content, Shannon entropy, linguistic complexity
└── gpu/                   # NVIDIA (CUDA) / AMD (ROCm) kernels — untested on hardware
    ├── device.mojo        # GenomicsDevice: DeviceContext wrapper + upload helpers
    ├── kmer_gpu.mojo      # kmer_extract_kernel[k]
    ├── align_gpu.mojo     # banded SW, WFA, gap-affine WFA kernels
    └── sliding_window_gpu.mojo # GC + entropy kernels

benchmarks/               # harness.mojo + bench_{kmer,alignment,memory}.mojo, record.sh
tests/                    # unit tests + real-genome fixtures (PhiX174, SARS-CoV-2, E. coli)
```

## What's implemented

| Domain | CPU | GPU kernel | Notes |
|---|---|---|---|
| DNA encode / decode / revcomp | ✅ | — | `NMask` tracks ambiguous bases |
| K-mer extraction (canonical hashes) | ✅ | ✅ | one block per sequence, threads stride positions |
| K-mer frequency table | ✅ | — | flat open-addressing hash table (`2^table_bits`) |
| Smith-Waterman (banded, local) | ✅ | ✅ | 3-row ring buffer; affine gaps |
| Needleman-Wunsch (banded, global) | ✅ | — | |
| Gap-affine WFA (global, cost-model) | ✅ | ✅ | furthest-reaching-point wavefronts, O(s²) |
| Sliding GC content | ✅ | ✅ | prefix-sum over packed-word GC popcount |
| Sliding Shannon entropy | ✅ | ✅ | sliding histogram (CPU) |
| Linguistic k-mer complexity | ✅ | — | |

**Known limitations**
- CIGAR traceback is stubbed — alignment functions return scores and endpoints, not the aligned path (`cigar` is empty).
- GPU kernels compile against MAX but have **not been run on a real accelerator**. Apple M-series is not supported by Mojo's GPU path; targets are NVIDIA (CUDA) and AMD (ROCm).
- GPU GC kernel uses a single-thread prefix sum; the entropy kernel recomputes its histogram per thread. Both are correctness-first, not yet optimized.
- No GPU benchmark wiring yet — bench files print a placeholder when no accelerator is present.

## Getting started

The project uses [pixi](https://pixi.sh) with Modular's MAX nightly channel.

```sh
# Run the full test suite (DNA, sequence, k-mer, alignment, sliding window, real-data)
pixi run test

# Run a single test file
pixi run mojo run tests/test_kmer.mojo

# Benchmarks
pixi run bench-kmer      # k-mer throughput
pixi run bench-align     # alignment throughput
pixi run bench-memory    # encoding overhead

# Record benchmarks to JSONL (benchmarks/results/YYYY-MM-DD_<host>.jsonl)
BENCH_HOST=my-machine ./benchmarks/record.sh

# Format sources (requires the dev environment)
pixi run --env dev fmt

# Build an installable package → genomics.mojopkg
pixi run build
```

### A tiny example

```mojo
from genomics.core.sequence import SequenceBatch, get_view
from genomics.cpu.sliding_window import gc_content_sliding

var batch = SequenceBatch()
var seq = "ACGTACGTNNGGCC".as_bytes()
batch.add_sequence(seq, len(seq))

var view = get_view(batch, 0)
# gc_content_sliding[window] writes (view.length - window + 1) floats
# into a caller-supplied buffer.
```

See `tests/` for worked usage of every public function, and `tests/fasta_loader.mojo` for Python-interop helpers that load FASTA fixtures.

## Benchmarking

`benchmarks/harness.mojo` provides LCG-seeded random-DNA generators and result printing on top of Mojo's `Bench` framework. `record.sh` parses the markdown report from `dump_report()` and appends one JSONL row per benchmark to `benchmarks/results/`; `analyze.py` summarizes those files. Field definitions live in `benchmarks/results/schema.md`.

Indicative single-thread CPU numbers on an M1 Mac (`2026-04-02`, throughput in GB/s of input processed):

| Benchmark | Throughput |
|---|---|
| k-mer extract (k=21, 10k×150bp reads) | ~0.14 GB/s |
| k-mer frequency table (k=21) | ~0.05 GB/s |
| Banded SW (band=32, 500bp) | ~0.22 GB/s |
| Banded NW (band=32, 300bp) | ~0.31 GB/s |
| 2-bit encode (10k×150bp) | ~0.26 GB/s |

These are baselines for tracking regressions, not tuned results.

## Testing against real genomes

`tests/fixtures/` holds real sequences — PhiX174 (NC_001422.1, 5386 bp), SARS-CoV-2 (NC_045512.2, 29903 bp), and an *E. coli* 16S region (V00613.1). `pixi run test-real-data` exercises the library end-to-end on these via the FASTA loader. Re-download with `bash tests/fixtures/download.sh`.

## Continuous integration

`.github/workflows/ci.yml` runs `pixi run test` on every push and pull request (Ubuntu, CPU-only), and runs the benchmarks in a non-blocking step. `benchmark.yml` handles benchmark recording.

## A note on Mojo syntax

This project tracks **MAX nightly**, where the language has diverged substantially from older tutorials and from most pretrained knowledge. Key differences used throughout the codebase:

| Old | Current |
|---|---|
| `fn` | `def` |
| `alias X = …` | `comptime X = …` |
| `let x` / `var x` | `var x` |
| `inout` | `mut` |
| `inout self` in `__init__` | `out self` |
| `@value` | `@fieldwise_init` + explicit traits |
| `from sys import …` | `from std.sys import …` |
| `constrained(…)` | `comptime assert …` |

`def` does **not** imply `raises`; numeric conversions are always explicit (`Float32(i)`, `Int(u)`); `List` uses bracket literals (`[1, 2, 3]`). See `CLAUDE.md` and `.agents/skills/mojo-syntax/` for the full set of rules.

## License

MIT © 2026 Mario Panuco. See [LICENSE](LICENSE).
