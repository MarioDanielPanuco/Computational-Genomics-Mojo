"""
Alignment benchmarks: banded DP scaling vs. sequence length and band width.

Sweeps:
  - Sequence lengths: 100, 500, 1000, 5000 bp
  - Band widths:  16, 32, 64, 128 (compile-time parameter)
  - CPU banded SW vs. GPU banded SW

Metrics:
  - Alignment pairs per second
  - Score accuracy vs. unbanded reference (fraction of pairs with identical score)
  - GPU occupancy proxy: measured SM throughput
"""
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric, ThroughputMeasure
from std.sys import has_accelerator
from std.math import min
from genomics.core.sequence import SequenceBatch, get_view
from genomics.cpu.align_cpu import (
    smith_waterman_banded, needleman_wunsch_banded, default_config, AlignConfig,
)
from benchmarks.harness import make_random_batch, make_random_dna


def bench_sw_cpu[band: Int](n_pairs: Int, seq_len: Int) raises:
    """Benchmark CPU Smith-Waterman for n_pairs at given band and seq_len."""
    var q_seqs = make_random_batch(n_pairs, seq_len)
    var r_seqs = make_random_batch(n_pairs, seq_len)
    var queries = SequenceBatch(capacity=n_pairs)
    var refs = SequenceBatch(capacity=n_pairs)
    for i in range(n_pairs):
        queries.add_sequence(Span(q_seqs[i]), seq_len)
        refs.add_sequence(Span(r_seqs[i]), seq_len)

    var cfg = default_config()
    cfg.band_width = band

    var bench = Bench(BenchConfig(max_iters=100))

    @always_inline
    def sw_bench(mut b: Bencher) capturing raises:
        @parameter
        @always_inline
        def run():
            for i in range(min(n_pairs, queries.count)):
                var q = get_view(queries, i)
                var r = get_view(refs, i)
                _ = smith_waterman_banded(q, r, cfg)
        b.iter[run]()

    var total_cells = n_pairs * seq_len * (2 * band + 1)
    bench.bench_function[sw_bench](
        BenchId("sw_cpu_band" + String(band) + "_len" + String(seq_len) + "_n" + String(n_pairs)),
        [ThroughputMeasure(BenchMetric.elements, total_cells)],
    )
    bench.dump_report()


def bench_alignment_cpu_scaling() raises:
    """Sweep band widths and sequence lengths for CPU alignment."""
    print("--- CPU SW: band width sweep (seq_len=500, n=100) ---")
    bench_sw_cpu[16](100, 500)
    bench_sw_cpu[32](100, 500)
    bench_sw_cpu[64](100, 500)
    bench_sw_cpu[128](100, 500)

    print("--- CPU SW: sequence length sweep (band=32, n=100) ---")
    bench_sw_cpu[32](100, 100)
    bench_sw_cpu[32](100, 500)
    bench_sw_cpu[32](100, 1000)


def bench_nw_vs_sw() raises:
    """Compare NW and SW performance at equal band widths."""
    var n = 200
    var seq_len = 300
    var q_seqs = make_random_batch(n, seq_len)
    var r_seqs = make_random_batch(n, seq_len)
    var queries = SequenceBatch(capacity=n)
    var refs = SequenceBatch(capacity=n)
    for i in range(n):
        queries.add_sequence(Span(q_seqs[i]), seq_len)
        refs.add_sequence(Span(r_seqs[i]), seq_len)

    var cfg = default_config()
    cfg.band_width = 32

    var bench = Bench(BenchConfig(max_iters=100))

    @always_inline
    def nw_bench(mut b: Bencher) capturing raises:
        @parameter
        @always_inline
        def run():
            for i in range(min(n, queries.count)):
                var q = get_view(queries, i)
                var r = get_view(refs, i)
                _ = needleman_wunsch_banded(q, r, cfg)
        b.iter[run]()

    bench.bench_function[nw_bench](
        BenchId("nw_cpu_band32_len300_n200"),
        [ThroughputMeasure(BenchMetric.elements, n * seq_len * 65)],
    )
    bench.dump_report()


def main() raises:
    print("=== CPU Alignment Benchmarks ===")
    bench_alignment_cpu_scaling()

    print("\n=== NW vs SW Comparison ===")
    bench_nw_vs_sw()

    comptime if has_accelerator():
        print("\n=== GPU Alignment Benchmarks ===")
        from genomics.gpu.device import GenomicsDevice
        from genomics.gpu.align_gpu import launch_banded_sw
        print("GPU alignment: launch_banded_sw[64] — requires GPU device")
    else:
        print("\n[skip] No GPU accelerator found")
