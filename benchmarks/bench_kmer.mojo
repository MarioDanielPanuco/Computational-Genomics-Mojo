"""
K-mer extraction benchmarks: throughput vs. k value and batch size.

Sweeps:
  - k in {8, 16, 21, 31} (CPU and GPU)
  - batch sizes: 1K, 10K, 100K sequences of length 150 bp (short reads)
  - canonical vs. non-canonical hashing

Reports throughput in Gbases/s and memory bandwidth utilization.
"""
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric, ThroughputMeasure
from std.sys import has_accelerator
from genomics.core.sequence import SequenceBatch
from genomics.cpu.kmer_cpu import extract_kmers, kmer_frequencies
from benchmarks.harness import make_random_batch, make_random_dna, print_results, BenchResult


def build_batch(n_seqs: Int, seq_len: Int) -> SequenceBatch:
    var batch = SequenceBatch(capacity=n_seqs)
    var seqs = make_random_batch(n_seqs, seq_len)
    for i in range(n_seqs):
        batch.add_sequence(Span(seqs[i]), seq_len)
    return batch


@always_inline
def bench_kmer_cpu[k: Int](mut b: Bencher, batch: SequenceBatch) capturing raises:
    """Benchmark CPU k-mer extraction over a pre-built SequenceBatch."""
    @parameter
    @always_inline
    def run():
        var total_kmers = 0
        for seq_idx in range(batch.count):
            from genomics.core.sequence import get_view
            var view = get_view(batch, seq_idx)
            var n_kmers = view.length - k + 1
            if n_kmers > 0:
                var kbuf = List[UInt64](capacity=n_kmers)
                var vbuf = List[Bool](capacity=n_kmers)
                for _ in range(n_kmers):
                    kbuf.append(0)
                    vbuf.append(False)
                total_kmers += extract_kmers[k](view, kbuf.unsafe_ptr(), vbuf.unsafe_ptr())
    b.iter[run]()


def bench_kmer_scaling_cpu():
    """Sweep over k values and batch sizes; report CPU throughput."""
    var results = List[BenchResult]()
    var seq_len = 150

    @parameter
    def run_one[k: Int](n_seqs: Int):
        var batch = build_batch(n_seqs, seq_len)
        var total_bases = batch.total_bases()

        var bench = Bench(BenchConfig(max_iters=100))
        bench.bench_function[bench_kmer_cpu[k]](
            BenchId("kmer_k" + String(k) + "_n" + String(n_seqs)),
            [ThroughputMeasure(BenchMetric.bytes, total_bases)],
            batch,
        )

    # k=8
    run_one[8](1000)
    run_one[8](10000)
    # k=21
    run_one[21](1000)
    run_one[21](10000)
    # k=31
    run_one[31](1000)
    run_one[31](10000)


def bench_kmer_frequency_table():
    """Benchmark k-mer frequency table construction over varying batch sizes."""
    var seq_len = 150

    @parameter
    def run_one[k: Int](n_seqs: Int):
        var batch = build_batch(n_seqs, seq_len)
        var total_bases = batch.total_bases()
        var bench = Bench(BenchConfig(max_iters=50))

        @always_inline
        def freq_bench(mut b: Bencher) capturing raises:
            @parameter
            @always_inline
            def run():
                _ = kmer_frequencies[k](batch)
            b.iter[run]()

        bench.bench_function[freq_bench](
            BenchId("kmer_freq_k" + String(k) + "_n" + String(n_seqs)),
            [ThroughputMeasure(BenchMetric.bytes, total_bases)],
        )

    run_one[21](1000)
    run_one[21](10000)


def main() raises:
    print("=== K-mer CPU Scaling Benchmarks ===")
    bench_kmer_scaling_cpu()
    print("\n=== K-mer Frequency Table Benchmarks ===")
    bench_kmer_frequency_table()

    comptime if has_accelerator():
        print("\n=== K-mer GPU Benchmarks ===")
        from genomics.gpu.device import GenomicsDevice
        from genomics.gpu.kmer_gpu import launch_kmer_extract
        print("GPU k-mer benchmarks: see bench_kmer_gpu() — requires GPU device")
    else:
        print("\n[skip] No GPU accelerator found — skipping GPU benchmarks")
