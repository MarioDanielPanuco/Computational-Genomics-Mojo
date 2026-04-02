"""
Memory and transfer benchmarks: host<->device bandwidth, encoding overhead,
and packed vs. ASCII representation trade-offs.
"""
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric, ThroughputMeasure
from std.sys import has_accelerator
from genomics.core.sequence import SequenceBatch
from benchmarks.harness import make_random_batch


def bench_encoding_overhead(n_seqs: Int, seq_len: Int) raises:
    """Measure the time spent encoding ASCII -> 2-bit during batch construction."""
    var seqs = make_random_batch(n_seqs, seq_len)
    var bench = Bench(BenchConfig(max_iters=200))

    @always_inline
    def encode_bench(mut b: Bencher) capturing raises:
        @parameter
        @always_inline
        def run():
            var batch = SequenceBatch(capacity=n_seqs)
            for i in range(n_seqs):
                batch.add_sequence(Span(seqs[i]), seq_len)
        b.iter[run]()

    var total_bases = n_seqs * seq_len
    bench.bench_function[encode_bench](
        BenchId("encode_n" + String(n_seqs) + "_len" + String(seq_len)),
        [ThroughputMeasure(BenchMetric.bytes, total_bases)],
    )
    bench.dump_report()


def _fmt_mb(bytes: Int) -> String:
    """Format byte count as X.X MB string."""
    var tenths = Int(Float64(bytes) / 104857.6 + 0.5)  # tenths of MB, rounded
    return String(tenths // 10) + "." + String(tenths % 10) + " MB"


def _fmt_ratio(a: Int, b: Int) -> String:
    """Format a/b as X.XX string."""
    var hundredths = Int(Float64(a) / Float64(b) * 100.0 + 0.5)
    return String(hundredths // 100) + "." + String((hundredths % 100) // 10) + String(hundredths % 10)


def bench_packed_density() raises:
    """Compare memory footprint: ASCII vs. 2-bit packed representation."""
    var n_seqs = 10000
    var seq_len = 150
    var ascii_bytes = n_seqs * seq_len
    var packed_bytes = n_seqs * ((seq_len + 31) // 32) * 8  # UInt64 words

    print("ASCII representation:  " + String(ascii_bytes) + " bytes (" + _fmt_mb(ascii_bytes) + ")")
    print("Packed 2-bit:          " + String(packed_bytes) + " bytes (" + _fmt_mb(packed_bytes) + ")")
    print("Compression ratio:     " + _fmt_ratio(ascii_bytes, packed_bytes) + "x")


def main() raises:
    print("=== Encoding Overhead ===")
    bench_encoding_overhead(1000, 150)
    bench_encoding_overhead(10000, 150)
    bench_encoding_overhead(1000, 1000)

    print("\n=== Memory Density ===")
    bench_packed_density()

    comptime if has_accelerator():
        print("\n=== GPU Transfer Bandwidth ===")
        from genomics.gpu.device import GenomicsDevice
        print("GPU upload benchmark: upload_batch_packed — requires GPU device")
    else:
        print("\n[skip] No GPU accelerator found")
