"""
Memory and transfer benchmarks: host<->device bandwidth, encoding overhead,
and packed vs. ASCII representation trade-offs.
"""
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric, ThroughputMeasure
from std.sys import has_accelerator
from genomics.core.sequence import SequenceBatch
from benchmarks.harness import make_random_batch


def bench_encoding_overhead(n_seqs: Int, seq_len: Int):
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


def bench_packed_density():
    """Compare memory footprint: ASCII vs. 2-bit packed representation."""
    var n_seqs = 10000
    var seq_len = 150
    var ascii_bytes = n_seqs * seq_len
    var packed_bytes = n_seqs * ((seq_len + 31) // 32) * 8  # UInt64 words

    print("ASCII representation:  {:>10} bytes ({:.1f} MB)".format(
        ascii_bytes, Float64(ascii_bytes) / 1048576.0))
    print("Packed 2-bit:          {:>10} bytes ({:.1f} MB)".format(
        packed_bytes, Float64(packed_bytes) / 1048576.0))
    print("Compression ratio:     {:.2f}x".format(
        Float64(ascii_bytes) / Float64(packed_bytes)))


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

        @always_inline
        def transfer_bench(mut b: Bencher) capturing raises:
            var device = GenomicsDevice()
            var n_seqs = 10000
            var seq_len = 150
            var seqs = make_random_batch(n_seqs, seq_len)
            var batch = SequenceBatch(capacity=n_seqs)
            for i in range(n_seqs):
                batch.add_sequence(Span(seqs[i]), seq_len)

            @parameter
            @always_inline
            def run(ctx: DeviceContext) raises:
                _ = device.upload_batch_packed(batch)
                device.synchronize()

            b.iter_custom[run](device.ctx)

        var bench = Bench(BenchConfig(max_iters=50))
        bench.bench_function[transfer_bench](
            BenchId("gpu_upload_10k_seqs"),
            [ThroughputMeasure(BenchMetric.bytes, 10000 * 150)],
        )
    else:
        print("\n[skip] No GPU accelerator found")
