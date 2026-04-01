"""
Benchmark harness: timing, throughput measurement, and result reporting.

Uses std.benchmark for accurate wall-clock measurement with warm-up and
statistical aggregation. GPU benchmarks use iter_custom with a DeviceContext
to include kernel launch and synchronize overhead.
"""
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric, ThroughputMeasure
from std.sys import has_accelerator


@fieldwise_init
struct BenchResult(Movable, Writable):
    """Summary of a single benchmark run."""
    var name: String
    var device: String      # "cpu" or "gpu"
    var throughput_gbps: Float64   # Gbases per second
    var mean_ns: Float64           # mean latency in nanoseconds
    var total_bytes: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "[", self.device, "] ", self.name,
            "  throughput=", self.throughput_gbps, " Gbases/s",
            "  latency=", self.mean_ns, " ns",
            "  data=", self.total_bytes // (1024 * 1024), " MB",
        )


def print_results(results: List[BenchResult]):
    """Print a formatted table of benchmark results."""
    print("=" * 70)
    print("{:<30} {:<5} {:>14} {:>12}".format("Name", "Dev", "Throughput", "Latency"))
    print("-" * 70)
    for i in range(len(results)):
        var r = results[i]
        print("{:<30} {:<5} {:>12.3f} Gb/s {:>10.1f} ns".format(
            r.name, r.device, r.throughput_gbps, r.mean_ns))
    print("=" * 70)


def make_random_dna(length: Int) -> List[UInt8]:
    """Generate a pseudo-random DNA sequence of the given length (ASCII A/C/G/T)."""
    var bases = [UInt8(65), UInt8(67), UInt8(71), UInt8(84)]  # A, C, G, T
    var seq = List[UInt8](capacity=length)
    # Simple LCG for reproducible pseudo-random sequence
    var state: UInt64 = 0x123456789ABCDEF0
    for _ in range(length):
        state = state * 6364136223846793005 + 1442695040888963407
        var idx = Int((state >> 62) & 3)
        seq.append(bases[idx])
    return seq


def make_random_batch(n_seqs: Int, seq_len: Int) -> List[List[UInt8]]:
    """Generate n_seqs pseudo-random DNA sequences each of length seq_len."""
    var batch = List[List[UInt8]](capacity=n_seqs)
    for i in range(n_seqs):
        var state: UInt64 = UInt64(i + 1) * 0x9E3779B97F4A7C15
        var seq = List[UInt8](capacity=seq_len)
        var bases = [UInt8(65), UInt8(67), UInt8(71), UInt8(84)]
        for _ in range(seq_len):
            state = state * 6364136223846793005 + 1442695040888963407
            var idx = Int((state >> 62) & 3)
            seq.append(bases[idx])
        batch.append(seq^)
    return batch
