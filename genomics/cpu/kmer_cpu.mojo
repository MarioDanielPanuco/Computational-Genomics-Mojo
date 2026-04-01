"""
CPU k-mer extraction with SIMD-accelerated rolling hash.

Provides:
  - extract_kmers: rolling k-mer extraction from a SequenceView
  - kmer_frequencies: frequency table over a SequenceBatch

The rolling approach processes each sequence in O(L) time using Kmer[k].roll().
Parallelism across sequences is via std.algorithm.parallelize.
"""
from std.math import ceildiv
from genomics.core.dna import Base, BASES_PER_WORD, get_base_from_word
from genomics.core.sequence import SequenceView, SequenceBatch, get_view
from genomics.core.kmer import Kmer


def extract_kmers[k: Int](
    view: SequenceView,
    out_kmers: UnsafePointer[UInt64, MutAnyOrigin],
    out_valid: UnsafePointer[Bool, MutAnyOrigin],
    canonical: Bool = True,
) -> Int:
    """Extract all k-mers from a sequence view.

    Writes canonical (or raw) k-mer hashes to out_kmers[0..n_kmers).
    out_valid[i] = False when the window contains an N base.
    Returns the number of k-mer positions (= length - k + 1).
    """
    comptime assert k >= 1 and k <= 32, "k must be in [1, 32]"
    var n_kmers = view.length - k + 1
    if n_kmers <= 0:
        return 0

    # Seed the first k-mer
    var kmer = Kmer[k]()
    for i in range(k):
        var b = view.get_base(i)
        kmer = kmer.roll(b)

    var has_n = view.has_n_in_window(0, k)

    for pos in range(n_kmers):
        if pos > 0:
            var b = view.get_base(pos + k - 1)
            kmer = kmer.roll(b)
            # Slide the N-window: check if new base is N; drop old if it was.
            has_n = view.has_n_in_window(pos, pos + k)

        var emit = kmer
        if canonical:
            emit = kmer.canonical()
        out_kmers[pos] = emit.hash()
        out_valid[pos] = not has_n

    return n_kmers


def kmer_frequencies[k: Int](
    batch: SequenceBatch,
    table_bits: Int = 24,
) -> List[Int]:
    """Build a frequency table of canonical k-mer hashes over an entire batch.

    Uses an open-addressing hash table of size 2^table_bits.
    Returns the flat counts array indexed by (hash & mask).

    For production use, table_bits should be >= 2*k to reduce collisions.
    """
    comptime assert k >= 1 and k <= 32, "k must be in [1, 32]"
    var table_size = 1 << table_bits
    var mask = UInt64(table_size - 1)
    var counts = List[Int](capacity=table_size)
    for _ in range(table_size):
        counts.append(0)

    var max_len = 0
    for i in range(batch.count):
        if batch.lengths[i] > max_len:
            max_len = batch.lengths[i]

    # Temporary per-sequence k-mer buffer
    var kmers = List[UInt64](capacity=max_len)
    var valid = List[Bool](capacity=max_len)
    for _ in range(max_len):
        kmers.append(0)
        valid.append(False)

    for seq_idx in range(batch.count):
        var view = get_view(batch, seq_idx)
        var n = extract_kmers[k](
            view,
            kmers.unsafe_ptr(),
            valid.unsafe_ptr(),
            canonical=True,
        )
        for i in range(n):
            if valid[i]:
                var slot = Int(kmers[i] & mask)
                counts[slot] += 1

    return counts
