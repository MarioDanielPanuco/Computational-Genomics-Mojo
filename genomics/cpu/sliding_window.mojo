"""
CPU sliding-window transforms: GC content, sequence entropy, complexity.

All transforms operate on SequenceView and write results to a caller-supplied
Float32 buffer of length (view.length - window + 1).

Uses popcount on packed UInt64 words for fast GC counting without decoding.
"""
from std.math import log2, ceildiv, min, max
from genomics.core.dna import BASES_PER_WORD, Base, get_base_from_word
from genomics.core.sequence import SequenceView


# ===----------------------------------------------------------------------=== #
# GC content
# ===----------------------------------------------------------------------=== #

@always_inline
def _count_gc_in_word(word: UInt64) -> Int:
    """Count G and C bases (2-bit values 0b01 and 0b10) in a packed word.

    A base is G or C when bit0 XOR bit1 = 1, i.e. exactly one of the two
    encoding bits is set.  We use the identity:
        gc_mask = (w ^ (w >> 1)) & 0x5555...  (odd bits = '1' iff bits differ)
    Then popcount gives the number of G/C bases.
    """
    var gc_bits = (word ^ (word >> 1)) & UInt64(0x5555555555555555)
    return Int(gc_bits.popcount())


def gc_content_sliding[window: Int](
    view: SequenceView,
    result: UnsafePointer[Float32, MutAnyOrigin],
):
    """Compute per-position GC fraction in a sliding window of size `window`.

    out[i] = fraction of G+C bases in view[i .. i+window).
    Output length = view.length - window + 1.

    Uses a prefix-sum over packed-word GC counts for O(L) total work.
    """
    comptime assert window >= 1, "window must be >= 1"
    var n_out = view.length - window + 1
    if n_out <= 0:
        return

    # Build prefix sum of GC counts per base position.
    # prefix[i] = total G+C bases in view[0..i).
    var n_words = view.word_count
    var prefix = List[Int](capacity=view.length + 1)
    prefix.append(0)

    for w in range(n_words):
        var word = view.packed[w]
        var base_start = w * BASES_PER_WORD
        var n_bases = min(BASES_PER_WORD, view.length - base_start)
        var running = prefix[base_start]
        for i in range(n_bases):
            var b = get_base_from_word(word, i)
            var is_gc = (b == Base.C or b == Base.G)
            prefix.append(running + Int(is_gc))
            running += Int(is_gc)

    # Slide the window using the prefix array
    var inv_window = Float32(1.0) / Float32(window)
    for i in range(n_out):
        var gc = prefix[i + window] - prefix[i]
        result[i] = Float32(gc) * inv_window


# ===----------------------------------------------------------------------=== #
# Sequence entropy (Shannon, 4-symbol alphabet)
# ===----------------------------------------------------------------------=== #

def sequence_entropy_sliding[window: Int](
    view: SequenceView,
    result: UnsafePointer[Float32, MutAnyOrigin],
):
    """Compute Shannon entropy (bits) of the base distribution in each window.

    H = -sum_b p(b) * log2(p(b))  over {A, C, G, T}
    Maximum is log2(4) = 2.0 bits for a uniform distribution.
    N bases are excluded from the frequency count (window effectively shrinks).

    out[i] = entropy of view[i .. i+window).
    Output length = view.length - window + 1.
    """
    comptime assert window >= 1, "window must be >= 1"
    var n_out = view.length - window + 1
    if n_out <= 0:
        return

    # Initial frequency counts for the first window
    var freq = [0, 0, 0, 0]  # A, C, G, T
    var valid_bases = 0
    for i in range(window):
        if not view.is_n(i):
            var b = Int(view.get_base(i))
            freq[b] += 1
            valid_bases += 1

    @always_inline
    def compute_entropy(f: List[Int], total: Int) -> Float32:
        if total == 0:
            return 0.0
        var h: Float32 = 0.0
        var inv_total = Float32(1.0) / Float32(total)
        for b in range(4):
            if f[b] > 0:
                var p = Float32(f[b]) * inv_total
                h -= p * log2(p)
        return h

    result[0] = compute_entropy(freq, valid_bases)

    # Slide: add incoming base, remove outgoing base
    for i in range(1, n_out):
        # Remove the base falling out of the window
        var out_pos = i - 1
        if not view.is_n(out_pos):
            var b = Int(view.get_base(out_pos))
            freq[b] -= 1
            valid_bases -= 1

        # Add the new base entering the window
        var in_pos = i + window - 1
        if not view.is_n(in_pos):
            var b = Int(view.get_base(in_pos))
            freq[b] += 1
            valid_bases += 1

        result[i] = compute_entropy(freq, valid_bases)


# ===----------------------------------------------------------------------=== #
# Linguistic complexity (k-mer diversity score)
# ===----------------------------------------------------------------------=== #

def complexity_score[k: Int = 2](view: SequenceView) -> Float32:
    """Compute linguistic complexity as the fraction of observed distinct k-mers.

    complexity = distinct_kmers_observed / min(n_possible, L - k + 1)
    where n_possible = 4^k and L = sequence length.

    A value near 1.0 indicates high sequence complexity; near 0 suggests
    low-complexity or repetitive regions.
    """
    comptime assert k >= 1 and k <= 10, "k must be in [1, 10] for complexity score"
    var n_possible = 1
    comptime for _ in range(k):
        n_possible *= 4

    var n_kmers = view.length - k + 1
    if n_kmers <= 0:
        return 0.0

    # Track observed k-mers using a small bit-set (n_possible <= 4^10 = 1M)
    var seen = List[Bool](capacity=n_possible)
    for _ in range(n_possible):
        seen.append(False)

    var kmer_val: Int = 0
    var mask = n_possible - 1

    # Seed initial k-mer
    for i in range(k):
        var b = Int(view.get_base(i))
        kmer_val = ((kmer_val << 2) | b) & mask

    var distinct = 0
    for pos in range(n_kmers):
        if pos > 0:
            var b = Int(view.get_base(pos + k - 1))
            kmer_val = ((kmer_val << 2) | b) & mask

        if not view.has_n_in_window(pos, pos + k):
            if not seen[kmer_val]:
                seen[kmer_val] = True
                distinct += 1

    var max_observable = min(n_possible, n_kmers)
    return Float32(distinct) / Float32(max_observable)
