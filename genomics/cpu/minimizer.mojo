"""
(w, k) minimizer sketching: sliding-window minimum over canonical k-mer hashes.

For a sequence of length L, windows of w consecutive k-mers are scanned
left-to-right.  The k-mer with the minimum canonical hash in each window is the
window's minimizer.  When consecutive windows share the same minimizer, it is
recorded only once (deduplication), keeping the sketch compact.

Sliding-window minimum is computed in O(L) time using a monotonic deque,
exactly as in minimap2 / Kraken2.

References:
  Roberts et al. (2004) Reducing storage requirements for biological sequence
    comparison.  Bioinformatics 20(18):3363–3369.
  Schleimer et al. (2003) Winnowing: local algorithms for document fingerprinting.
    ACM SIGMOD, pp. 76–85.
"""
from genomics.core.sequence import SequenceView
from genomics.core.kmer import Kmer


struct MinimizerSketch(Movable):
    """Deduplicated (w,k) minimizer set extracted from a single sequence.

    hashes[i] and positions[i] describe the i-th distinct minimizer:
      - hashes[i]    : canonical k-mer hash of the minimizer
      - positions[i] : 0-based start position of that k-mer in the sequence

    Density guarantee: at least one minimizer per w consecutive k-mers.
    """
    var hashes: List[UInt64]
    var positions: List[Int]
    var k: Int
    var w: Int

    def __init__(out self, k: Int, w: Int):
        self.hashes = List[UInt64]()
        self.positions = List[Int]()
        self.k = k
        self.w = w

    def size(self) -> Int:
        """Number of distinct minimizers in the sketch."""
        return len(self.hashes)


def extract_minimizers[k: Int](
    view: SequenceView,
    w: Int,
    canonical: Bool = True,
) -> MinimizerSketch:
    """Extract (w,k) minimizers from a sequence view.

    Scans every window of w consecutive k-mers and picks the one with the
    smallest hash.  Each minimizer position is emitted at most once.

    N-containing k-mers receive hash = ~UInt64(0) (sentinel maximum) so they
    are never selected unless every k-mer in the window is N-masked.

    Args:
        view:      packed 2-bit sequence to sketch
        w:         window width in k-mers (typically 5–20; w=1 keeps every k-mer)
        canonical: if True, use canonical (strand-symmetric) k-mer hashes

    Returns:
        MinimizerSketch with deduplicated (hash, position) pairs.
    """
    var sketch = MinimizerSketch(k=k, w=w)
    var n_kmers = view.length - k + 1
    if n_kmers <= 0 or n_kmers < w:
        return sketch^

    # --- Phase 1: compute a canonical hash for every k-mer position ---
    var hash_buf = List[UInt64](capacity=n_kmers)

    var kmer = Kmer[k]()
    for i in range(k):
        kmer = kmer.roll(view.get_base(i))

    comptime SENTINEL: UInt64 = ~UInt64(0)  # marks N-masked positions

    for pos in range(n_kmers):
        if pos > 0:
            kmer = kmer.roll(view.get_base(pos + k - 1))

        if view.has_n_in_window(pos, pos + k):
            hash_buf.append(SENTINEL)
        else:
            var emit = kmer.canonical() if canonical else kmer
            hash_buf.append(emit.hash())

    # --- Phase 2: sliding-window minimum via monotonic deque ---
    # Deque stores k-mer *indices* with strictly increasing hash values
    # (ties broken by smaller index = leftmost position).
    # Front of the deque is always the index of the current window minimum.
    var dq_cap = w + 1          # max deque size = w
    var dq_buf = List[Int](capacity=dq_cap)
    for _ in range(dq_cap):
        dq_buf.append(0)
    var dq_lo = 0               # front pointer (inclusive)
    var dq_hi = 0               # back pointer  (exclusive)

    var last_min_pos = -1       # position of the minimizer emitted last

    for i in range(n_kmers):
        var h_i = hash_buf[i]

        # Maintain monotone-increasing invariant: drop back elements whose
        # hash is >= h_i (they can never be the minimum while i is in the window).
        while dq_lo < dq_hi:
            var back_idx = (dq_hi - 1) % dq_cap
            if hash_buf[dq_buf[back_idx]] >= h_i:
                dq_hi -= 1
            else:
                break
        dq_buf[dq_hi % dq_cap] = i
        dq_hi += 1

        if i >= w - 1:
            # Drop front elements that have slid out of window [i-w+1, i]
            while dq_buf[dq_lo % dq_cap] < i - w + 1:
                dq_lo += 1

            var min_pos = dq_buf[dq_lo % dq_cap]
            if min_pos != last_min_pos:
                sketch.hashes.append(hash_buf[min_pos])
                sketch.positions.append(min_pos)
                last_min_pos = min_pos

    return sketch^
