"""
Unit tests for (w,k) minimizer extraction.

Property tests verify algorithmic invariants:
  1. Every minimizer hash is the minimum hash in at least one w-length window.
  2. No position is emitted twice (deduplication).
  3. Every w-length window contains at least one reported minimizer position.
  4. w=1 produces one minimizer per k-mer (no deduplication possible).

Hand-crafted tests verify specific counts and positions against manually
computed expected values.
"""
from std.testing import assert_equal, assert_true, TestSuite
from genomics.core.sequence import SequenceBatch, get_view
from genomics.cpu.minimizer import extract_minimizers, MinimizerSketch


def make_batch(seq: String) -> SequenceBatch:
    var batch = SequenceBatch()
    var buf = List[UInt8](capacity=len(seq))
    for cp in seq.codepoints():
        buf.append(UInt8(Int(cp)))
    batch.add_sequence(Span(buf), len(seq))
    return batch^


# ===----------------------------------------------------------------------=== #
# Helper: collect all k-mer hashes from a view using the same Kmer logic
# ===----------------------------------------------------------------------=== #

def _all_kmer_hashes[k: Int](seq: String) -> List[UInt64]:
    """Return canonical Kmer[k] hashes for every position in seq."""
    from genomics.core.kmer import Kmer
    var batch = make_batch(seq)
    var view = get_view(batch, 0)
    var n = view.length - k + 1
    var result = List[UInt64]()
    if n <= 0:
        return result^
    var kmer = Kmer[k]()
    for i in range(k):
        kmer = kmer.roll(view.get_base(i))
    for pos in range(n):
        if pos > 0:
            kmer = kmer.roll(view.get_base(pos + k - 1))
        result.append(kmer.canonical().hash())
    return result^


# ===----------------------------------------------------------------------=== #
# Property tests
# ===----------------------------------------------------------------------=== #

def test_minimizer_hash_is_window_minimum() raises:
    """Every reported hash must be the minimum in at least one w-length window."""
    var seq = "ACGTACGTACGTACGT"
    var batch = make_batch(seq)
    var view = get_view(batch, 0)
    comptime K = 4
    comptime W = 5
    var sketch = extract_minimizers[K](view, W)
    var hashes = _all_kmer_hashes[K](seq)
    var n_kmers = len(hashes)

    for si in range(sketch.size()):
        var reported_hash = sketch.hashes[si]
        var reported_pos  = sketch.positions[si]
        # Find any window containing this position where it is the minimum
        var is_valid = False
        var w_start_lo = max(0, reported_pos - W + 1)
        var w_start_hi = min(reported_pos, n_kmers - W)
        for ws in range(w_start_lo, w_start_hi + 1):
            var is_min = True
            for wi in range(W):
                if hashes[ws + wi] < reported_hash:
                    is_min = False
                    break
            if is_min:
                is_valid = True
                break
        assert_true(is_valid)


def test_minimizer_no_duplicate_positions() raises:
    """No position should appear more than once in the sketch."""
    var seq = "ACGTACGTACGTACGTACGT"
    var batch = make_batch(seq)
    var view = get_view(batch, 0)
    comptime K = 4
    comptime W = 4
    var sketch = extract_minimizers[K](view, W)

    for i in range(sketch.size()):
        for j in range(i + 1, sketch.size()):
            assert_true(sketch.positions[i] != sketch.positions[j])


def test_minimizer_every_window_covered() raises:
    """Every w-kmer window must contain at least one reported minimizer position."""
    var seq = "ACGTACGTACGTACGTACGT"
    var batch = make_batch(seq)
    var view = get_view(batch, 0)
    comptime K = 4
    comptime W = 5
    var sketch = extract_minimizers[K](view, W)
    var n_kmers = view.length - K + 1
    var n_windows = n_kmers - W + 1

    for ws in range(n_windows):
        var covered = False
        for si in range(sketch.size()):
            var p = sketch.positions[si]
            if p >= ws and p < ws + W:
                covered = True
                break
        assert_true(covered)


def test_minimizer_w1_is_all_kmers() raises:
    """w=1 means each k-mer is its own window; sketch size equals n_kmers."""
    var seq = "ACGTACGT"
    var batch = make_batch(seq)
    var view = get_view(batch, 0)
    comptime K = 3
    comptime W = 1
    var sketch = extract_minimizers[K](view, W)
    var n_kmers = view.length - K + 1
    assert_equal(sketch.size(), n_kmers)


# ===----------------------------------------------------------------------=== #
# Hand-crafted tests with known expected values
# ===----------------------------------------------------------------------=== #

def test_minimizer_empty_result_short_seq() raises:
    """Sequence shorter than k+w-1 produces an empty sketch."""
    # k=4, w=3 → need length >= 6; "ACGT" (len=4) → 0 minimizers
    var batch = make_batch("ACGT")
    var view = get_view(batch, 0)
    var sketch = extract_minimizers[4](view, 3)
    assert_equal(sketch.size(), 0)


def test_minimizer_identical_sequence() raises:
    """A run of identical bases: every k-mer is the same, so one minimizer covers all."""
    # "AAAAAAAAAA" with k=3, w=3: all k-mers are "AAA" → same hash throughout.
    # Because hashes are equal, the deque never evicts the front entry (>= h_i
    # condition includes equality), so new entries push the leftmost out only
    # via the window-expiry pop-front.  Each window keeps its leftmost k-mer
    # until the window slides past it, producing one minimizer per window step
    # that changes the leftmost position — but since all hashes are equal the
    # deque collapses to the newest element after the '>= h_i' pop-back.
    # The exact count depends on tie-breaking; what must hold is coverage.
    var seq = "AAAAAAAAAA"
    var batch = make_batch(seq)
    var view = get_view(batch, 0)
    comptime K = 3
    comptime W = 3
    var sketch = extract_minimizers[K](view, W)
    # Coverage: every window must be covered
    var n_kmers = view.length - K + 1
    var n_windows = n_kmers - W + 1
    for ws in range(n_windows):
        var covered = False
        for si in range(sketch.size()):
            var p = sketch.positions[si]
            if p >= ws and p < ws + W:
                covered = True
                break
        assert_true(covered)


def test_minimizer_single_window() raises:
    """Exactly one window → exactly one minimizer."""
    # k=4, w=2 → need L = k+w-1 = 5 bases → one window of 2 k-mers
    var batch = make_batch("ACGTA")
    var view = get_view(batch, 0)
    var sketch = extract_minimizers[4](view, 2)
    assert_equal(sketch.size(), 1)


def test_minimizer_positions_in_range() raises:
    """All minimizer positions must lie in [0, length - k]."""
    var seq = "ACGTACGTACGTACGT"
    var batch = make_batch(seq)
    var view = get_view(batch, 0)
    comptime K = 4
    comptime W = 4
    var sketch = extract_minimizers[K](view, W)
    var max_pos = view.length - K
    for si in range(sketch.size()):
        assert_true(sketch.positions[si] >= 0)
        assert_true(sketch.positions[si] <= max_pos)


def test_minimizer_k_equals_w() raises:
    """k == w is a valid configuration (common in practice for small sketches)."""
    var seq = "ACGTACGTACGT"
    var batch = make_batch(seq)
    var view = get_view(batch, 0)
    comptime K = 4
    comptime W = 4  # same as k
    var sketch = extract_minimizers[K](view, W)
    assert_true(sketch.size() > 0)
    # Coverage check
    var n_kmers = view.length - K + 1
    var n_windows = n_kmers - W + 1
    for ws in range(n_windows):
        var covered = False
        for si in range(sketch.size()):
            var p = sketch.positions[si]
            if p >= ws and p < ws + W:
                covered = True
                break
        assert_true(covered)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
