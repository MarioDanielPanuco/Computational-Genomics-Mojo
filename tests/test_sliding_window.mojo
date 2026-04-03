"""
Unit tests for CPU sliding-window transforms: GC content, entropy, complexity.
"""
from std.testing import assert_equal, assert_true, TestSuite
from std.math import abs
from genomics.core.sequence import SequenceBatch, get_view
from genomics.cpu.sliding_window import (
    gc_content_sliding, sequence_entropy_sliding, complexity_score,
)


def make_batch(seq: String) -> SequenceBatch:
    var batch = SequenceBatch()
    var buf = List[UInt8](capacity=len(seq))
    for cp in seq.codepoints():
        buf.append(UInt8(Int(cp)))
    batch.add_sequence(Span(buf), len(seq))
    return batch^


def _alloc_out(n: Int) -> List[Float32]:
    var buf = List[Float32](capacity=n)
    for _ in range(n):
        buf.append(0.0)
    return buf^


# ===----------------------------------------------------------------------=== #
# GC content
# ===----------------------------------------------------------------------=== #

def test_gc_all_at() raises:
    """All A/T sequence → GC fraction = 0.0 everywhere."""
    var batch = make_batch("ATATAT")
    var view = get_view(batch, 0)
    comptime w = 4
    var buf = _alloc_out(view.length - w + 1)
    gc_content_sliding[w](view, buf.unsafe_ptr())
    for i in range(len(buf)):
        assert_equal(buf[i], Float32(0.0))


def test_gc_all_gc() raises:
    """All G/C sequence → GC fraction = 1.0 everywhere."""
    var batch = make_batch("GCGCGC")
    var view = get_view(batch, 0)
    comptime w = 4
    var buf = _alloc_out(view.length - w + 1)
    gc_content_sliding[w](view, buf.unsafe_ptr())
    for i in range(len(buf)):
        assert_equal(buf[i], Float32(1.0))


def test_gc_half() raises:
    """ACGT window-4 → GC = 0.5 (C and G are 2 of 4 bases)."""
    var batch = make_batch("ACGT")
    var view = get_view(batch, 0)
    comptime w = 4
    var buf = _alloc_out(1)
    gc_content_sliding[w](view, buf.unsafe_ptr())
    assert_equal(buf[0], Float32(0.5))


def test_gc_sliding_shift() raises:
    """AAACCC window-3: first window=0.0, last window=1.0."""
    var batch = make_batch("AAACCC")
    var view = get_view(batch, 0)
    comptime w = 3
    var buf = _alloc_out(view.length - w + 1)  # 4 positions
    gc_content_sliding[w](view, buf.unsafe_ptr())
    assert_equal(buf[0], Float32(0.0))  # AAA
    assert_equal(buf[3], Float32(1.0))  # CCC


def test_gc_mixed_window() raises:
    """AACCGG window-2: values match per-pair GC fraction."""
    var batch = make_batch("AACCGG")
    var view = get_view(batch, 0)
    comptime w = 2
    var buf = _alloc_out(5)
    gc_content_sliding[w](view, buf.unsafe_ptr())
    assert_equal(buf[0], Float32(0.0))   # AA
    assert_equal(buf[1], Float32(0.5))   # AC
    assert_equal(buf[2], Float32(1.0))   # CC
    assert_equal(buf[3], Float32(1.0))   # CG
    assert_equal(buf[4], Float32(1.0))   # GG


# ===----------------------------------------------------------------------=== #
# Sequence entropy
# ===----------------------------------------------------------------------=== #

def test_entropy_constant_base() raises:
    """Single-symbol window → entropy = 0.0."""
    var batch = make_batch("AAAA")
    var view = get_view(batch, 0)
    comptime w = 4
    var buf = _alloc_out(1)
    sequence_entropy_sliding[w](view, buf.unsafe_ptr())
    assert_equal(buf[0], Float32(0.0))


def test_entropy_uniform() raises:
    """ACGT window-4 → max entropy ≈ 2.0 bits."""
    var batch = make_batch("ACGT")
    var view = get_view(batch, 0)
    comptime w = 4
    var buf = _alloc_out(1)
    sequence_entropy_sliding[w](view, buf.unsafe_ptr())
    assert_true(abs(buf[0] - Float32(2.0)) < Float32(0.001))


def test_entropy_two_symbols() raises:
    """ATAT window-4 → entropy = 1.0 bit (50/50 A/T)."""
    var batch = make_batch("ATAT")
    var view = get_view(batch, 0)
    comptime w = 4
    var buf = _alloc_out(1)
    sequence_entropy_sliding[w](view, buf.unsafe_ptr())
    assert_true(abs(buf[0] - Float32(1.0)) < Float32(0.001))


def test_entropy_increases_with_diversity() raises:
    """AAAACGT window-4: entropy at first window < entropy at last window."""
    var batch = make_batch("AAAACGT")
    var view = get_view(batch, 0)
    comptime w = 4
    var buf = _alloc_out(view.length - w + 1)  # 4 positions
    sequence_entropy_sliding[w](view, buf.unsafe_ptr())
    # First window AAAA → 0.0, last window ACGT → 2.0
    assert_equal(buf[0], Float32(0.0))
    assert_true(abs(buf[3] - Float32(2.0)) < Float32(0.001))


# ===----------------------------------------------------------------------=== #
# Linguistic complexity
# ===----------------------------------------------------------------------=== #

def test_complexity_single_base() raises:
    """All-same base → one distinct k-mer out of many possible → low complexity."""
    var batch = make_batch("AAAAAA")
    var view = get_view(batch, 0)
    # k=1: 1 distinct out of min(4, 6) = 4 → score = 0.25
    var score = complexity_score[1](view)
    assert_true(score < Float32(0.5))


def test_complexity_all_four_bases() raises:
    """ACGT: all 4 1-mers observed → complexity = 1.0."""
    var batch = make_batch("ACGT")
    var view = get_view(batch, 0)
    # k=1: 4 distinct out of min(4, 4) = 4 → score = 1.0
    var score = complexity_score[1](view)
    assert_equal(score, Float32(1.0))


def test_complexity_increases_with_diversity() raises:
    """More diverse sequence has higher complexity than repetitive one."""
    var low_batch = make_batch("AAAAAAAA")
    var high_batch = make_batch("ACGTACGT")
    var low_view = get_view(low_batch, 0)
    var high_view = get_view(high_batch, 0)
    var low_score = complexity_score[2](low_view)
    var high_score = complexity_score[2](high_view)
    assert_true(high_score > low_score)


def test_complexity_short_sequence() raises:
    """Sequence shorter than k → complexity = 0.0."""
    var batch = make_batch("AC")
    var view = get_view(batch, 0)
    var score = complexity_score[4](view)
    assert_equal(score, Float32(0.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
