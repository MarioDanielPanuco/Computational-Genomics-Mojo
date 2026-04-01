"""
Unit tests for SequenceBatch construction, encoding, and SequenceView access.
"""
from std.testing import assert_equal, assert_true, assert_false, TestSuite
from genomics.core.dna import Base, BASES_PER_WORD
from genomics.core.sequence import SequenceBatch, SequenceView, get_view


def ascii_to_batch(seqs: List[String]) -> SequenceBatch:
    var batch = SequenceBatch(capacity=len(seqs))
    for i in range(len(seqs)):
        var s = seqs[i]
        var buf = List[UInt8](capacity=len(s))
        for cp in s.codepoints():
            buf.append(UInt8(Int(cp)))
        batch.add_sequence(Span(buf), len(s))
    return batch^


def test_batch_count() raises:
    var batch = ascii_to_batch(["ACGT", "TTTT", "AAAA"])
    assert_equal(batch.count, 3)


def test_batch_total_bases() raises:
    var batch = ascii_to_batch(["ACGT", "ACGTACGT"])
    assert_equal(batch.total_bases(), 12)


def test_sequence_view_bases() raises:
    var batch = ascii_to_batch(["ACGT"])
    var view = get_view(batch, 0)
    assert_equal(view.length, 4)
    assert_equal(view.get_base(0), Base.A)
    assert_equal(view.get_base(1), Base.C)
    assert_equal(view.get_base(2), Base.G)
    assert_equal(view.get_base(3), Base.T)


def test_sequence_view_longer() raises:
    # 33-base sequence (spans two packed words)
    var seq = "ACGTACGTACGTACGTACGTACGTACGTACGTA"  # 33 chars
    var batch = ascii_to_batch([seq])
    var view = get_view(batch, 0)
    assert_equal(view.length, 33)
    assert_equal(view.word_count, 2)
    assert_equal(view.get_base(32), Base.A)  # last base


def test_n_base_tracking() raises:
    # ACNT — N at position 2
    var batch = SequenceBatch()
    var buf: List[UInt8] = [65, 67, 78, 84]  # A C N T
    batch.add_sequence(Span(buf), 4)

    var view = get_view(batch, 0)
    assert_false(view.is_n(0))
    assert_false(view.is_n(1))
    assert_true(view.is_n(2))
    assert_false(view.is_n(3))


def test_n_window_detection() raises:
    # ACNTACGT — N at position 2
    var batch = SequenceBatch()
    var buf: List[UInt8] = [65, 67, 78, 84, 65, 67, 71, 84]  # ACNTACGT
    batch.add_sequence(Span(buf), 8)

    var view = get_view(batch, 0)
    assert_true(view.has_n_in_window(0, 4))   # [0,4) includes pos 2
    assert_true(view.has_n_in_window(1, 4))   # [1,4) includes pos 2
    assert_false(view.has_n_in_window(3, 8))  # [3,8) skips pos 2
    assert_false(view.has_n_in_window(4, 8))  # [4,8) no N


def test_multiple_sequences_independent() raises:
    var batch = ascii_to_batch(["AAAA", "CCCC", "GGGG", "TTTT"])
    for i in range(4):
        var view = get_view(batch, i)
        assert_equal(view.length, 4)
        var expected_base = UInt8(i)  # A=0, C=1, G=2, T=3
        for j in range(4):
            assert_equal(view.get_base(j), expected_base)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
