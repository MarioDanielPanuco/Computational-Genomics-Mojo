"""
Unit tests for CPU banded alignment (Smith-Waterman and Needleman-Wunsch).

Includes a hand-verified test case used as a regression anchor for GPU kernels.
"""
from std.testing import assert_equal, assert_true, TestSuite
from genomics.core.sequence import SequenceBatch, get_view
from genomics.cpu.align_cpu import (
    smith_waterman_banded, needleman_wunsch_banded, default_config,
)


def make_batch(seq: String) -> SequenceBatch:
    var batch = SequenceBatch()
    var buf = List[UInt8](capacity=len(seq))
    for cp in seq.codepoints():
        buf.append(UInt8(Int(cp)))
    batch.add_sequence(Span(buf), len(seq))
    return batch^


def test_sw_identical_sequences() raises:
    """Two identical sequences yield maximum score: len * match_score."""
    var cfg = default_config()
    cfg.band_width = 8

    var q_batch = make_batch("ACGTACGT")
    var r_batch = make_batch("ACGTACGT")
    var result = smith_waterman_banded(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(result.score, 8 * cfg.match_score)


def test_sw_single_mismatch() raises:
    """ACGTACGT vs ACGAACGT — one mismatch; local best should be >= 8 (trailing ACGT match).

    Hand-verified regression anchor for GPU kernel validation.
    """
    var cfg = default_config()
    cfg.band_width = 4

    var q_batch = make_batch("ACGTACGT")
    var r_batch = make_batch("ACGAACGT")
    var result = smith_waterman_banded(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_true(result.score > 0)
    assert_true(result.score >= 8)


def test_sw_no_match() raises:
    """AAAA vs TTTT — no matching bases; SW score must be 0."""
    var cfg = default_config()
    cfg.band_width = 4

    var q_batch = make_batch("AAAA")
    var r_batch = make_batch("TTTT")
    var result = smith_waterman_banded(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(result.score, 0)


def test_nw_identical() raises:
    """NW global alignment of identical sequences: score = len * match."""
    var cfg = default_config()
    cfg.band_width = 4

    var q_batch = make_batch("ACGT")
    var r_batch = make_batch("ACGT")
    var result = needleman_wunsch_banded(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(result.score, 4 * cfg.match_score)


def test_nw_all_mismatch() raises:
    """NW global alignment of completely different seqs: 4 mismatches = 4 * (-3) = -12."""
    var cfg = default_config()
    cfg.band_width = 4

    var q_batch = make_batch("AAAA")
    var r_batch = make_batch("TTTT")
    var result = needleman_wunsch_banded(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(result.score, 4 * cfg.mismatch)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
