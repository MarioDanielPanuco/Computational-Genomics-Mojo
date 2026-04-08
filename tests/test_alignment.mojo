"""
Unit tests for CPU banded alignment (Smith-Waterman, Needleman-Wunsch) and WFA.

Includes a hand-verified test case used as a regression anchor for GPU kernels.
"""
from std.testing import assert_equal, assert_true, TestSuite
from genomics.core.sequence import SequenceBatch, get_view
from genomics.cpu.align_cpu import (
    smith_waterman_banded, needleman_wunsch_banded, default_config,
    wfa_affine_cpu, default_wfa_config,
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


# ===----------------------------------------------------------------------=== #
# WFA tests
# ===----------------------------------------------------------------------=== #

def test_wfa_identical_sequences() raises:
    """Identical sequences cost 0 — extend phase hits (qlen, rlen) immediately."""
    var cfg = default_wfa_config()
    var q_batch = make_batch("ACGTACGT")
    var r_batch = make_batch("ACGTACGT")
    var cost = wfa_affine_cpu(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(cost, 0)


def test_wfa_single_mismatch() raises:
    """One substitution costs x = 4."""
    var cfg = default_wfa_config()
    var q_batch = make_batch("ACGT")
    var r_batch = make_batch("ACAT")   # pos 2: G→A
    var cost = wfa_affine_cpu(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(cost, cfg.x)


def test_wfa_single_deletion() raises:
    """Query 'ACGT' aligned to ref 'ACGGT' — one deletion costs o + e = 8."""
    var cfg = default_wfa_config()
    var q_batch = make_batch("ACGT")
    var r_batch = make_batch("ACGGT")
    var cost = wfa_affine_cpu(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(cost, cfg.o + cfg.e)


def test_wfa_gap_extend() raises:
    """Query 'ACGT' vs ref 'ACGGGT' — two-base deletion costs o + 2*e = 10."""
    var cfg = default_wfa_config()
    var q_batch = make_batch("ACGT")
    var r_batch = make_batch("ACGGGT")
    var cost = wfa_affine_cpu(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(cost, cfg.o + 2 * cfg.e)


def test_wfa_single_insertion() raises:
    """Query 'ACGGT' vs ref 'ACGT' — one insertion costs o + e = 8."""
    var cfg = default_wfa_config()
    var q_batch = make_batch("ACGGT")
    var r_batch = make_batch("ACGT")
    var cost = wfa_affine_cpu(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(cost, cfg.o + cfg.e)


def test_wfa_all_mismatches() raises:
    """AAAA vs TTTT — 4 mismatches cost 4 * x = 16."""
    var cfg = default_wfa_config()
    var q_batch = make_batch("AAAA")
    var r_batch = make_batch("TTTT")
    var cost = wfa_affine_cpu(get_view(q_batch, 0), get_view(r_batch, 0), cfg)
    assert_equal(cost, 4 * cfg.x)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
