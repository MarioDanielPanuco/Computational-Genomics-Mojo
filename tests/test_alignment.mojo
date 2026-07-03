"""
Unit tests for CPU banded alignment (Smith-Waterman, Needleman-Wunsch) and WFA.

Includes a hand-verified test case used as a regression anchor for GPU kernels.
"""
from std.testing import assert_equal, assert_true, TestSuite
from genomics.core.sequence import SequenceBatch, get_view
from genomics.cpu.align_cpu import (
    smith_waterman_banded, needleman_wunsch_banded, default_config,
    wfa_affine_cpu, default_wfa_config,
    wfa_affine_cigar_cpu, WFAResult,
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


# ===----------------------------------------------------------------------=== #
# WFA CIGAR traceback tests
# ===----------------------------------------------------------------------=== #

def _cigar_score(cigar: String, x: Int, o: Int, e: Int) -> Int:
    """Recompute alignment cost from a CIGAR string.

    Iterates over codepoints (ASCII CIGAR is safe): digits build the run
    count, op characters dispatch to the cost formula.
    Op ASCII: '='=61, 'X'=88, 'I'=73, 'D'=68.
    """
    var cost = 0
    var cnt = 0
    for cp in cigar.codepoints():
        var ch = Int(cp)
        if ch >= 48 and ch <= 57:        # digit '0'..'9'
            cnt = cnt * 10 + ch - 48
        elif ch == 88:                    # 'X' mismatch
            cost += cnt * x
            cnt = 0
        elif ch == 73 or ch == 68:       # 'I' or 'D' gap
            cost += o + cnt * e
            cnt = 0
        else:                             # '=' match — free
            cnt = 0
    return cost


def _cigar_query_len(cigar: String) -> Int:
    """Count query bases consumed (= + X + I ops)."""
    var total = 0
    var cnt = 0
    for cp in cigar.codepoints():
        var ch = Int(cp)
        if ch >= 48 and ch <= 57:
            cnt = cnt * 10 + ch - 48
        elif ch == 61 or ch == 88 or ch == 73:  # = X I
            total += cnt
            cnt = 0
        else:
            cnt = 0
    return total


def _cigar_ref_len(cigar: String) -> Int:
    """Count reference bases consumed (= + X + D ops)."""
    var total = 0
    var cnt = 0
    for cp in cigar.codepoints():
        var ch = Int(cp)
        if ch >= 48 and ch <= 57:
            cnt = cnt * 10 + ch - 48
        elif ch == 61 or ch == 88 or ch == 68:  # = X D
            total += cnt
            cnt = 0
        else:
            cnt = 0
    return total


def test_wfa_cigar_identical() raises:
    """Identical sequences → score 0 and all-match CIGAR."""
    var cfg = default_wfa_config()
    var q = make_batch("ACGTACGT")
    var r = make_batch("ACGTACGT")
    var res = wfa_affine_cigar_cpu(get_view(q, 0), get_view(r, 0), cfg)
    assert_equal(res.score, 0)
    assert_equal(res.cigar, String("8="))


def _check_cigar_pair(q_str: String, r_str: String) raises:
    """Assert CIGAR score == wfa_affine_cpu and CIGAR lengths match sequences."""
    var cfg = default_wfa_config()
    var q = make_batch(q_str)
    var r = make_batch(r_str)
    var expected = wfa_affine_cpu(get_view(q, 0), get_view(r, 0), cfg)
    var res      = wfa_affine_cigar_cpu(get_view(q, 0), get_view(r, 0), cfg)
    assert_equal(res.score, expected)
    assert_equal(_cigar_score(res.cigar, cfg.x, cfg.o, cfg.e), expected)
    assert_equal(_cigar_query_len(res.cigar), len(q_str))
    assert_equal(_cigar_ref_len(res.cigar),   len(r_str))


def test_wfa_cigar_score_matches_wfa() raises:
    """CIGAR score must equal wfa_affine_cpu for several sequence pairs."""
    _check_cigar_pair("ACGT", "ACAT")       # 1 mismatch
    _check_cigar_pair("ACGGT", "ACGT")      # 1 insertion
    _check_cigar_pair("ACGT", "ACGGT")      # 1 deletion
    _check_cigar_pair("ACGT", "ACGGGT")     # 2-base deletion
    _check_cigar_pair("AAAA", "TTTT")       # all mismatches


def test_wfa_cigar_lengths() raises:
    """CIGAR query/ref lengths must match the actual sequence lengths."""
    _check_cigar_pair("ACGT", "ACAT")
    _check_cigar_pair("ACGGT", "ACGT")
    _check_cigar_pair("ACGT", "ACGGT")
    _check_cigar_pair("AAAA", "TTTT")
    _check_cigar_pair("ACGTACGT", "ACGTACGT")


def test_wfa_cigar_single_mismatch() raises:
    """ACGT vs ACAT: one mismatch at position 2 → '2=1X1='."""
    var cfg = default_wfa_config()
    var q = make_batch("ACGT")
    var r = make_batch("ACAT")
    var res = wfa_affine_cigar_cpu(get_view(q, 0), get_view(r, 0), cfg)
    assert_equal(res.score, cfg.x)
    assert_equal(res.cigar, String("2=1X1="))


def test_wfa_cigar_single_insertion() raises:
    """ACGGT vs ACGT: one insertion → '3=1I1='."""
    var cfg = default_wfa_config()
    var q = make_batch("ACGGT")
    var r = make_batch("ACGT")
    var res = wfa_affine_cigar_cpu(get_view(q, 0), get_view(r, 0), cfg)
    assert_equal(res.score, cfg.o + cfg.e)
    assert_equal(res.cigar, String("3=1I1="))


def test_wfa_cigar_single_deletion() raises:
    """ACGT vs ACGGT: one deletion → '3=1D1='."""
    var cfg = default_wfa_config()
    var q = make_batch("ACGT")
    var r = make_batch("ACGGT")
    var res = wfa_affine_cigar_cpu(get_view(q, 0), get_view(r, 0), cfg)
    assert_equal(res.score, cfg.o + cfg.e)
    assert_equal(res.cigar, String("3=1D1="))


def test_wfa_cigar_gap_extend() raises:
    """ACGT vs ACGGGT: 2-base deletion → '3=2D1='."""
    var cfg = default_wfa_config()
    var q = make_batch("ACGT")
    var r = make_batch("ACGGGT")
    var res = wfa_affine_cigar_cpu(get_view(q, 0), get_view(r, 0), cfg)
    assert_equal(res.score, cfg.o + 2 * cfg.e)
    assert_equal(res.cigar, String("3=2D1="))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
