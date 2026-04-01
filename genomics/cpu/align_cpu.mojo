"""
CPU banded Smith-Waterman and Needleman-Wunsch alignment.

Serves as a reference baseline for validating GPU alignment results.

Band: only cells within `band_width` diagonals of the main diagonal are
computed. Memory: 3 rows of width (2*band+1) in a ring buffer → O(band) space.
"""
from std.math import min, max
from genomics.core.sequence import SequenceView


@fieldwise_init
struct AlignConfig(Copyable, Movable):
    """Scoring parameters for pairwise alignment."""
    var match_score: Int
    var mismatch: Int     # penalty (negative value expected)
    var gap_open: Int     # affine gap open penalty (negative)
    var gap_extend: Int   # affine gap extension penalty (negative)
    var band_width: Int


@fieldwise_init
struct AlignResult(Movable):
    """Result of a pairwise alignment."""
    var score: Int
    var query_start: Int
    var ref_start: Int
    var query_end: Int
    var ref_end: Int
    var cigar: List[UInt8]   # run-length encoded: high 4 bits = op, low 4 bits = len


def default_config() -> AlignConfig:
    return AlignConfig(
        match_score=2,
        mismatch=-3,
        gap_open=-5,
        gap_extend=-2,
        band_width=32,
    )


# CIGAR operation codes
comptime CIGAR_MATCH: UInt8 = 0   # M
comptime CIGAR_INS: UInt8 = 1     # I (insertion in query)
comptime CIGAR_DEL: UInt8 = 2     # D (deletion from query)


def smith_waterman_banded(
    query: SequenceView,
    reference: SequenceView,
    cfg: AlignConfig,
) -> AlignResult:
    """Banded Smith-Waterman local alignment.

    Restricts computation to the diagonal band [main_diag - band, main_diag + band].
    Returns the highest-scoring local alignment found within the band.
    """
    var band = cfg.band_width
    var qlen = query.length
    var rlen = reference.length
    var width = 2 * band + 1

    # DP matrices: H (best), E (gap in query/deletion), F (gap in ref/insertion)
    # Using flat 2-row ring buffer: current and previous row.
    var H_prev = List[Int](capacity=width)
    var H_curr = List[Int](capacity=width)
    var E_curr = List[Int](capacity=width)

    for _ in range(width):
        H_prev.append(0)
        H_curr.append(0)
        E_curr.append(0)

    var best_score = 0
    var best_qi = 0
    var best_ri = 0

    for qi in range(qlen):
        # Reset current row
        for j in range(width):
            H_curr[j] = 0
            E_curr[j] = 0

        for ri_offset in range(width):
            var ri = qi - band + ri_offset
            if ri < 0 or ri >= rlen:
                continue

            # Match/mismatch score
            var same = query.get_base(qi) == reference.get_base(ri)
            var sub = cfg.match_score if same else cfg.mismatch

            # Predecessor diagonal index in prev row
            var diag_prev = ri_offset  # maps to (qi-1, ri-1) in prev row coordinates

            var h_diag = H_prev[diag_prev] if diag_prev >= 0 and diag_prev < width else 0
            var h_up = H_prev[ri_offset + 1] if ri_offset + 1 < width else 0
            var h_left = H_curr[ri_offset - 1] if ri_offset > 0 else 0

            # Affine gap: E = max(E_prev + gap_extend, H_up + gap_open + gap_extend)
            var e_val = max(E_curr[ri_offset - 1] + cfg.gap_extend if ri_offset > 0 else -1000000,
                            h_up + cfg.gap_open + cfg.gap_extend)

            # F = max(F_left + gap_extend, H_left + gap_open + gap_extend)
            # (simplified: not storing F separately — use H_left)
            var f_val = h_left + cfg.gap_open + cfg.gap_extend

            var cell = max(0, max(h_diag + sub, max(e_val, f_val)))
            H_curr[ri_offset] = cell
            E_curr[ri_offset] = e_val

            if cell > best_score:
                best_score = cell
                best_qi = qi
                best_ri = ri

        # Swap rows
        for j in range(width):
            H_prev[j] = H_curr[j]

    return AlignResult(
        score=best_score,
        query_start=0,
        ref_start=0,
        query_end=best_qi,
        ref_end=best_ri,
        cigar=List[UInt8](),  # traceback not implemented in this baseline
    )


def needleman_wunsch_banded(
    query: SequenceView,
    reference: SequenceView,
    cfg: AlignConfig,
) -> AlignResult:
    """Banded Needleman-Wunsch global alignment.

    Returns the global alignment score within the diagonal band.
    """
    var band = cfg.band_width
    var qlen = query.length
    var rlen = reference.length
    var width = 2 * band + 1
    var NEG_INF = -1000000

    var H_prev = List[Int](capacity=width)
    var H_curr = List[Int](capacity=width)

    for j in range(width):
        var ri = -band + j
        H_prev[j] = max(NEG_INF, cfg.gap_open + ri * cfg.gap_extend) if ri >= 0 else NEG_INF

    for qi in range(qlen):
        for j in range(width):
            H_curr[j] = NEG_INF

        for ri_offset in range(width):
            var ri = qi - band + ri_offset
            if ri < 0 or ri >= rlen:
                continue

            var same = query.get_base(qi) == reference.get_base(ri)
            var sub = cfg.match_score if same else cfg.mismatch

            var diag_prev = ri_offset
            var h_diag = H_prev[diag_prev] if diag_prev >= 0 and diag_prev < width else NEG_INF
            var h_up = H_prev[ri_offset + 1] if ri_offset + 1 < width else NEG_INF
            var h_left = H_curr[ri_offset - 1] if ri_offset > 0 else NEG_INF

            var cell = max(h_diag + sub,
                           max(h_up + cfg.gap_open + cfg.gap_extend,
                               h_left + cfg.gap_open + cfg.gap_extend))
            H_curr[ri_offset] = cell

        for j in range(width):
            H_prev[j] = H_curr[j]

    # Score is at position where qi = qlen-1, ri = rlen-1
    var final_offset = (rlen - 1) - (qlen - 1) + band
    var final_score = NEG_INF
    if final_offset >= 0 and final_offset < width:
        final_score = H_prev[final_offset]

    return AlignResult(
        score=final_score,
        query_start=0,
        ref_start=0,
        query_end=qlen - 1,
        ref_end=rlen - 1,
        cigar=List[UInt8](),
    )
