"""
CPU pairwise alignment: banded Smith-Waterman, Needleman-Wunsch, and gap-affine WFA.

Serves as a reference baseline for validating GPU alignment results.

Banded SW/NW: only cells within `band_width` diagonals of the main diagonal are
computed. Memory: 3 rows of width (2*band+1) in a ring buffer → O(band) space.

WFA (Wavefront Alignment): iterates by alignment cost instead of by matrix row.
Stores only the "furthest-reaching point" (FRP) per diagonal, giving O(s²) memory
where s is the optimal alignment cost. The extend phase skips matching bases in O(1)
per matched run, making WFA fast on similar sequences.
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


@fieldwise_init
struct WFAConfig(Copyable, Movable):
    """Penalty model for gap-affine WFA. All values are positive costs."""
    var x: Int          # mismatch cost
    var o: Int          # gap open cost
    var e: Int          # gap extend cost per base
    var max_error: Int  # bail out after this many edits (returns -1)


def default_wfa_config() -> WFAConfig:
    return WFAConfig(x=4, o=6, e=2, max_error=200)


def wfa_affine_cpu(
    query: SequenceView,
    ref_seq: SequenceView,
    cfg: WFAConfig,
) -> Int:
    """Gap-affine WFA global alignment. Returns minimum cost, or -1 if > max_error.

    Maintains three circular-buffer wavefront arrays:
      M[s,k] — furthest query row reachable on diagonal k at cost s ending in match/mismatch
      I[s,k] — same, ending in insertion (gap in reference; query row advances)
      D[s,k] — same, ending in deletion  (gap in query;   query row stays)

    Diagonal k = row - col. Target diagonal = qlen - rlen.
    Termination: M[s, target_k] >= qlen.

    Recurrences (Convention A — row indexing):
      I[s,k] = max(M[s-o-e, k-1]+1,  I[s-e, k-1]+1)
      D[s,k] = max(M[s-o-e, k+1],    D[s-e, k+1])
      M[s,k] = max(M[s-x, k]+1,      I[s,k],  D[s,k])

    Extend phase after each score: advance while query[row]==ref[col].
    """
    var qlen = query.length
    var rlen = ref_seq.length
    if qlen == 0 and rlen == 0:
        return 0

    var x = cfg.x
    var o = cfg.o
    var e = cfg.e
    var max_err = cfg.max_error

    # Diagonal index range: at score s, diagonals in [-s, s], clamped to [-rlen, qlen].
    # We allocate for [-max_err, max_err].
    var n_diags = 2 * max_err + 1
    var diag_offset = max_err  # index of diagonal 0

    # Circular buffer sizes
    var m_hist = max(x, o + e) + 1   # M looks back max(x, o+e) steps
    var ie_hist = e + 1               # I/D look back e steps

    comptime NEG_INF = -1000000

    # Flat circular buffers: index = (score % hist_size) * n_diags + (diag + diag_offset)
    var M_buf = List[Int](capacity=m_hist * n_diags)
    var I_buf = List[Int](capacity=ie_hist * n_diags)
    var D_buf = List[Int](capacity=ie_hist * n_diags)

    for _ in range(m_hist * n_diags):
        M_buf.append(NEG_INF)
    for _ in range(ie_hist * n_diags):
        I_buf.append(NEG_INF)
        D_buf.append(NEG_INF)

    # Base case: M[score=0, diag=0] = 0 (zero bases consumed at cost 0)
    M_buf[0 * n_diags + diag_offset] = 0

    # Extend from score 0: advance along diagonal 0 through matches
    var row0 = M_buf[0 * n_diags + diag_offset]
    var col0 = row0 - 0  # diagonal 0 → col == row
    while row0 < qlen and col0 >= 0 and col0 < rlen:
        if query.get_base(row0) != ref_seq.get_base(col0):
            break
        row0 += 1
        col0 += 1
    M_buf[0 * n_diags + diag_offset] = row0

    # Check termination at score 0
    var target_d = qlen - rlen
    if target_d >= -max_err and target_d <= max_err:
        if M_buf[0 * n_diags + diag_offset + target_d] >= qlen:
            return 0

    for s in range(1, max_err + 1):
        var m_slot = s % m_hist
        var ie_slot = s % ie_hist

        # Initialize this score's slots to NEG_INF
        for j in range(n_diags):
            M_buf[m_slot * n_diags + j] = NEG_INF
            I_buf[ie_slot * n_diags + j] = NEG_INF
            D_buf[ie_slot * n_diags + j] = NEG_INF

        var lo = max(-rlen, -s)
        var hi = min(qlen, s)
        if target_d < lo or target_d > hi:
            lo = min(lo, target_d)
            hi = max(hi, target_d)
        lo = max(lo, -max_err)
        hi = min(hi, max_err)

        for d in range(lo, hi + 1):
            var d_idx = d + diag_offset

            # --- I recurrence: I[s,k] = max(M[s-o-e, k-1]+1, I[s-e, k-1]+1)
            var i_val = NEG_INF
            if d_idx > 0:
                if s >= o + e:
                    var mv = M_buf[((s - o - e) % m_hist) * n_diags + d_idx - 1]
                    if mv != NEG_INF:
                        i_val = max(i_val, mv + 1)
                if s >= e:
                    var iv = I_buf[((s - e) % ie_hist) * n_diags + d_idx - 1]
                    if iv != NEG_INF:
                        i_val = max(i_val, iv + 1)
            I_buf[ie_slot * n_diags + d_idx] = i_val

            # --- D recurrence: D[s,k] = max(M[s-o-e, k+1], D[s-e, k+1])
            var d_val = NEG_INF
            if d_idx < n_diags - 1:
                if s >= o + e:
                    var mv = M_buf[((s - o - e) % m_hist) * n_diags + d_idx + 1]
                    if mv != NEG_INF:
                        d_val = max(d_val, mv)
                if s >= e:
                    var dv = D_buf[((s - e) % ie_hist) * n_diags + d_idx + 1]
                    if dv != NEG_INF:
                        d_val = max(d_val, dv)
            D_buf[ie_slot * n_diags + d_idx] = d_val

            # --- M recurrence: M[s,k] = max(M[s-x, k]+1, I[s,k], D[s,k])
            var m_val = NEG_INF
            if s >= x:
                var mv = M_buf[((s - x) % m_hist) * n_diags + d_idx]
                if mv != NEG_INF:
                    m_val = max(m_val, mv + 1)
            if i_val != NEG_INF:
                m_val = max(m_val, i_val)
            if d_val != NEG_INF:
                m_val = max(m_val, d_val)
            M_buf[m_slot * n_diags + d_idx] = m_val

        # --- Extend phase: advance through matches on each active diagonal
        for d in range(lo, hi + 1):
            var d_idx = d + diag_offset
            var row = M_buf[m_slot * n_diags + d_idx]
            if row == NEG_INF or row < 0:
                continue
            var col = row - d
            while row < qlen and col >= 0 and col < rlen:
                if query.get_base(row) != ref_seq.get_base(col):
                    break
                row += 1
                col += 1
            M_buf[m_slot * n_diags + d_idx] = row

        # --- Termination check
        if target_d >= lo and target_d <= hi:
            if M_buf[m_slot * n_diags + diag_offset + target_d] >= qlen:
                return s

    return -1  # exceeded max_error


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
        if ri < 0:
            H_prev[j] = NEG_INF
        elif ri == 0:
            H_prev[j] = 0  # H[-1][-1] = 0: zero query and reference bases consumed
        else:
            H_prev[j] = cfg.gap_open + ri * cfg.gap_extend

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
