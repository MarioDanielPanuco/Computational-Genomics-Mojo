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


@fieldwise_init
struct WFAResult(Movable):
    """WFA alignment result with score and extended CIGAR traceback."""
    var score: Int
    var cigar: String  # run-length extended CIGAR e.g. "3=1X2I5=" (=match X mismatch I ins D del)


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


def wfa_affine_cigar_cpu(
    query: SequenceView,
    ref_seq: SequenceView,
    cfg: WFAConfig,
) -> WFAResult:
    """Gap-affine WFA with full CIGAR traceback via complete wavefront history.

    Unlike wfa_affine_cpu (O(s) circular buffer), this stores all wavefront
    scores — O(s * max_diags) memory — so backtracking is possible.

    Returns WFAResult with the optimal cost and an extended CIGAR string using
    '=' (sequence match), 'X' (mismatch), 'I' (insertion), 'D' (deletion).
    Returns score=-1 with empty CIGAR if alignment cost exceeds cfg.max_error.
    """
    var qlen = query.length
    var rlen = ref_seq.length
    if qlen == 0 and rlen == 0:
        return WFAResult(score=0, cigar=String(""))

    var x = cfg.x
    var o = cfg.o
    var e = cfg.e
    var max_err = cfg.max_error

    var n_diags = 2 * max_err + 1
    var offset = max_err
    comptime NEG_INF = -1_000_000

    # Full wavefront history (not circular) so backtracking can reach any score.
    # Index: s * n_diags + (diagonal + offset)
    var total = (max_err + 1) * n_diags
    var M_pre  = List[Int](capacity=total)   # furthest row BEFORE extend phase
    var M_full = List[Int](capacity=total)   # furthest row AFTER  extend phase
    var I_arr  = List[Int](capacity=total)   # I wavefront (insertion state)
    var D_arr  = List[Int](capacity=total)   # D wavefront (deletion  state)

    for _ in range(total):
        M_pre.append(NEG_INF)
        M_full.append(NEG_INF)
        I_arr.append(NEG_INF)
        D_arr.append(NEG_INF)

    # --- Score 0 base case: diagonal 0, extend through leading matches ---
    M_pre[offset] = 0
    var r0 = 0
    while r0 < qlen and r0 < rlen and query.get_base(r0) == ref_seq.get_base(r0):
        r0 += 1
    M_full[offset] = r0

    var target_d = qlen - rlen
    var opt_score = -1

    if target_d >= -max_err and target_d <= max_err:
        if M_full[offset + target_d] >= qlen:
            opt_score = 0

    # --- Main WFA loop ---
    if opt_score == -1:
        for s in range(1, max_err + 1):
            var lo = max(-rlen, -s)
            var hi = min(qlen, s)
            if target_d < lo:
                lo = min(lo, target_d)
            if target_d > hi:
                hi = max(hi, target_d)
            lo = max(lo, -max_err)
            hi = min(hi, max_err)

            for d in range(lo, hi + 1):
                var di = d + offset

                # I[s,d] = max(M_full[s-o-e, d-1]+1, I[s-e, d-1]+1)
                var i_val = NEG_INF
                if di > 0:
                    if s >= o + e:
                        var mv = M_full[(s - o - e) * n_diags + di - 1]
                        if mv != NEG_INF:
                            i_val = max(i_val, mv + 1)
                    if s >= e:
                        var iv = I_arr[(s - e) * n_diags + di - 1]
                        if iv != NEG_INF:
                            i_val = max(i_val, iv + 1)
                I_arr[s * n_diags + di] = i_val

                # D[s,d] = max(M_full[s-o-e, d+1], D[s-e, d+1])
                var d_val = NEG_INF
                if di < n_diags - 1:
                    if s >= o + e:
                        var mv = M_full[(s - o - e) * n_diags + di + 1]
                        if mv != NEG_INF:
                            d_val = max(d_val, mv)
                    if s >= e:
                        var dv = D_arr[(s - e) * n_diags + di + 1]
                        if dv != NEG_INF:
                            d_val = max(d_val, dv)
                D_arr[s * n_diags + di] = d_val

                # M[s,d] = max(M_full[s-x, d]+1, I[s,d], D[s,d])
                var m_val = NEG_INF
                if s >= x:
                    var mv = M_full[(s - x) * n_diags + di]
                    if mv != NEG_INF:
                        m_val = max(m_val, mv + 1)
                if i_val != NEG_INF:
                    m_val = max(m_val, i_val)
                if d_val != NEG_INF:
                    m_val = max(m_val, d_val)
                M_pre[s * n_diags + di] = m_val

                # Extend through matches on this diagonal
                if m_val != NEG_INF:
                    var row = m_val
                    var col = row - d
                    while row < qlen and col >= 0 and col < rlen:
                        if query.get_base(row) != ref_seq.get_base(col):
                            break
                        row += 1
                        col += 1
                    M_full[s * n_diags + di] = row

            # Termination check
            if target_d >= lo and target_d <= hi:
                if M_full[s * n_diags + offset + target_d] >= qlen:
                    opt_score = s
                    break

    if opt_score == -1:
        return WFAResult(score=-1, cigar=String(""))

    # --- Backtrack: build ops in reverse (last alignment op first) ---
    # Op codes: 0='=', 1='X', 2='I', 3='D'
    var ops_rev = List[UInt8]()
    var cur_s = opt_score
    var cur_d = target_d
    var cur_matrix = 0   # 0=M, 1=I, 2=D
    var cur_row = M_full[opt_score * n_diags + target_d + offset]  # = qlen

    while True:
        var di = cur_d + offset

        if cur_matrix == 0:  # --- M state ---
            var pre_r = M_pre[cur_s * n_diags + di]
            # Emit match run (extend phase)
            for _ in range(cur_row - pre_r):
                ops_rev.append(UInt8(0))  # '='
            cur_row = pre_r

            if cur_s == 0 and cur_d == 0:
                break  # reached initial state; pre_r == 0

            # Priority: mismatch, then I, then D (any valid optimal path)
            if cur_s >= x:
                var prev_m = M_full[(cur_s - x) * n_diags + di]
                if prev_m != NEG_INF and prev_m + 1 == cur_row:
                    ops_rev.append(UInt8(1))  # 'X'
                    cur_s -= x
                    cur_row = prev_m   # = M_full[cur_s][cur_d] (extended)
                    continue

            var i_here = I_arr[cur_s * n_diags + di]
            if i_here != NEG_INF and i_here == cur_row:
                cur_matrix = 1
                continue      # cur_row already equals i_here

            var d_here = D_arr[cur_s * n_diags + di]
            if d_here != NEG_INF and d_here == cur_row:
                cur_matrix = 2
                continue

            break  # should not reach here on a valid WFA trace

        elif cur_matrix == 1:  # --- I state: row advanced, diagonal increased ---
            ops_rev.append(UInt8(2))  # 'I'
            var prev_d = cur_d - 1
            var prev_di = prev_d + offset

            # Prefer gap-open source (M) over gap-extend source (I)
            if cur_s >= o + e and prev_di >= 0:
                var mv = M_full[(cur_s - o - e) * n_diags + prev_di]
                if mv != NEG_INF and mv + 1 == cur_row:
                    cur_s -= o + e
                    cur_d = prev_d
                    cur_row = mv   # extended M at new (cur_s, cur_d)
                    cur_matrix = 0
                    continue

            if cur_s >= e and prev_di >= 0:
                var iv = I_arr[(cur_s - e) * n_diags + prev_di]
                if iv != NEG_INF and iv + 1 == cur_row:
                    cur_s -= e
                    cur_d = prev_d
                    cur_row = iv   # I value at new (cur_s, cur_d)
                    cur_matrix = 1
                    continue

            break  # should not reach here

        else:  # cur_matrix == 2, --- D state: col advanced, diagonal decreased ---
            ops_rev.append(UInt8(3))  # 'D'
            var prev_d = cur_d + 1
            var prev_di = prev_d + offset

            # Prefer gap-open source (M) over gap-extend source (D)
            if cur_s >= o + e and prev_di < n_diags:
                var mv = M_full[(cur_s - o - e) * n_diags + prev_di]
                if mv != NEG_INF and mv == cur_row:
                    cur_s -= o + e
                    cur_d = prev_d
                    cur_row = mv
                    cur_matrix = 0
                    continue

            if cur_s >= e and prev_di < n_diags:
                var dv = D_arr[(cur_s - e) * n_diags + prev_di]
                if dv != NEG_INF and dv == cur_row:
                    cur_s -= e
                    cur_d = prev_d
                    cur_row = dv
                    cur_matrix = 2
                    continue

            break  # should not reach here

    # --- Build RLE CIGAR string (ops_rev is reversed; read from end to start) ---
    var cigar = String("")
    var ops_len = len(ops_rev)
    var i = ops_len - 1
    while i >= 0:
        var op = ops_rev[i]
        var cnt = 1
        while i - cnt >= 0 and ops_rev[i - cnt] == op:
            cnt += 1
        cigar += String(cnt)
        if op == 0:
            cigar += "="
        elif op == 1:
            cigar += "X"
        elif op == 2:
            cigar += "I"
        else:
            cigar += "D"
        i -= cnt

    return WFAResult(score=opt_score, cigar=cigar)
