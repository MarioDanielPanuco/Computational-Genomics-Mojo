"""
GPU banded Smith-Waterman alignment with wavefront-style diagonal parallelism.

Architecture:
  - One thread-block per sequence pair.
  - Shared memory holds 3 DP rows of width (2*BAND+1) in Int16 for score efficiency.
  - Threads are assigned anti-diagonal cells: at step d, up to min(d+1, 2*BAND+1)
    cells are independent and can be computed in parallel.
  - A second wavefront kernel (wavefront_align_kernel) implements the
    Wavefront Alignment (WFA) approach for long-read scenarios.

Shared memory budget:
  3 rows × (2×BAND+1) × 2 bytes (Int16) per block.
  At BAND=64: 3 × 129 × 2 = 774 bytes — well within the 48 KB shared memory limit.
"""
from std.math import min, max, ceildiv
from std.gpu import block_idx, thread_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from genomics.gpu.device import GenomicsDevice


comptime ALIGN_BLOCK_SIZE: Int = 128
comptime DEFAULT_BAND: Int = 64

# Scoring constants (embedded in kernel; override by launching different params)
comptime MATCH: Int = 2
comptime MISMATCH: Int = -3
comptime GAP_OPEN: Int = -5
comptime GAP_EXT: Int = -2
comptime NEG_INF: Int = -30000  # fits in Int16, safe sentinel

# WFA default penalty model (positive costs)
comptime WFA_MISMATCH: Int = 4   # x: mismatch cost
comptime WFA_GAP_OPEN: Int = 6   # o: gap open cost
comptime WFA_GAP_EXT: Int  = 2   # e: gap extend cost per base
comptime WFA_BLOCK_SIZE: Int = 128


# ===----------------------------------------------------------------------=== #
# Utility: decode base from packed word
# ===----------------------------------------------------------------------=== #

@always_inline
def _decode_base(
    packed_ptr: UnsafePointer[UInt64, MutAnyOrigin],
    pos: Int,
) -> UInt8:
    var word_idx = pos // 32
    var bit_pos = pos % 32
    var shift = UInt64(62 - bit_pos * 2)
    return UInt8((packed_ptr[word_idx] >> shift) & 3)


# ===----------------------------------------------------------------------=== #
# Banded Smith-Waterman kernel
# ===----------------------------------------------------------------------=== #

def banded_sw_kernel[band: Int, LT: TensorLayout](
    queries: TileTensor[DType.uint64, LT, MutAnyOrigin],
    refs: TileTensor[DType.uint64, LT, MutAnyOrigin],
    q_lengths: TileTensor[DType.int32, LT, MutAnyOrigin],
    r_lengths: TileTensor[DType.int32, LT, MutAnyOrigin],
    q_offsets: TileTensor[DType.int32, LT, MutAnyOrigin],
    r_offsets: TileTensor[DType.int32, LT, MutAnyOrigin],
    out_scores: TileTensor[DType.int32, LT, MutAnyOrigin],
):
    """Banded Smith-Waterman: one thread-block per sequence pair.

    Shared memory: 3 rows × (2*band+1) Int16 scores (H only; simplified affine).
    Threads are partitioned across the band width. Synchronization happens at
    each anti-diagonal step.

    out_scores[pair_idx] = best Smith-Waterman score within the band.
    """
    comptime assert band >= 1, "band must be >= 1"
    comptime WIDTH = 2 * band + 1

    var pair_idx = Int(block_idx.x)
    var qlen = Int(rebind[Scalar[DType.int32]](q_lengths[pair_idx]))
    var rlen = Int(rebind[Scalar[DType.int32]](r_lengths[pair_idx]))
    var q_off = Int(rebind[Scalar[DType.int32]](q_offsets[pair_idx]))
    var r_off = Int(rebind[Scalar[DType.int32]](r_offsets[pair_idx]))

    var tid = Int(thread_idx.x)

    # Allocate 3 shared rows for ring-buffer DP (row indices 0/1/2 mod 3)
    var H = stack_allocation[DType.int16,
        address_space=AddressSpace.SHARED](row_major[3, WIDTH]())
    H.fill(0)
    barrier()

    var best_score: Int32 = 0

    for qi in range(qlen):
        var curr_row = (qi + 1) % 3
        var prev_row = qi % 3
        var prev2_row = (qi - 1 + 3) % 3

        # Clear current row
        var j = tid
        while j < WIDTH:
            H[curr_row, j] = rebind[H.ElementType](Int16(0))
            j += Int(block_dim.x)
        barrier()

        # Compute each cell in the band
        j = tid
        while j < WIDTH:
            var ri = qi - band + j
            if ri >= 0 and ri < rlen:
                var q_base = _decode_base(queries.unsafe_ptr() + q_off, qi)
                var r_base = _decode_base(refs.unsafe_ptr() + r_off, ri)
                var sub = MATCH if q_base == r_base else MISMATCH

                # Diagonal predecessor: (qi-1, ri-1) → row prev_row, col j
                var h_diag = Int(rebind[Scalar[DType.int16]](H[prev_row, j])) if j >= 0 and j < WIDTH else NEG_INF
                # Up predecessor: (qi-1, ri) → prev_row, col j+1
                var h_up = Int(rebind[Scalar[DType.int16]](H[prev_row, j + 1])) if j + 1 < WIDTH else NEG_INF
                # Left predecessor: (qi, ri-1) → curr_row, col j-1
                var h_left = Int(rebind[Scalar[DType.int16]](H[curr_row, j - 1])) if j > 0 else NEG_INF

                var cell = max(0, max(h_diag + sub,
                                     max(h_up + GAP_OPEN + GAP_EXT,
                                         h_left + GAP_OPEN + GAP_EXT)))
                H[curr_row, j] = rebind[H.ElementType](Int16(min(cell, 32767)))

                if Int32(cell) > best_score:
                    best_score = Int32(cell)
            j += Int(block_dim.x)

        barrier()

    # Thread 0 writes the result
    if tid == 0:
        out_scores[pair_idx] = rebind[out_scores.ElementType](best_score)


# ===----------------------------------------------------------------------=== #
# Wavefront alignment kernel (WFA-style)
# ===----------------------------------------------------------------------=== #

def wavefront_align_kernel[LT: TensorLayout](
    queries: TileTensor[DType.uint64, LT, MutAnyOrigin],
    refs: TileTensor[DType.uint64, LT, MutAnyOrigin],
    q_lengths: TileTensor[DType.int32, LT, MutAnyOrigin],
    r_lengths: TileTensor[DType.int32, LT, MutAnyOrigin],
    q_offsets: TileTensor[DType.int32, LT, MutAnyOrigin],
    r_offsets: TileTensor[DType.int32, LT, MutAnyOrigin],
    out_scores: TileTensor[DType.int32, LT, MutAnyOrigin],
):
    """Wavefront Alignment (WFA) kernel — edit-distance style.

    Each thread owns one diagonal.  Advances the furthest-reaching point (FRP)
    by extending through matching bases.  Score = number of edit operations.
    Stored in shared memory: wavefront[diag] = FRP row index.

    This implementation covers edit distance (unit costs).  Affine gap scoring
    requires three wavefront arrays (M, I, D) and is left as an extension.
    """
    comptime MAX_DIAGS = 512  # covers sequences up to ~512 bp

    var pair_idx = Int(block_idx.x)
    var qlen = Int(rebind[Scalar[DType.int32]](q_lengths[pair_idx]))
    var rlen = Int(rebind[Scalar[DType.int32]](r_lengths[pair_idx]))
    var q_off = Int(rebind[Scalar[DType.int32]](q_offsets[pair_idx]))
    var r_off = Int(rebind[Scalar[DType.int32]](r_offsets[pair_idx]))
    var tid = Int(thread_idx.x)
    var n_diags = qlen + rlen + 1
    var lo_diag = -rlen
    var target_diag = qlen - rlen

    # Shared wavefront: wf[diag_offset] = FRP row index for that diagonal
    var wf = stack_allocation[DType.int32,
        address_space=AddressSpace.SHARED](row_major[MAX_DIAGS]())
    var wf_prev = stack_allocation[DType.int32,
        address_space=AddressSpace.SHARED](row_major[MAX_DIAGS]())

    # Initialize: score 0, only diagonal 0 is reachable
    if tid < MAX_DIAGS:
        wf[tid] = rebind[wf.ElementType](Int32(NEG_INF))
        wf_prev[tid] = rebind[wf_prev.ElementType](Int32(NEG_INF))
    if tid == 0:
        wf[rlen] = rebind[wf.ElementType](Int32(0))  # diagonal 0 offset by rlen
    barrier()

    var score = 0
    var found = False

    while score <= n_diags and not found:
        # Each thread handles one diagonal
        var d = lo_diag + tid
        var d_off = d + rlen  # offset into wf array

        if d_off >= 0 and d_off < MAX_DIAGS and tid < n_diags:
            var row = Int(rebind[Scalar[DType.int32]](wf[d_off]))
            if row != NEG_INF:
                var col = row - d
                # Extend along matching bases
                while row < qlen and col >= 0 and col < rlen:
                    var qb = _decode_base(queries.unsafe_ptr() + q_off, row)
                    var rb = _decode_base(refs.unsafe_ptr() + r_off, col)
                    if qb != rb:
                        break
                    row += 1
                    col += 1
                wf[d_off] = rebind[wf.ElementType](Int32(row))

                # Check if we've reached the end
                if row >= qlen and col >= rlen:
                    found = True

        barrier()

        if not found:
            # Expand wavefront by 1 edit (substitution, insertion, deletion)
            var d_s = lo_diag + tid
            var d_off_s = d_s + rlen

            var from_sub: Int32 = NEG_INF
            var from_ins: Int32 = NEG_INF
            var from_del: Int32 = NEG_INF

            if d_off_s - 1 >= 0 and d_off_s - 1 < MAX_DIAGS:
                var v = Int(rebind[Scalar[DType.int32]](wf[d_off_s - 1]))
                if v != NEG_INF:
                    from_del = Int32(v + 1)  # deletion: stay on same col, advance row
            if d_off_s + 1 >= 0 and d_off_s + 1 < MAX_DIAGS:
                var v = Int(rebind[Scalar[DType.int32]](wf[d_off_s + 1]))
                if v != NEG_INF:
                    from_ins = Int32(v)      # insertion: advance col, stay on row
            if d_off_s >= 0 and d_off_s < MAX_DIAGS:
                var v = Int(rebind[Scalar[DType.int32]](wf[d_off_s]))
                if v != NEG_INF:
                    from_sub = Int32(v + 1)  # substitution: advance both

            var new_row = max(max(Int(from_sub), Int(from_ins)), Int(from_del))
            wf_prev[d_off_s] = rebind[wf_prev.ElementType](
                Int32(NEG_INF) if new_row == NEG_INF else Int32(new_row))

        barrier()

        # Copy wf_prev to wf
        if tid < MAX_DIAGS:
            var v = rebind[Scalar[DType.int32]](wf_prev[tid])
            if Int(v) > Int(rebind[Scalar[DType.int32]](wf[tid])):
                wf[tid] = rebind[wf.ElementType](v)
        barrier()

        score += 1

    if tid == 0:
        out_scores[pair_idx] = rebind[out_scores.ElementType](Int32(score - 1))


# ===----------------------------------------------------------------------=== #
# Gap-affine WFA kernel
# ===----------------------------------------------------------------------=== #

def wfa_affine_kernel[
    max_error: Int,
    x: Int,
    o: Int,
    e: Int,
    LT: TensorLayout,
](
    queries:    TileTensor[DType.uint64, LT, MutAnyOrigin],
    refs:       TileTensor[DType.uint64, LT, MutAnyOrigin],
    q_lengths:  TileTensor[DType.int32,  LT, MutAnyOrigin],
    r_lengths:  TileTensor[DType.int32,  LT, MutAnyOrigin],
    q_offsets:  TileTensor[DType.int32,  LT, MutAnyOrigin],
    r_offsets:  TileTensor[DType.int32,  LT, MutAnyOrigin],
    out_scores: TileTensor[DType.int32,  LT, MutAnyOrigin],
):
    """Gap-affine WFA global alignment — one thread-block per sequence pair.

    Penalty model: mismatch costs x, gap of length l costs o + l*e.
    Shared memory: three circular-buffer wavefront arrays (M, I, D).
      M[s,k] — furthest query row on diagonal k at score s (match/mismatch end)
      I[s,k] — same, ending in insertion (gap in reference; row advances)
      D[s,k] — same, ending in deletion  (gap in query; col advances)
    Diagonal k = row - col; target diagonal = qlen - rlen.
    NEXT phase expands wavefronts; EXTEND phase advances through matches.
    out_scores[pair_idx] = min cost, or -1 if cost > max_error.
    """
    comptime assert max_error >= 1, "max_error must be >= 1"
    comptime assert e >= 1, "gap extend cost must be >= 1"
    comptime MAX_DIAGS = 2 * max_error + 1
    # M needs history back max(x, o+e) steps; +1 for circular buffer
    comptime M_HIST_SIZE = (x if x > o + e else o + e) + 1
    comptime IE_HIST_SIZE = e + 1
    comptime WFA_NEG = -30000  # fits in Int32, safe NEG_INF sentinel

    var pair_idx = Int(block_idx.x)
    var qlen  = Int(rebind[Scalar[DType.int32]](q_lengths[pair_idx]))
    var rlen  = Int(rebind[Scalar[DType.int32]](r_lengths[pair_idx]))
    var q_off = Int(rebind[Scalar[DType.int32]](q_offsets[pair_idx]))
    var r_off = Int(rebind[Scalar[DType.int32]](r_offsets[pair_idx]))
    var tid   = Int(thread_idx.x)
    var target_d = qlen - rlen

    # Guard: if length difference alone exceeds budget, no alignment possible.
    if target_d < -max_error or target_d > max_error:
        if tid == 0:
            out_scores[pair_idx] = rebind[out_scores.ElementType](Int32(-1))
        return

    # Shared memory circular buffers: flat index = slot * MAX_DIAGS + (d + max_error)
    # M history: M_HIST_SIZE slots; I/D history: IE_HIST_SIZE slots
    var M_sh = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](
        row_major[M_HIST_SIZE * MAX_DIAGS]()
    )
    var I_sh = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](
        row_major[IE_HIST_SIZE * MAX_DIAGS]()
    )
    var D_sh = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](
        row_major[IE_HIST_SIZE * MAX_DIAGS]()
    )
    # found_sh[0] = 0 (not found), or s+1 (found at score s)
    var found_sh = stack_allocation[DType.int32, address_space=AddressSpace.SHARED](
        row_major[1]()
    )

    # Initialize all history to WFA_NEG (thread-stride loop)
    var j = tid
    while j < M_HIST_SIZE * MAX_DIAGS:
        M_sh[j] = rebind[M_sh.ElementType](Int32(WFA_NEG))
        j += Int(block_dim.x)
    j = tid
    while j < IE_HIST_SIZE * MAX_DIAGS:
        I_sh[j] = rebind[I_sh.ElementType](Int32(WFA_NEG))
        D_sh[j] = rebind[D_sh.ElementType](Int32(WFA_NEG))
        j += Int(block_dim.x)
    if tid == 0:
        found_sh[0] = rebind[found_sh.ElementType](Int32(0))
        M_sh[0 * MAX_DIAGS + max_error] = rebind[M_sh.ElementType](Int32(0))
    barrier()

    # Extend score 0 on diagonal 0 (only diagonal active at s=0)
    if tid == 0:
        var row = Int(rebind[Scalar[DType.int32]](M_sh[0 * MAX_DIAGS + max_error]))
        var col = row  # diagonal 0 → col == row
        while row < qlen and col >= 0 and col < rlen:
            if _decode_base(queries.unsafe_ptr() + q_off, row) != _decode_base(refs.unsafe_ptr() + r_off, col):
                break
            row += 1
            col += 1
        M_sh[0 * MAX_DIAGS + max_error] = rebind[M_sh.ElementType](Int32(row))
        if target_d == 0 and row >= qlen:
            found_sh[0] = rebind[found_sh.ElementType](Int32(1))  # score 0 + 1
    barrier()

    if Int(rebind[Scalar[DType.int32]](found_sh[0])) != 0:
        if tid == 0:
            out_scores[pair_idx] = rebind[out_scores.ElementType](Int32(0))
        return

    for s in range(1, max_error + 1):
        var m_slot  = s % M_HIST_SIZE
        var ie_slot = s % IE_HIST_SIZE

        # Clear this score's slots before writing (thread-stride loop)
        var j2 = tid
        while j2 < MAX_DIAGS:
            M_sh[m_slot  * MAX_DIAGS + j2] = rebind[M_sh.ElementType](Int32(WFA_NEG))
            I_sh[ie_slot * MAX_DIAGS + j2] = rebind[I_sh.ElementType](Int32(WFA_NEG))
            D_sh[ie_slot * MAX_DIAGS + j2] = rebind[D_sh.ElementType](Int32(WFA_NEG))
            j2 += Int(block_dim.x)
        barrier()

        # Active diagonal range at score s
        var lo = max(max(-rlen, -s), -max_error)
        var hi = min(min(qlen,   s),  max_error)

        # NEXT phase: expand wavefront for score s (threads stride over diagonals)
        var d = lo + tid
        while d <= hi:
            var d_idx = d + max_error

            # I[s,k] = max(M[s-o-e, k-1]+1, I[s-e, k-1]+1)  (insertion: row advances)
            var i_val = WFA_NEG
            if d_idx > 0:
                if s >= o + e:
                    var mv = Int(rebind[Scalar[DType.int32]](
                        M_sh[((s - o - e) % M_HIST_SIZE) * MAX_DIAGS + d_idx - 1]))
                    if mv != WFA_NEG:
                        i_val = max(i_val, mv + 1)
                if s >= e:
                    var iv = Int(rebind[Scalar[DType.int32]](
                        I_sh[((s - e) % IE_HIST_SIZE) * MAX_DIAGS + d_idx - 1]))
                    if iv != WFA_NEG:
                        i_val = max(i_val, iv + 1)
            I_sh[ie_slot * MAX_DIAGS + d_idx] = rebind[I_sh.ElementType](Int32(i_val))

            # D[s,k] = max(M[s-o-e, k+1], D[s-e, k+1])  (deletion: col advances)
            var d_val = WFA_NEG
            if d_idx < MAX_DIAGS - 1:
                if s >= o + e:
                    var mv = Int(rebind[Scalar[DType.int32]](
                        M_sh[((s - o - e) % M_HIST_SIZE) * MAX_DIAGS + d_idx + 1]))
                    if mv != WFA_NEG:
                        d_val = max(d_val, mv)
                if s >= e:
                    var dv = Int(rebind[Scalar[DType.int32]](
                        D_sh[((s - e) % IE_HIST_SIZE) * MAX_DIAGS + d_idx + 1]))
                    if dv != WFA_NEG:
                        d_val = max(d_val, dv)
            D_sh[ie_slot * MAX_DIAGS + d_idx] = rebind[D_sh.ElementType](Int32(d_val))

            # M[s,k] = max(M[s-x, k]+1, I[s,k], D[s,k])
            var m_val = WFA_NEG
            if s >= x:
                var mv = Int(rebind[Scalar[DType.int32]](
                    M_sh[((s - x) % M_HIST_SIZE) * MAX_DIAGS + d_idx]))
                if mv != WFA_NEG:
                    m_val = max(m_val, mv + 1)
            if i_val != WFA_NEG:
                m_val = max(m_val, i_val)
            if d_val != WFA_NEG:
                m_val = max(m_val, d_val)
            M_sh[m_slot * MAX_DIAGS + d_idx] = rebind[M_sh.ElementType](Int32(m_val))

            d += Int(block_dim.x)
        barrier()

        # EXTEND phase: advance through matching bases (threads stride over diagonals)
        d = lo + tid
        while d <= hi:
            var d_idx = d + max_error
            var row = Int(rebind[Scalar[DType.int32]](M_sh[m_slot * MAX_DIAGS + d_idx]))
            if row != WFA_NEG and row >= 0:
                var col = row - d
                while row < qlen and col >= 0 and col < rlen:
                    if _decode_base(queries.unsafe_ptr() + q_off, row) != _decode_base(refs.unsafe_ptr() + r_off, col):
                        break
                    row += 1
                    col += 1
                M_sh[m_slot * MAX_DIAGS + d_idx] = rebind[M_sh.ElementType](Int32(row))
                if d == target_d and row >= qlen:
                    found_sh[0] = rebind[found_sh.ElementType](Int32(s + 1))
            d += Int(block_dim.x)
        barrier()

        if Int(rebind[Scalar[DType.int32]](found_sh[0])) != 0:
            if tid == 0:
                var result_s = Int(rebind[Scalar[DType.int32]](found_sh[0])) - 1
                out_scores[pair_idx] = rebind[out_scores.ElementType](Int32(result_s))
            return

    # Exceeded max_error
    if tid == 0:
        out_scores[pair_idx] = rebind[out_scores.ElementType](Int32(-1))


# ===----------------------------------------------------------------------=== #
# Host-side launch wrappers
# ===----------------------------------------------------------------------=== #

def launch_banded_sw[band: Int](
    device: GenomicsDevice,
    queries_buf: DeviceBuffer[DType.uint64],
    refs_buf: DeviceBuffer[DType.uint64],
    q_lengths_buf: DeviceBuffer[DType.int32],
    r_lengths_buf: DeviceBuffer[DType.int32],
    q_offsets_buf: DeviceBuffer[DType.int32],
    r_offsets_buf: DeviceBuffer[DType.int32],
    n_pairs: Int,
    total_words: Int,
) raises -> DeviceBuffer[DType.int32]:
    """Launch banded Smith-Waterman for n_pairs sequence pairs.

    Returns a device buffer of length n_pairs with alignment scores.
    """
    var scores_buf = device.ctx.enqueue_create_buffer[DType.int32](n_pairs)
    scores_buf.enqueue_fill(0)

    var flat_layout = row_major(Idx(total_words))
    var pair_layout = row_major(Idx(n_pairs))

    comptime kernel = banded_sw_kernel[band, type_of(flat_layout)]
    device.ctx.enqueue_function[kernel, kernel](
        TileTensor(queries_buf, flat_layout),
        TileTensor(refs_buf, flat_layout),
        TileTensor(q_lengths_buf, pair_layout),
        TileTensor(r_lengths_buf, pair_layout),
        TileTensor(q_offsets_buf, pair_layout),
        TileTensor(r_offsets_buf, pair_layout),
        TileTensor(scores_buf, pair_layout),
        grid_dim=n_pairs,
        block_dim=ALIGN_BLOCK_SIZE,
    )
    return scores_buf


def launch_wfa_affine[
    max_error: Int,
    x: Int,
    o: Int,
    e: Int,
](
    device: GenomicsDevice,
    queries_buf: DeviceBuffer[DType.uint64],
    refs_buf: DeviceBuffer[DType.uint64],
    q_lengths_buf: DeviceBuffer[DType.int32],
    r_lengths_buf: DeviceBuffer[DType.int32],
    q_offsets_buf: DeviceBuffer[DType.int32],
    r_offsets_buf: DeviceBuffer[DType.int32],
    n_pairs: Int,
    total_words: Int,
) raises -> DeviceBuffer[DType.int32]:
    """Launch gap-affine WFA for n_pairs sequence pairs.

    Returns a device buffer of length n_pairs with minimum alignment costs.
    A value of -1 indicates the cost exceeded max_error.
    """
    var scores_buf = device.ctx.enqueue_create_buffer[DType.int32](n_pairs)
    scores_buf.enqueue_fill(0)

    var flat_layout = row_major(Idx(total_words))
    var pair_layout = row_major(Idx(n_pairs))

    comptime kernel = wfa_affine_kernel[max_error, x, o, e, type_of(flat_layout)]
    device.ctx.enqueue_function[kernel, kernel](
        TileTensor(queries_buf, flat_layout),
        TileTensor(refs_buf, flat_layout),
        TileTensor(q_lengths_buf, pair_layout),
        TileTensor(r_lengths_buf, pair_layout),
        TileTensor(q_offsets_buf, pair_layout),
        TileTensor(r_offsets_buf, pair_layout),
        TileTensor(scores_buf, pair_layout),
        grid_dim=n_pairs,
        block_dim=WFA_BLOCK_SIZE,
    )
    return scores_buf
