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
