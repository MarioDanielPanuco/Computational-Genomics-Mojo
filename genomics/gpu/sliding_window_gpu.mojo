"""
GPU sliding-window transforms: GC content and Shannon entropy.

One thread-block per sequence.  Each thread computes one or more windows
using a prefix-sum approach stored in shared memory.

Memory layout:
  - Shared: prefix GC count array of length (block_size + window) UInt16
  - Each block loads one sequence's packed words, computes the prefix sum,
    then each thread reads two prefix entries to get its window's GC count.
"""
from std.math import ceildiv, log2
from std.gpu import block_idx, thread_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from genomics.core.dna import BASES_PER_WORD, BASE_BITS, Base
from genomics.gpu.device import GenomicsDevice


comptime SW_BLOCK_SIZE: Int = 256
comptime SW_TILE: Int = 64   # bases loaded per tile into shared memory


# ===----------------------------------------------------------------------=== #
# GC content kernel
# ===----------------------------------------------------------------------=== #

def gc_content_kernel[window: Int, LT: TensorLayout](
    packed: TileTensor[DType.uint64, LT, MutAnyOrigin],
    lengths: TileTensor[DType.int32, LT, MutAnyOrigin],
    offsets: TileTensor[DType.int32, LT, MutAnyOrigin],
    out_gc: TileTensor[DType.float32, LT, MutAnyOrigin],
):
    """Compute per-position GC fraction in a sliding window for each sequence.

    Block-per-sequence.  Each thread handles one output position.
    Uses a shared-memory prefix-sum of per-base GC indicators for O(1) window queries.
    """
    comptime assert window >= 1, "window must be >= 1"

    var seq_idx = Int(block_idx.x)
    var seq_len = Int(rebind[Scalar[DType.int32]](lengths[seq_idx]))
    var word_off = Int(rebind[Scalar[DType.int32]](offsets[seq_idx]))
    var n_out = seq_len - window + 1
    if n_out <= 0:
        return

    # Shared prefix array: prefix[i] = sum of GC flags for bases [word_off*32, word_off*32+i)
    # Size: seq_len + 1  (capped in practice by launching blocks for reasonable seq lengths)
    var prefix = stack_allocation[DType.int32,
        address_space=AddressSpace.SHARED](row_major[4096]())

    var tid = Int(thread_idx.x)
    var stride = Int(block_dim.x)

    # Phase 1: Each thread fills prefix[tid+1] for base at position tid
    # (prefix[0] = 0 is implied; we compute prefix[pos+1])
    var pos = tid
    while pos < seq_len:
        var word_idx = pos // BASES_PER_WORD
        var bit_pos = pos % BASES_PER_WORD
        var w = rebind[Scalar[DType.uint64]](packed[word_off + word_idx])
        var shift = UInt64(62 - bit_pos * 2)
        var b = (UInt64(w) >> shift) & 3
        # b == 1 (C) or b == 2 (G) → is_gc
        var is_gc = Int((b == 1 or b == 2))
        prefix[pos + 1] = is_gc
        pos += stride

    barrier()

    # Phase 2: Inclusive prefix sum (single thread for simplicity; production
    # would use a parallel scan, e.g. Blelloch or Kogge-Stone).
    if tid == 0:
        prefix[0] = 0
        for i in range(seq_len):
            prefix[i + 1] += prefix[i]

    barrier()

    # Phase 3: Each thread outputs one GC fraction
    pos = tid
    var inv_w = Float32(1.0) / Float32(window)
    while pos < n_out:
        var gc = rebind[Scalar[DType.int32]](prefix[pos + window]) - rebind[Scalar[DType.int32]](prefix[pos])
        out_gc[word_off + pos] = rebind[out_gc.ElementType](Float32(gc) * inv_w)
        pos += stride


# ===----------------------------------------------------------------------=== #
# Entropy kernel
# ===----------------------------------------------------------------------=== #

def entropy_kernel[window: Int, LT: TensorLayout](
    packed: TileTensor[DType.uint64, LT, MutAnyOrigin],
    lengths: TileTensor[DType.int32, LT, MutAnyOrigin],
    offsets: TileTensor[DType.int32, LT, MutAnyOrigin],
    out_entropy: TileTensor[DType.float32, LT, MutAnyOrigin],
):
    """Compute per-position Shannon entropy (bits) of base distribution in each window.

    Strategy: for each output position, each thread independently computes a
    4-bin frequency histogram over its window and evaluates H = -Σ p*log2(p).
    This avoids shared-memory coordination at the cost of O(window) work per
    thread. For large windows, a sliding-histogram approach is more efficient.
    """
    comptime assert window >= 1, "window must be >= 1"

    var seq_idx = Int(block_idx.x)
    var seq_len = Int(rebind[Scalar[DType.int32]](lengths[seq_idx]))
    var word_off = Int(rebind[Scalar[DType.int32]](offsets[seq_idx]))
    var n_out = seq_len - window + 1
    if n_out <= 0:
        return

    var tid = Int(thread_idx.x)
    var stride = Int(block_dim.x)

    var pos = tid
    while pos < n_out:
        # Build 4-base histogram for window [pos, pos+window)
        var freq = [0, 0, 0, 0]
        var valid_count = 0

        for i in range(window):
            var abs_pos = pos + i
            var word_idx = abs_pos // BASES_PER_WORD
            var bit_pos = abs_pos % BASES_PER_WORD
            var w = rebind[Scalar[DType.uint64]](packed[word_off + word_idx])
            var shift = UInt64(62 - bit_pos * 2)
            var b = Int((UInt64(w) >> shift) & 3)
            freq[b] += 1
            valid_count += 1

        var h: Float32 = 0.0
        if valid_count > 0:
            var inv_total = Float32(1.0) / Float32(valid_count)
            for b in range(4):
                if freq[b] > 0:
                    var p = Float32(freq[b]) * inv_total
                    h -= p * log2(p)

        out_entropy[word_off + pos] = rebind[out_entropy.ElementType](h)
        pos += stride


# ===----------------------------------------------------------------------=== #
# Host-side launch wrappers
# ===----------------------------------------------------------------------=== #

def launch_gc_content[window: Int](
    device: GenomicsDevice,
    packed_buf: DeviceBuffer[DType.uint64],
    lengths_buf: DeviceBuffer[DType.int32],
    offsets_buf: DeviceBuffer[DType.int32],
    n_seqs: Int,
    total_words: Int,
) raises -> DeviceBuffer[DType.float32]:
    """Launch GC content kernel. Returns output buffer of length total_words."""
    var out_buf = device.ctx.enqueue_create_buffer[DType.float32](total_words)
    out_buf.enqueue_fill(0.0)

    var layout = row_major(Idx(total_words))
    var len_layout = row_major(Idx(n_seqs))

    comptime kernel = gc_content_kernel[window, type_of(layout)]
    device.ctx.enqueue_function[kernel, kernel](
        TileTensor(packed_buf, layout),
        TileTensor(lengths_buf, len_layout),
        TileTensor(offsets_buf, len_layout),
        TileTensor(out_buf, layout),
        grid_dim=n_seqs,
        block_dim=SW_BLOCK_SIZE,
    )
    return out_buf


def launch_entropy[window: Int](
    device: GenomicsDevice,
    packed_buf: DeviceBuffer[DType.uint64],
    lengths_buf: DeviceBuffer[DType.int32],
    offsets_buf: DeviceBuffer[DType.int32],
    n_seqs: Int,
    total_words: Int,
) raises -> DeviceBuffer[DType.float32]:
    """Launch entropy kernel. Returns output buffer of length total_words."""
    var out_buf = device.ctx.enqueue_create_buffer[DType.float32](total_words)
    out_buf.enqueue_fill(0.0)

    var layout = row_major(Idx(total_words))
    var len_layout = row_major(Idx(n_seqs))

    comptime kernel = entropy_kernel[window, type_of(layout)]
    device.ctx.enqueue_function[kernel, kernel](
        TileTensor(packed_buf, layout),
        TileTensor(lengths_buf, len_layout),
        TileTensor(offsets_buf, len_layout),
        TileTensor(out_buf, layout),
        grid_dim=n_seqs,
        block_dim=SW_BLOCK_SIZE,
    )
    return out_buf
