"""
GPU k-mer extraction kernel.

Each thread-block handles one sequence.  Threads within a block cooperate to
extract all k-mers from that sequence in parallel, with each thread responsible
for BASES_PER_THREAD consecutive positions.

Coalescing: packed UInt64 words for a sequence are stored row-major in the
batch buffer.  Consecutive threads within a block read consecutive words from
the same row → coalesced 512-bit cache-line reads.

Output: out_kmers[seq_offset + pos] = canonical k-mer hash for position pos.
        out_valid[seq_offset + pos] = 0 if any N in window, 1 otherwise.
"""
from std.math import ceildiv
from std.gpu import global_idx, block_idx, thread_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from genomics.core.dna import (
    BASES_PER_WORD, BASE_BITS, complement_word, reverse_2bit_pairs,
)
from genomics.core.kmer import Kmer
from genomics.gpu.device import GenomicsDevice


comptime KMER_BLOCK_SIZE: Int = 256


@always_inline
def _extract_base_from_packed(
    packed_ptr: UnsafePointer[UInt64, MutAnyOrigin],
    pos: Int,
) -> UInt8:
    var word_idx = pos // BASES_PER_WORD
    var bit_pos = pos % BASES_PER_WORD
    var shift = 62 - bit_pos * BASE_BITS
    return UInt8((packed_ptr[word_idx] >> UInt64(shift)) & 3)


@always_inline
def _compute_kmer_bits[k: Int](
    packed_ptr: UnsafePointer[UInt64, MutAnyOrigin],
    start: Int,
) -> UInt64:
    """Build a k-mer by reading k consecutive bases from a packed array."""
    var bits: UInt64 = 0
    for i in range(k):
        var b = _extract_base_from_packed(packed_ptr, start + i)
        bits = (bits << 2) | UInt64(b)
    # Left-align: shift to top 2*k bits
    return bits << UInt64(64 - 2 * k)


@always_inline
def _has_n_in_range(
    nmask_ptr: UnsafePointer[UInt64, MutAnyOrigin],
    start: Int,
    length: Int,
) -> Bool:
    """Return True if any N bit is set in nmask for positions [start, start+length)."""
    for i in range(length):
        var pos = start + i
        var word_idx = pos // BASES_PER_WORD
        var bit_pos = pos % BASES_PER_WORD
        if (nmask_ptr[word_idx] >> UInt64(bit_pos)) & 1 == 1:
            return True
    return False


@always_inline
def _canonical_kmer_bits[k: Int](bits: UInt64) -> UInt64:
    """Return min(bits, rev_comp_bits) for strand-agnostic canonical form."""
    var rc = reverse_2bit_pairs(complement_word(bits)) << UInt64(64 - 2 * k)
    return rc if rc < bits else bits


@always_inline
def _murmurhash3_mix(h: UInt64) -> UInt64:
    var x = h
    x ^= x >> 33
    x *= UInt64(0xFF51AFD7ED558CCD)
    x ^= x >> 33
    x *= UInt64(0xC4CEB9FE1A85EC53)
    x ^= x >> 33
    return x


# ===----------------------------------------------------------------------=== #
# GPU kernel
# ===----------------------------------------------------------------------=== #

def kmer_extract_kernel[k: Int, LT: TensorLayout](
    packed: TileTensor[DType.uint64, LT, MutAnyOrigin],
    nmasks: TileTensor[DType.uint64, LT, MutAnyOrigin],
    lengths: TileTensor[DType.int32, LT, MutAnyOrigin],
    offsets: TileTensor[DType.int32, LT, MutAnyOrigin],
    out_kmers: TileTensor[DType.uint64, LT, MutAnyOrigin],
    out_valid: TileTensor[DType.uint8, LT, MutAnyOrigin],
):
    """One thread-block per sequence.  Each thread processes KMER_BLOCK_SIZE positions.

    packed[seq_word_offset + word] = packed UInt64 words for the sequence.
    offsets[seq_idx]               = first word index for sequence seq_idx.
    lengths[seq_idx]               = sequence length in bases.
    out_kmers[seq_word_offset + p] = canonical hash for k-mer at position p.
    out_valid[seq_word_offset + p] = 1 if no N in window, else 0.
    """
    comptime assert k >= 1 and k <= 32, "k must be in [1, 32]"

    var seq_idx = Int(block_idx.x)
    var seq_len = Int(rebind[Scalar[DType.int32]](lengths[seq_idx]))
    var word_off = Int(rebind[Scalar[DType.int32]](offsets[seq_idx]))
    var n_kmers = seq_len - k + 1

    if n_kmers <= 0:
        return

    var tid = Int(thread_idx.x)
    var stride = Int(block_dim.x)

    # Each thread processes positions: tid, tid+stride, tid+2*stride, ...
    var pos = tid
    while pos < n_kmers:
        # Build k-mer bits from packed array
        var bits: UInt64 = 0
        for i in range(k):
            var abs_pos = pos + i
            var word_idx = abs_pos // BASES_PER_WORD
            var bit_pos = abs_pos % BASES_PER_WORD
            var w = rebind[Scalar[DType.uint64]](packed[word_off + word_idx])
            var shift = UInt64(62 - bit_pos * 2)
            bits = (bits << 2) | ((UInt64(w) >> shift) & 3)

        # Left-align in UInt64
        bits <<= UInt64(64 - 2 * k)

        # Check for N bases in this window
        var has_n: Bool = False
        for i in range(k):
            var abs_pos = pos + i
            var word_idx = abs_pos // BASES_PER_WORD
            var bit_pos = abs_pos % BASES_PER_WORD
            var nm = rebind[Scalar[DType.uint64]](nmasks[word_off + word_idx])
            if (UInt64(nm) >> UInt64(bit_pos)) & 1 == 1:
                has_n = True
                break

        # Canonical form and hash
        var canonical_bits = _canonical_kmer_bits[k](bits)
        var hashed = _murmurhash3_mix(canonical_bits)

        out_kmers[word_off + pos] = rebind[out_kmers.ElementType](hashed)
        out_valid[word_off + pos] = rebind[out_valid.ElementType](UInt8(0 if has_n else 1))

        pos += stride


# ===----------------------------------------------------------------------=== #
# Host-side launch wrapper
# ===----------------------------------------------------------------------=== #

def launch_kmer_extract[k: Int](
    device: GenomicsDevice,
    packed_buf: DeviceBuffer[DType.uint64],
    nmask_buf: DeviceBuffer[DType.uint64],
    lengths_buf: DeviceBuffer[DType.int32],
    offsets_buf: DeviceBuffer[DType.int32],
    n_seqs: Int,
    total_words: Int,
) raises -> (DeviceBuffer[DType.uint64], DeviceBuffer[DType.uint8]):
    """Allocate output buffers and launch the k-mer extraction kernel.

    Returns (out_kmers_buf, out_valid_buf), each of length total_words
    (positions beyond n_kmers for a sequence are left uninitialized).
    """
    var out_kmers = device.ctx.enqueue_create_buffer[DType.uint64](total_words)
    var out_valid = device.ctx.enqueue_create_buffer[DType.uint8](total_words)
    out_kmers.enqueue_fill(0)
    out_valid.enqueue_fill(0)

    var layout = row_major(Idx(total_words))
    var packed_t = TileTensor(packed_buf, layout)
    var nmask_t = TileTensor(nmask_buf, layout)
    var out_k_t = TileTensor(out_kmers, layout)
    var out_v_t = TileTensor(out_valid, layout)

    var len_layout = row_major(Idx(n_seqs))
    var lengths_t = TileTensor(lengths_buf, len_layout)
    var offsets_t = TileTensor(offsets_buf, len_layout)

    comptime kernel = kmer_extract_kernel[k, type_of(layout)]
    device.ctx.enqueue_function[kernel, kernel](
        packed_t, nmask_t, lengths_t, offsets_t, out_k_t, out_v_t,
        grid_dim=n_seqs,
        block_dim=KMER_BLOCK_SIZE,
    )

    return (out_kmers, out_valid)
