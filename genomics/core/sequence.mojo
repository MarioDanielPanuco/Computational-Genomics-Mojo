"""
SequenceBatch: structure-of-arrays container for 2-bit packed DNA sequences.

Layout:
  packed[offsets[i] .. offsets[i] + words_for(i))  — packed UInt64 words
  n_masks[offsets[i] .. offsets[i] + words_for(i)) — one NMask per packed word
  lengths[i]  — sequence length in bases
  count       — number of sequences in this batch

This SoA layout enables coalesced GPU memory access: when one GPU thread-block
handles one sequence, it reads a contiguous run of UInt64 words from `packed`.

SequenceView owns a shallow copy of the slice for its sequence.
For GPU paths, pass the raw List buffers (via .unsafe_ptr()) directly to kernels.
"""
from std.math import ceildiv, min, max
from genomics.core.dna import (
    NMask, Base, ascii_to_2bit, get_base_from_word,
    BASES_PER_WORD, BASE_BITS,
)


struct SequenceBatch(Movable):
    """SoA batch of DNA sequences stored as left-aligned 2-bit packed UInt64 words."""
    var packed: List[UInt64]
    var n_masks: List[NMask]
    var offsets: List[Int]
    var lengths: List[Int]
    var count: Int

    def __init__(out self, capacity: Int = 256):
        self.packed = List[UInt64](capacity=capacity * 8)
        self.n_masks = List[NMask](capacity=capacity * 8)
        self.offsets = List[Int](capacity=capacity)
        self.lengths = List[Int](capacity=capacity)
        self.count = 0

    def add_sequence(mut self, src: Span[UInt8, _], length: Int):
        """Encode and append an ASCII DNA sequence.

        Non-ACGT characters are stored as A in the packed word and flagged
        in the corresponding NMask.
        """
        var word_start = len(self.packed)
        var n_words = ceildiv(length, BASES_PER_WORD)

        for w in range(n_words):
            var base_offset = w * BASES_PER_WORD
            var n = min(length - base_offset, BASES_PER_WORD)
            var nmask = NMask.empty()
            var word: UInt64 = 0

            for i in range(n):
                var b = ascii_to_2bit(src[base_offset + i])
                if b == Base.N:
                    nmask.set_n(i)
                    b = 0
                var shift = 62 - i * BASE_BITS
                word |= UInt64(b) << UInt64(shift)

            self.packed.append(word)
            self.n_masks.append(nmask)

        self.offsets.append(word_start)
        self.lengths.append(length)
        self.count += 1

    def total_bases(self) -> Int:
        var total = 0
        for i in range(self.count):
            total += self.lengths[i]
        return total

    def words_for(self, idx: Int) -> Int:
        return ceildiv(self.lengths[idx], BASES_PER_WORD)


struct SequenceView(Movable):
    """Owned slice of one sequence's packed words and N-masks.

    Created via get_view(batch, idx) — copies the slice on construction.
    Preferred for CPU code paths.  GPU kernels use SequenceBatch buffers directly.
    """
    var packed: List[UInt64]
    var n_masks: List[NMask]
    var length: Int
    var word_count: Int

    def __init__(out self, var packed: List[UInt64], var n_masks: List[NMask],
                 length: Int, word_count: Int):
        self.packed = packed^
        self.n_masks = n_masks^
        self.length = length
        self.word_count = word_count

    @always_inline
    def get_base(self, pos: Int) -> UInt8:
        """Extract base at position pos (0-indexed)."""
        var word_idx = pos // BASES_PER_WORD
        var bit_pos = pos % BASES_PER_WORD
        return get_base_from_word(self.packed[word_idx], bit_pos)

    @always_inline
    def is_n(self, pos: Int) -> Bool:
        """Return True if position pos is an ambiguous (N) base."""
        var word_idx = pos // BASES_PER_WORD
        var bit_pos = pos % BASES_PER_WORD
        return self.n_masks[word_idx].is_n(bit_pos)

    @always_inline
    def has_n_in_window(self, start: Int, end: Int) -> Bool:
        """Return True if any N exists in the half-open window [start, end)."""
        var first_word = start // BASES_PER_WORD
        var last_word = (end - 1) // BASES_PER_WORD
        for w in range(first_word, last_word + 1):
            var lo = max(0, start - w * BASES_PER_WORD)
            var hi = min(BASES_PER_WORD, end - w * BASES_PER_WORD)
            if self.n_masks[w].any_n_in_range(lo, hi):
                return True
        return False


def get_view(batch: SequenceBatch, idx: Int) -> SequenceView:
    """Copy sequence idx from a SequenceBatch into a standalone SequenceView."""
    var off = batch.offsets[idx]
    var n_words = ceildiv(batch.lengths[idx], BASES_PER_WORD)
    var packed = List[UInt64](capacity=n_words)
    var n_masks = List[NMask](capacity=n_words)
    for i in range(n_words):
        packed.append(batch.packed[off + i])
        n_masks.append(batch.n_masks[off + i])
    return SequenceView(
        packed=packed^,
        n_masks=n_masks^,
        length=batch.lengths[idx],
        word_count=n_words,
    )
