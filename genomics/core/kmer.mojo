"""
K-mer representation with compile-time k parameter.

Kmer[k] stores a k-mer as a left-aligned 2-bit packed UInt64.
Supports rolling (sliding), canonical form (strand-agnostic), and hashing.
k must satisfy 1 <= k <= 32 (fits in one UInt64).
"""
from genomics.core.dna import Base, reverse_complement_word, BASES_PER_WORD


struct Kmer[k: Int](Copyable, Movable, Equatable, ImplicitlyCopyable):
    """A k-mer stored as left-aligned 2-bit packed bases in a UInt64.

    The k bases occupy the top 2*k bits.  The lower (64 - 2k) bits are zero.

    Example for k=4, sequence ACGT:
        bits = 0b00_01_10_11_0000...0000 (left-aligned, 56 trailing zeros)
    """
    # Mask covering only the top 2*k bits (the k-mer itself)
    comptime MASK: UInt64 = ~UInt64(0) << UInt64(64 - 2 * Self.k)

    var bits: UInt64

    def __init__(out self, bits: UInt64 = 0):
        self.bits = bits & Self.MASK

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.bits == other.bits

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self.bits != other.bits

    @always_inline
    def roll(self, next_base: UInt8) -> Kmer[Self.k]:
        """Slide the k-mer one base to the right, appending next_base.

        Shifts left by 2 bits (drops leftmost base) and ORs in the new base
        at the rightmost 2-bit slot of the k-mer window.
        """
        # After <<2, the bottom 2 bits of the 2k-wide window are zero.
        # The new base belongs at bit position (64 - 2k), left-shifted by that amount.
        return Kmer[Self.k]((self.bits << 2) | (UInt64(next_base) << UInt64(64 - 2 * Self.k)))

    @always_inline
    def rev_comp(self) -> Kmer[Self.k]:
        """Return the reverse complement of this k-mer."""
        # reverse_complement_word operates on 32 bases filling a full UInt64.
        # Our k-mer is left-aligned in 2*k bits. Complement and bit-reverse,
        # then re-align left by shifting.
        var full = self.bits | (self.bits >> UInt64(2 * Self.k))  # fill lower bits (harmless)
        var rc_full = reverse_complement_word(full)
        # After full 64-bit reversal, our k bases are now in the rightmost 2k bits.
        # Shift left to re-align.
        return Kmer[Self.k](rc_full << UInt64(64 - 2 * Self.k))

    @always_inline
    def canonical(self) -> Kmer[Self.k]:
        """Return min(self, rev_comp(self)) for strand-agnostic indexing."""
        var rc = self.rev_comp()
        if self.bits <= rc.bits:
            return self
        return rc

    @always_inline
    def hash(self) -> UInt64:
        """MurmurHash3-style 64-bit finalizer for the k-mer bits."""
        var h = self.bits
        h ^= h >> 33
        h *= UInt64(0xFF51AFD7ED558CCD)
        h ^= h >> 33
        h *= UInt64(0xC4CEB9FE1A85EC53)
        h ^= h >> 33
        return h

    @always_inline
    def to_string(self) -> String:
        """Decode the k-mer back to its ASCII sequence."""
        var s = String("")
        for i in range(Self.k):
            var shift = 62 - i * 2
            var b = UInt8((self.bits >> UInt64(shift)) & 3)
            if b == 0:
                s += "A"
            elif b == 1:
                s += "C"
            elif b == 2:
                s += "G"
            else:
                s += "T"
        return s
