"""
Core DNA encoding: 2-bit packed representation.

Encoding:  A=0b00  C=0b01  G=0b10  T=0b11
One UInt64 holds 32 bases (most-significant pair = leftmost base).
Reverse complement: bitwise NOT of all bits, then bit-reverse the 2-bit pair order.
N/ambiguous bases are tracked in a parallel NMask bitmask (not stored inline).
"""
from std.math import min, max


comptime BASE_BITS: Int = 2
comptime BASES_PER_WORD: Int = 32  # bases packed into one UInt64


struct Base:
    """2-bit encoding constants for the four canonical DNA bases."""
    comptime A: UInt8 = 0b00
    comptime C: UInt8 = 0b01
    comptime G: UInt8 = 0b10
    comptime T: UInt8 = 0b11
    comptime N: UInt8 = 0xFF  # sentinel — must be tracked in NMask, not stored inline


@always_inline
def ascii_to_2bit(c: UInt8) -> UInt8:
    """Convert an ASCII nucleotide character to its 2-bit encoding.

    Handles both upper- and lower-case A/C/G/T. Returns Base.N for any
    other character (N, gaps, IUPAC ambiguity codes, etc.).
    """
    if c == 65 or c == 97:    # A / a
        return Base.A
    if c == 67 or c == 99:    # C / c
        return Base.C
    if c == 71 or c == 103:   # G / g
        return Base.G
    if c == 84 or c == 116:   # T / t
        return Base.T
    return Base.N


@always_inline
def base_to_ascii(b: UInt8) -> UInt8:
    """Convert a 2-bit encoded base back to its uppercase ASCII character."""
    if b == Base.A:
        return 65   # A
    if b == Base.C:
        return 67   # C
    if b == Base.G:
        return 71   # G
    return 84       # T  (also catches invalid values gracefully)


# ===----------------------------------------------------------------------=== #
# Word-level complement and reverse complement
# ===----------------------------------------------------------------------=== #

@always_inline
def complement_word(word: UInt64) -> UInt64:
    """Bitwise complement of all 32 packed bases.

    XOR / NOT flips every 2-bit pair:
        A (00) <-> T (11)
        C (01) <-> G (10)
    """
    return ~word


@always_inline
def reverse_2bit_pairs(word: UInt64) -> UInt64:
    """Reverse the order of all 32 two-bit base pairs within a UInt64.

    Uses a 5-round swap-halves bit-reversal down to 2-bit granularity.
    """
    var w = word
    # Swap 32-bit halves
    w = (w >> 32) | (w << 32)
    # Swap 16-bit groups within each 32-bit half
    w = ((w >> 16) & UInt64(0x0000FFFF0000FFFF)) | ((w & UInt64(0x0000FFFF0000FFFF)) << 16)
    # Swap 8-bit groups
    w = ((w >> 8) & UInt64(0x00FF00FF00FF00FF)) | ((w & UInt64(0x00FF00FF00FF00FF)) << 8)
    # Swap 4-bit groups
    w = ((w >> 4) & UInt64(0x0F0F0F0F0F0F0F0F)) | ((w & UInt64(0x0F0F0F0F0F0F0F0F)) << 4)
    # Swap 2-bit pairs — base granularity
    w = ((w >> 2) & UInt64(0x3333333333333333)) | ((w & UInt64(0x3333333333333333)) << 2)
    return w


@always_inline
def reverse_complement_word(word: UInt64) -> UInt64:
    """Reverse complement all 32 packed bases in a UInt64."""
    return reverse_2bit_pairs(complement_word(word))


@always_inline
def get_base_from_word(word: UInt64, pos: Int) -> UInt8:
    """Extract a single base at position pos (0 = leftmost) from a packed word."""
    var shift = 62 - pos * BASE_BITS
    return UInt8((word >> UInt64(shift)) & 3)


# ===----------------------------------------------------------------------=== #
# NMask — parallel ambiguous-base bitmask
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct NMask(Copyable, Movable, ImplicitlyCopyable):
    """Bitmask tracking N (ambiguous) bases — one bit per position within a word.

    One NMask covers up to 32 positions (matching one packed UInt64 word).
    Bit i = 1 means position i within that word is an ambiguous base.
    """
    var data: UInt64

    @always_inline
    def is_n(self, pos: Int) -> Bool:
        return (self.data >> UInt64(pos)) & 1 == 1

    @always_inline
    def set_n(mut self, pos: Int):
        self.data |= UInt64(1) << UInt64(pos)

    @always_inline
    def any_n_in_range(self, start: Int, end: Int) -> Bool:
        """Return True if any N exists in bit positions [start, end)."""
        var bits = end - start
        var mask: UInt64
        if bits >= 64:
            mask = ~UInt64(0)
        else:
            mask = (UInt64(1) << UInt64(bits)) - 1
        return ((self.data >> UInt64(start)) & mask) != 0

    @always_inline
    def has_any_n(self) -> Bool:
        return self.data != 0

    @staticmethod
    def empty() -> NMask:
        return NMask(0)
