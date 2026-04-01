"""
Unit tests for genomics.core.dna: encoding, complement, reverse complement, NMask.
"""
from std.testing import assert_equal, assert_true, assert_false, TestSuite
from genomics.core.dna import (
    Base, ascii_to_2bit, base_to_ascii,
    complement_word, reverse_2bit_pairs, reverse_complement_word,
    get_base_from_word, NMask, BASES_PER_WORD,
)


def test_ascii_to_2bit() raises:
    assert_equal(ascii_to_2bit(65), Base.A)   # A
    assert_equal(ascii_to_2bit(97), Base.A)   # a
    assert_equal(ascii_to_2bit(67), Base.C)   # C
    assert_equal(ascii_to_2bit(99), Base.C)   # c
    assert_equal(ascii_to_2bit(71), Base.G)   # G
    assert_equal(ascii_to_2bit(103), Base.G)  # g
    assert_equal(ascii_to_2bit(84), Base.T)   # T
    assert_equal(ascii_to_2bit(116), Base.T)  # t
    assert_equal(ascii_to_2bit(78), Base.N)   # N → unknown
    assert_equal(ascii_to_2bit(45), Base.N)   # '-' → unknown


def test_base_roundtrip() raises:
    for b in range(4):
        var ascii = base_to_ascii(UInt8(b))
        assert_equal(ascii_to_2bit(ascii), UInt8(b))


def test_complement_word_all_a() raises:
    # All A (00) → should become all T (11)
    var all_a: UInt64 = 0x0000000000000000
    var comp = complement_word(all_a)
    assert_equal(comp, UInt64(0xFFFFFFFFFFFFFFFF))


def test_complement_word_all_t() raises:
    # All T (11) → should become all A (00)
    var all_t: UInt64 = 0xFFFFFFFFFFFFFFFF
    var comp = complement_word(all_t)
    assert_equal(comp, UInt64(0x0000000000000000))


def test_complement_word_acgt() raises:
    # Pack ACGT into top 8 bits: A=00, C=01, G=10, T=11 → 0b00011011 << 56
    var word = UInt64(0b00011011) << 56
    var comp = complement_word(word)
    # Complement: T=11, G=10, C=01, A=00 → 0b11100100 << 56
    var expected = UInt64(0b11100100) << 56
    # The complement of the full 64-bit word, not just top 8 bits
    # ~word = flip all 64 bits
    assert_equal(comp, ~word)
    # Also verify the base encoding is correct
    assert_equal(get_base_from_word(comp, 0), Base.T)
    assert_equal(get_base_from_word(comp, 1), Base.G)
    assert_equal(get_base_from_word(comp, 2), Base.C)
    assert_equal(get_base_from_word(comp, 3), Base.A)


def test_reverse_2bit_pairs_single_base() raises:
    # Place base T (11) at position 0 (leftmost), rest A (00)
    var word = UInt64(0b11) << 62
    var rev = reverse_2bit_pairs(word)
    # After reversal, T should be at position 31 (rightmost)
    assert_equal(get_base_from_word(rev, 31), Base.T)
    assert_equal(get_base_from_word(rev, 0), Base.A)


def test_reverse_complement_word() raises:
    # ACGT packed in positions 0-3, rest A
    var word: UInt64 = 0
    word |= UInt64(Base.A) << 62   # pos 0
    word |= UInt64(Base.C) << 60   # pos 1
    word |= UInt64(Base.G) << 58   # pos 2
    word |= UInt64(Base.T) << 56   # pos 3

    var rc = reverse_complement_word(word)
    # Rev comp of ACGT = ACGT (palindrome)
    assert_equal(get_base_from_word(rc, 28), Base.A)
    assert_equal(get_base_from_word(rc, 29), Base.C)
    assert_equal(get_base_from_word(rc, 30), Base.G)
    assert_equal(get_base_from_word(rc, 31), Base.T)


def test_get_base_from_word() raises:
    # Pack ACGT at positions 0-3
    var word: UInt64 = 0
    word |= UInt64(Base.A) << 62
    word |= UInt64(Base.C) << 60
    word |= UInt64(Base.G) << 58
    word |= UInt64(Base.T) << 56
    assert_equal(get_base_from_word(word, 0), Base.A)
    assert_equal(get_base_from_word(word, 1), Base.C)
    assert_equal(get_base_from_word(word, 2), Base.G)
    assert_equal(get_base_from_word(word, 3), Base.T)
    assert_equal(get_base_from_word(word, 4), Base.A)  # zeros = A


def test_nmask_basic() raises:
    var nm = NMask.empty()
    assert_false(nm.has_any_n())

    nm.set_n(0)
    assert_true(nm.is_n(0))
    assert_false(nm.is_n(1))
    assert_true(nm.has_any_n())

    nm.set_n(31)
    assert_true(nm.is_n(31))
    assert_true(nm.any_n_in_range(0, 32))
    assert_false(nm.any_n_in_range(1, 31))


def test_nmask_range_check() raises:
    var nm = NMask.empty()
    nm.set_n(15)
    assert_true(nm.any_n_in_range(10, 20))
    assert_false(nm.any_n_in_range(0, 15))
    assert_false(nm.any_n_in_range(16, 32))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
