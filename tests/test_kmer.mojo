"""
Unit tests for k-mer representation, rolling, canonical form, and CPU extraction.
"""
from std.testing import assert_equal, assert_true, assert_false, TestSuite
from genomics.core.kmer import Kmer
from genomics.core.dna import Base
from genomics.core.sequence import SequenceBatch, get_view
from genomics.cpu.kmer_cpu import extract_kmers


def test_kmer_roll_basic() raises:
    # Build k=4 k-mer from ACGT manually
    var k = Kmer[4]()
    k = k.roll(Base.A)
    k = k.roll(Base.C)
    k = k.roll(Base.G)
    k = k.roll(Base.T)
    # to_string should give ACGT
    assert_equal(k.to_string(), "ACGT")


def test_kmer_roll_sliding() raises:
    # Seed ACGT, then roll in A → should give CGTA
    var k = Kmer[4]()
    k = k.roll(Base.A)
    k = k.roll(Base.C)
    k = k.roll(Base.G)
    k = k.roll(Base.T)
    k = k.roll(Base.A)
    assert_equal(k.to_string(), "CGTA")


def test_kmer_canonical_palindrome() raises:
    # ATAT and its rev comp ATAT — canonical should equal itself
    var k = Kmer[4]()
    k = k.roll(Base.A)
    k = k.roll(Base.T)
    k = k.roll(Base.A)
    k = k.roll(Base.T)
    var c = k.canonical()
    assert_equal(c.to_string(), k.to_string())


def test_kmer_canonical_strand_agnostic() raises:
    # k=4: AAAA and its rev comp TTTT — canonical should be AAAA (smaller)
    var k = Kmer[4]()
    for _ in range(4):
        k = k.roll(Base.A)
    var rc_k = Kmer[4]()
    for _ in range(4):
        rc_k = rc_k.roll(Base.T)

    var c_k = k.canonical()
    var c_rc = rc_k.canonical()
    # Both should map to the same canonical form
    assert_equal(c_k.bits, c_rc.bits)


def test_kmer_hash_deterministic() raises:
    var k1 = Kmer[4]()
    var k2 = Kmer[4]()
    for _ in range(4):
        k1 = k1.roll(Base.A)
        k2 = k2.roll(Base.A)
    assert_equal(k1.hash(), k2.hash())


def test_kmer_hash_distinct() raises:
    var k1 = Kmer[4]()
    var k2 = Kmer[4]()
    for _ in range(4):
        k1 = k1.roll(Base.A)
    for _ in range(4):
        k2 = k2.roll(Base.T)
    assert_true(k1.hash() != k2.hash())


def test_extract_kmers_count() raises:
    # Sequence ACGTACGT (length 8), k=4 → 5 k-mers
    var batch = SequenceBatch()
    var seq: List[UInt8] = [65, 67, 71, 84, 65, 67, 71, 84]  # ACGTACGT
    batch.add_sequence(Span(seq), 8)

    var view = get_view(batch, 0)
    var n_kmers = view.length - 4 + 1
    var kbuf = List[UInt64](capacity=n_kmers)
    var vbuf = List[Bool](capacity=n_kmers)
    for _ in range(n_kmers):
        kbuf.append(0)
        vbuf.append(False)

    var count = extract_kmers[4](view, kbuf.unsafe_ptr(), vbuf.unsafe_ptr())
    assert_equal(count, 5)
    for i in range(5):
        assert_true(vbuf[i])  # no N bases, all valid


def test_extract_kmers_with_n() raises:
    # ACNTACGT — N at position 2; k=4 windows [0,4) and [1,5) contain N
    var batch = SequenceBatch()
    var seq: List[UInt8] = [65, 67, 78, 84, 65, 67, 71, 84]  # ACNTACGT
    batch.add_sequence(Span(seq), 8)

    var view = get_view(batch, 0)
    var n_kmers = 5
    var kbuf = List[UInt64](capacity=n_kmers)
    var vbuf = List[Bool](capacity=n_kmers)
    for _ in range(n_kmers):
        kbuf.append(0)
        vbuf.append(False)

    _ = extract_kmers[4](view, kbuf.unsafe_ptr(), vbuf.unsafe_ptr())
    # Windows 0-3 and 1-4 include position 2 (N) → invalid
    assert_false(vbuf[0])
    assert_false(vbuf[1])
    # Window 4-7: ACGT → valid
    assert_true(vbuf[4])


def test_kmer_k21_roll() raises:
    # Verify k=21 k-mer can be built and produces a string of length 21
    var k = Kmer[21]()
    var bases = [Base.A, Base.C, Base.G, Base.T]
    for i in range(21):
        k = k.roll(bases[i % 4])
    var s = k.to_string()
    assert_equal(len(s), 21)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
