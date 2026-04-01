from genomics.core.dna import Base, ascii_to_2bit, get_base_from_word, NMask, BASES_PER_WORD, BASE_BITS
from genomics.core.sequence import SequenceBatch, get_view

def main() raises:
    print("=== ascii_to_2bit ===")
    print("A(65):", ascii_to_2bit(65), "expected:", Base.A)
    print("C(67):", ascii_to_2bit(67), "expected:", Base.C)
    print("G(71):", ascii_to_2bit(71), "expected:", Base.G)
    print("T(84):", ascii_to_2bit(84), "expected:", Base.T)

    print("=== manual word encode ACGT ===")
    var word: UInt64 = 0
    var bases = [UInt8(65), UInt8(67), UInt8(71), UInt8(84)]
    for i in range(4):
        var b = ascii_to_2bit(bases[i])
        var shift = 62 - i * BASE_BITS
        word |= UInt64(b) << UInt64(shift)
        print("i=", i, "b=", b, "shift=", shift)
    print("word:", word)
    for i in range(4):
        print("  decode[", i, "]=", get_base_from_word(word, i))

    print("=== via SequenceBatch ===")
    var batch = SequenceBatch()
    var buf: List[UInt8] = [65, 67, 71, 84]
    batch.add_sequence(Span(buf), 4)
    print("packed[0]:", batch.packed[0])
    print("nmask[0].data:", batch.n_masks[0].data)

    var view = get_view(batch, 0)
    for i in range(4):
        print("  view.get_base(", i, ")=", view.get_base(i))

    print("=== NMask test ===")
    var nm = NMask.empty()
    nm.set_n(2)
    print("NMask data after set_n(2):", nm.data, "is_n(2):", nm.is_n(2))

    print("=== ACNT batch ===")
    var batch2 = SequenceBatch()
    var buf2: List[UInt8] = [65, 67, 78, 84]
    batch2.add_sequence(Span(buf2), 4)
    print("nmask[0].data:", batch2.n_masks[0].data)
    var view2 = get_view(batch2, 0)
    print("is_n(0):", view2.is_n(0))
    print("is_n(1):", view2.is_n(1))
    print("is_n(2):", view2.is_n(2))
    print("is_n(3):", view2.is_n(3))
