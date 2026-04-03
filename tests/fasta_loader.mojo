"""
FASTA file loader using Python interop.

Provides helpers for loading real genomics fixtures into SequenceBatch
objects for integration testing.
"""
from std.python import Python
from genomics.core.sequence import SequenceBatch


def load_fasta_seq(path: String) raises -> String:
    """Parse a single-sequence FASTA file; return uppercase nucleotide string."""
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "r")
    var content = f.read()
    f.close()
    var lines = content.splitlines()
    # Filter header lines and join in Python to avoid PythonObject→Int conversion
    var parse = Python.evaluate(
        "lambda ls: ''.join(l.strip().upper() for l in ls"
        " if not l.lstrip().startswith('>'))"
    )
    return String(py=parse(lines))


def make_batch_from_seq(seq: String) -> SequenceBatch:
    """Encode a raw sequence string into a single-entry SequenceBatch."""
    var batch = SequenceBatch(capacity=1)
    var buf = List[UInt8](capacity=len(seq))
    for cp in seq.codepoints():
        buf.append(UInt8(Int(cp)))
    batch.add_sequence(Span(buf), len(seq))
    return batch^


def make_batch_substr(seq: String, start: Int, end: Int) -> SequenceBatch:
    """Encode seq[start:end] into a single-entry SequenceBatch."""
    var batch = SequenceBatch(capacity=1)
    var length = end - start
    var buf = List[UInt8](capacity=length)
    var i = 0
    for cp in seq.codepoints():
        if i >= start and i < end:
            buf.append(UInt8(Int(cp)))
        i += 1
        if i >= end:
            break
    batch.add_sequence(Span(buf), length)
    return batch^
