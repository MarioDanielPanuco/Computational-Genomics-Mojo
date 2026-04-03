"""
Integration tests using real public genomics data.

Fixtures (committed to tests/fixtures/):
  phix174.fasta   — NC_001422.1  PhiX174 bacteriophage  5,386 bp  GC 44.7%
  sars_cov2.fasta — NC_045512.2  SARS-CoV-2            29,903 bp  GC 37.97%
  ecoli_16s.fasta — V00613.1     E. coli 16S rRNA       1,542 bp  GC 54.7%

Tests validate that library functions produce biologically correct results
at realistic sequence lengths, not just hand-crafted 4-33 bp examples.
"""
from std.testing import assert_equal, assert_true, TestSuite
from genomics.core.sequence import get_view
from genomics.cpu.kmer_cpu import extract_kmers, kmer_frequencies
from genomics.cpu.align_cpu import smith_waterman_banded, default_config
from genomics.cpu.sliding_window import (
    gc_content_sliding, sequence_entropy_sliding, complexity_score,
)
from tests.fasta_loader import load_fasta_seq, make_batch_from_seq, make_batch_substr


# ===----------------------------------------------------------------------=== #
# K-mer tests
# ===----------------------------------------------------------------------=== #

def test_phix_kmer_count() raises:
    """PhiX174 should yield exactly 5366 valid 21-mers (no N bases in genome)."""
    var seq = load_fasta_seq("tests/fixtures/phix174.fasta")
    var batch = make_batch_from_seq(seq)
    var view = get_view(batch, 0)
    comptime K = 21
    var n_pos = view.length - K + 1
    var kmers = List[UInt64](capacity=n_pos)
    var valid = List[Bool](capacity=n_pos)
    for _ in range(n_pos):
        kmers.append(0)
        valid.append(False)
    var count = extract_kmers[K](view, kmers.unsafe_ptr(), valid.unsafe_ptr())
    assert_equal(count, 5366)
    var valid_count = 0
    for i in range(count):
        if valid[i]:
            valid_count += 1
    assert_equal(valid_count, 5366)


def test_phix_kmer_frequency_distribution() raises:
    """PhiX174 21-mer frequency table: total==5366, most k-mers are unique."""
    var seq = load_fasta_seq("tests/fixtures/phix174.fasta")
    var batch = make_batch_from_seq(seq)
    # table_bits=20 → 1M slots, low collision probability for 5366 k-mers
    var counts = kmer_frequencies[21](batch, table_bits=20)
    var total = 0
    var n_nonzero = 0
    var n_singleton = 0
    for i in range(len(counts)):
        if counts[i] > 0:
            total += counts[i]
            n_nonzero += 1
            if counts[i] == 1:
                n_singleton += 1
    assert_equal(total, 5366)
    # Most 21-mers in PhiX174 are unique (appear exactly once)
    assert_true(n_singleton > n_nonzero // 2)


# ===----------------------------------------------------------------------=== #
# GC content tests
# ===----------------------------------------------------------------------=== #

def test_phix_gc_content() raises:
    """PhiX174 GC fraction averaged over 100-bp windows should be ~44.7%."""
    var seq = load_fasta_seq("tests/fixtures/phix174.fasta")
    var batch = make_batch_from_seq(seq)
    var view = get_view(batch, 0)
    comptime WIN = 100
    var n_out = view.length - WIN + 1
    var buf = List[Float32](capacity=n_out)
    for _ in range(n_out):
        buf.append(Float32(0.0))
    gc_content_sliding[WIN](view, buf.unsafe_ptr())
    var total = Float32(0.0)
    for i in range(n_out):
        total += buf[i]
    var avg = total / Float32(n_out)
    assert_true(avg >= Float32(0.43) and avg <= Float32(0.47))


def test_sars_gc_content() raises:
    """SARS-CoV-2 GC fraction averaged over 1000-bp windows should be ~37.97%."""
    var seq = load_fasta_seq("tests/fixtures/sars_cov2.fasta")
    var batch = make_batch_from_seq(seq)
    var view = get_view(batch, 0)
    comptime WIN = 1000
    var n_out = view.length - WIN + 1
    var buf = List[Float32](capacity=n_out)
    for _ in range(n_out):
        buf.append(Float32(0.0))
    gc_content_sliding[WIN](view, buf.unsafe_ptr())
    var total = Float32(0.0)
    for i in range(n_out):
        total += buf[i]
    var avg = total / Float32(n_out)
    assert_true(avg >= Float32(0.36) and avg <= Float32(0.40))


def test_ecoli_16s_gc_content() raises:
    """E. coli 16S rRNA region (V00613.1, 4957 bp) GC fraction should be ~49.9%."""
    var seq = load_fasta_seq("tests/fixtures/ecoli_16s.fasta")
    var batch = make_batch_from_seq(seq)
    var view = get_view(batch, 0)
    comptime WIN = 100
    var n_out = view.length - WIN + 1
    var buf = List[Float32](capacity=n_out)
    for _ in range(n_out):
        buf.append(Float32(0.0))
    gc_content_sliding[WIN](view, buf.unsafe_ptr())
    var total = Float32(0.0)
    for i in range(n_out):
        total += buf[i]
    var avg = total / Float32(n_out)
    assert_true(avg >= Float32(0.47) and avg <= Float32(0.53))


# ===----------------------------------------------------------------------=== #
# Alignment tests
# ===----------------------------------------------------------------------=== #

def test_sw_self_alignment() raises:
    """Aligning a 100-bp PhiX region to itself should yield score = 100 * match_score."""
    var seq = load_fasta_seq("tests/fixtures/phix174.fasta")
    var batch = make_batch_substr(seq, 0, 100)
    var view1 = get_view(batch, 0)
    var view2 = get_view(batch, 0)
    var cfg = default_config()
    var result = smith_waterman_banded(view1, view2, cfg)
    assert_equal(result.score, 100 * cfg.match_score)


def test_sw_cross_genome_discrimination() raises:
    """PhiX vs SARS 100-bp window should score far lower than self-alignment."""
    var phix_seq = load_fasta_seq("tests/fixtures/phix174.fasta")
    var sars_seq = load_fasta_seq("tests/fixtures/sars_cov2.fasta")

    var phix_batch = make_batch_substr(phix_seq, 0, 100)
    var sars_batch = make_batch_substr(sars_seq, 0, 100)

    var cfg = default_config()

    # Self-alignment baseline
    var self_view1 = get_view(phix_batch, 0)
    var self_view2 = get_view(phix_batch, 0)
    var self_result = smith_waterman_banded(self_view1, self_view2, cfg)

    # Cross-genome alignment
    var phix_view = get_view(phix_batch, 0)
    var sars_view = get_view(sars_batch, 0)
    var cross_result = smith_waterman_banded(phix_view, sars_view, cfg)

    assert_true(cross_result.score < self_result.score // 4)


# ===----------------------------------------------------------------------=== #
# Sliding-window tests
# ===----------------------------------------------------------------------=== #

def test_entropy_real_sequence() raises:
    """PhiX174[500:650] entropy over the full 150-bp window should exceed 1.5 bits."""
    var seq = load_fasta_seq("tests/fixtures/phix174.fasta")
    var batch = make_batch_substr(seq, 500, 650)
    var view = get_view(batch, 0)
    comptime WIN = 150
    var n_out = view.length - WIN + 1  # == 1
    var buf = List[Float32](capacity=n_out)
    for _ in range(n_out):
        buf.append(Float32(0.0))
    sequence_entropy_sliding[WIN](view, buf.unsafe_ptr())
    assert_true(buf[0] > Float32(1.5))


def test_complexity_real_sequence() raises:
    """PhiX174[200:300] should have high linguistic complexity (> 0.8) for k=2."""
    var seq = load_fasta_seq("tests/fixtures/phix174.fasta")
    var batch = make_batch_substr(seq, 200, 300)
    var view = get_view(batch, 0)
    var score = complexity_score[2](view)
    assert_true(score > Float32(0.8))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
