# Research Direction & Showcase Roadmap — Computational Genomics in Mojo

## Context

The repo today is a well-built **bag of primitives**: 2-bit DNA packing, `Kmer[k]`, SoA `SequenceBatch`, banded SW/NW, gap-affine WFA, sliding-window stats — each with a CPU reference and (mostly) a matching GPU kernel. It's solid engineering, but it has no *thesis*. To turn it into a "cooler research project" that (a) reads as a coherent story and (b) can be shown off through an interactive site, it needs a spine: one flagship capability that the primitives feed into, plus one or two recent, name-recognizable algorithms that signal "this person is current" to NVIDIA, Modular, and biotech audiences.

This document is a **research roadmap / idea menu**, not a code-change spec. GPU hardware validation and kernel micro-optimization are intentionally left thin — the user already has a device-handoff plan for that.

Note: the audience/direction clarifying questions were posed but not answered (user away). This plan therefore **recommends** a primary spine (sketching & comparison) and lays the alternatives beside it so the direction can be re-steered on return.

---

## The core idea: give the primitives a destination

Almost every primitive already in the repo is a building block of **genome sketching and comparison** — the branch of genomics behind metagenomic classification, average-nucleotide-identity (ANI) estimation, and "what is this sequence?" search. This is the natural spine because it reuses what exists, is embarrassingly parallel (ideal for the GPU story), and produces *visually stunning* output (heatmaps, dotplots) for an interactive demo.

**Recommended spine — "A GPU-native genome sketching & comparison engine in Mojo":**

```
sequences ─► minimizers ─► FracMinHash sketch ─► all-vs-all ANI / containment ─► heatmap + dotplot
              (new)          (new)                 (new)                          (interactive site)
   ▲ reuses Kmer[k], canonical(), hash(), SequenceBatch, NMask
```

This is a clean 3-algorithm arc on top of the current k-mer code, and each stage is a recognizable, citable modern method.

---

## Modern algorithms worth implementing (ranked by impact ÷ effort)

Each entry notes **why it's impressive**, **who it lands with**, and **fit with the existing code**.

### Tier 1 — high leverage, builds directly on current code

1. **Minimizers** (Roberts 2004; Schleimer 2003) — the (w,k) minimizer: for each window of `w` consecutive k-mers, keep the one with the smallest hash. Foundation of minimap2, Kraken2, etc.
   - *Impressive because:* it's the indexing primitive of essentially every modern aligner/classifier; a fast SIMD/GPU minimizer sketch is genuinely useful.
   - *Fit:* trivial extension of `Kmer[k].roll()` + `hash()` you already have. Sliding-window-minimum is a nice bit-parallel/monotonic-deque problem.
   - *Audience:* all three.

2. **FracMinHash / bottom-sketch MinHash** (Ondov *Mash* 2016; Irber *sourmash* / FracMinHash 2022) — keep all hashes below a threshold (FracMinHash) or the bottom-s hashes (MinHash) to build a genome sketch; Jaccard/containment between sketches estimates ANI in O(sketch size).
   - *Impressive because:* it's how sourmash/Mash do million-genome comparison; FracMinHash's containment estimate is current best practice.
   - *Fit:* a filter + sort over the k-mer hashes you already produce.
   - *Audience:* biotech (real tool), Modular (elegant), NVIDIA (all-vs-all is a GPU matrix problem).

3. **skani-style ANI** (Shaw & Yu, *Nature Methods* 2023) — the current state of the art for fast, accurate ANI from sketches, robust for incomplete/MAG genomes.
   - *Impressive because:* it's a 2023 *Nature Methods* method; matching its numbers on real genomes is strong scientific credibility.
   - *Fit:* the natural "payoff" stage that consumes minimizers + FracMinHash.

### Tier 2 — standout novelty, moderate effort

4. **Strobemers** (Sahlin, *Genome Research* 2021) and **Syncmers** (Edgar, *PeerJ* 2021) — newer alternatives to minimizers that are more robust to indels / more conserved across mutations.
   - *Impressive because:* few implementations exist, especially none in Mojo; signals you read current literature, not just textbooks. A benchmark of k-mer vs minimizer vs syncmer vs strobemer seed conservation is a compelling, publishable-flavored figure.
   - *Fit:* same hashing/rolling machinery.

5. **BiWFA — bidirectional WFA** (Marco-Sola *et al.*, *Bioinformatics* 2023) — extends your existing WFA to O(s) memory *with* full traceback, closing your current "CIGAR traceback stubbed" gap in the modern way rather than the classic-DP way.
   - *Impressive because:* it's the current best gap-affine exact aligner and directly upgrades code you already wrote.
   - *Fit:* you have `wfa_affine_cpu`; BiWFA is the meet-in-the-middle version. High continuity.
   - *Audience:* NVIDIA (they care about WFA on GPU), Modular.

### Tier 3 — ambitious spines (pick only if pivoting away from sketching)

6. **Minimizer seed–chain–extend read mapper** (the minimap2 pipeline; Li 2018) — minimizer index → anchor **chaining** (a 1-D DP over anchors, the interesting new algorithm) → WFA/BiWFA extension. NVIDIA actively funds GPU minimap2 (mm2-gb, 2024).
   - *Bigger scope*; strongest "systems" signal to NVIDIA. Consumes Tier-1 + Tier-2 work as sub-steps, so Tier 1 is a prerequisite either way.

7. **Genomic language-model inference via MAX** — run **HyenaDNA** (Nguyen 2023), **Caduceus** (Schiff 2024, Mamba/SSM-based), or **Evo** (Nguyen 2024) through Modular's MAX engine for a task like promoter/enhancer classification or variant-effect prediction. Your 2-bit encoder becomes the tokenizer.
   - *Most on-brand for Modular* (MAX is their inference product) and for NVIDIA-BioNeMo; ML-heavy, a different skillset. Best as an *optional headline* on top of the classical core, not the whole project.

---

## Interactive site — architecture options

Mojo doesn't have a mature WASM path, so the pattern is **Mojo compute backend + web frontend**:

- **Backend:** expose the library as a **Python extension module** (you already have the `mojo-python-interop` skill and FASTA-loader interop), wrap with a thin **FastAPI** service. Endpoints: `/sketch`, `/compare`, `/dotplot`, `/align`.
- **Frontend:** static site (React/Svelte or even Observable) calling the API. High-value, low-effort visualizations:
  - **Minimizer/anchor dotplot** of two pasted genomes (à la D-GENIES/minimap2) — visually striking, cheap to compute, and the single best "wow" artifact.
  - **ANI / MinHash similarity heatmap** across an uploaded set of genomes.
  - **Live sliding-window GC / entropy / complexity track** (a mini genome browser) — reuses `sliding_window.mojo` as-is.
  - **k-mer spectrum** histogram as you type/paste.
- **Showpiece framing:** a "paste two viral genomes, watch them align and sketch in real time" page using the PhiX174 / SARS-CoV-2 / E. coli fixtures already in `tests/fixtures/` as one-click presets.

---

## Suggested phased roadmap (spine = sketching engine)

- **Phase 0 — Narrative & polish:** README (done) frames the project as a sketching/comparison engine; pick the thesis sentence. Close the WFA traceback gap via **BiWFA** (Tier-2 #5) so alignment is demo-ready.
- **Phase 1 — Minimizers** (`genomics/cpu/minimizer.mojo` + GPU kernel): (w,k) minimizer sketch reusing `Kmer[k]`. Validate count/positions against a Python reference on the fixtures.
- **Phase 2 — FracMinHash sketch + Jaccard/containment ANI:** sketch struct, set-similarity, ANI estimate. Validate against `sourmash`/`Mash` numbers on the real genomes.
- **Phase 3 — Novelty benchmark:** add **syncmers** (and optionally strobemers); produce a seed-conservation-under-mutation figure comparing k-mer vs minimizer vs syncmer. This is the "research paper" moment.
- **Phase 4 — Interactive site:** Python-extension + FastAPI backend, dotplot + heatmap frontend, fixture presets.
- **Phase 5 (optional headline):** MAX-powered genomic-LM classifier (HyenaDNA/Caduceus) sharing the 2-bit tokenizer — the Modular showcase.

GPU validation and kernel tuning fold into whichever phase touches a kernel, handled during your separate device handoff.

---

## Audience framing (so wins are legible)

- **NVIDIA:** lead with GPU all-vs-all sketch comparison and BiWFA/chaining kernels; benchmark against Parabricks/GenomeWorks/mm2-gb where possible.
- **Modular:** lead with performance-per-line Mojo idioms and (Phase 5) a real MAX inference integration — "Mojo doing genomics end-to-end."
- **Biotech/bioinformatics:** lead with correctness vs. reference tools (sourmash, skani, minimap2) on real genomes and a usable interactive tool.

---

## Verification / how we'd validate each track

- **Correctness:** extend the `tests/` + `tests/fixtures/` pattern — every new algorithm gets a Python-reference cross-check (`mojo-python-interop`) on PhiX174/SARS-CoV-2/E. coli, e.g. minimizer sets vs. a NumPy sliding-min, FracMinHash Jaccard vs. `sourmash`, ANI vs. `skani`/`fastANI`.
- **Performance:** add `benchmarks/bench_sketch.mojo` following the `Bench`/`record.sh` pattern; track throughput regressions in the existing JSONL scheme.
- **Demo:** stand up the FastAPI backend locally, hit `/dotplot` and `/compare` with the fixtures, confirm the frontend renders the dotplot/heatmap.

---

## Open questions for the user (re-ask on return)

1. **Spine choice** — endorse the sketching/comparison engine, or pivot to the read-mapper or genomic-LM spine?
2. **ML/MAX appetite** — is Phase 5 (a MAX model showcase) in scope, or stay classical?
3. **Primary audience** — NVIDIA / Modular / biotech (or all) — to weight benchmarking vs. elegance vs. scientific validation.
4. **Interactive site ambition** — portfolio showpiece (a few curated demos) vs. a genuinely usable tool?
