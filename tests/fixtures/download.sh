#!/usr/bin/env bash
# Re-download all genomics fixture files from NCBI Entrez.
# Run from the tests/fixtures/ directory: bash download.sh
set -euo pipefail
EFETCH="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=fasta&retmode=text&id="
curl -sSL "${EFETCH}NC_001422.1" -o phix174.fasta
curl -sSL "${EFETCH}NC_045512.2" -o sars_cov2.fasta
curl -sSL "${EFETCH}V00613.1"    -o ecoli_16s.fasta
echo "Downloaded phix174.fasta, sars_cov2.fasta, ecoli_16s.fasta"
