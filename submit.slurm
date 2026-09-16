#!/bin/bash
#SBATCH --job-name=DipSVFilter
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01-00:00:00
#SBATCH --output=DipSVFilter-%j.out
#SBATCH --error=DipSVFilter-%j.err

# Submit from the repository root with VCF, BAM, REFERENCE, MODEL, and OUTPUT set.
# Supply your cluster's account and partition through sbatch options.
set -euo pipefail
: "${VCF:?Set VCF to a sorted candidate VCF}"
: "${BAM:?Set BAM to a sorted and indexed reference-aligned BAM}"
: "${REFERENCE:?Set REFERENCE to an indexed FASTA}"
: "${MODEL:?Set MODEL to a checkpoint for the current architecture}"
: "${OUTPUT:?Set OUTPUT to a new output directory}"

pipeline_threads=${SLURM_CPUS_PER_TASK:-1}
alignment_jobs=$((pipeline_threads / 4))
if ((alignment_jobs < 1)); then alignment_jobs=1; fi
alignment_threads=$((pipeline_threads / alignment_jobs))
mkdir -p "$OUTPUT"
uv sync --locked
uv run python -m construct_haplotypes.preprocess_variants \
    --variant-file-path "$VCF" --output-variant-file-path "$OUTPUT/candidates.vcf"
uv run python -m construct_haplotypes.construct_haplotypes \
    --variant-file-path "$OUTPUT/candidates.vcf" --reference-file-path "$REFERENCE" \
    --output-directory "$OUTPUT/haplotypes"
uv run python -m filter_contig_pairs \
    --fasta "$OUTPUT/haplotypes" --bam "$BAM" \
    --output-directory "$OUTPUT/prefilter" --threads "$pipeline_threads"
uv run python -m align_cluster_reads_to_haps "$OUTPUT/prefilter" \
    --preset "${PRESET:-map-hifi}" --jobs "$alignment_jobs" \
    --minimap2-threads "$alignment_threads"
uv run python -m encode_pair_target_windows "$OUTPUT/prefilter" --threads "$pipeline_threads"
uv run python -m classify_contig_pairs --model "$MODEL" --cluster-dir "$OUTPUT/prefilter"
uv run python -m export_filtered_vcf \
    --cluster-dir "$OUTPUT/prefilter" --input-vcf "$OUTPUT/candidates.vcf" \
    --output-vcf "$OUTPUT/filtered.vcf" --decision-tsv "$OUTPUT/decisions.tsv" \
    --max-present-ins-allele-score 0.95 --min-absent-del-allele-score 0.5
