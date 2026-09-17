# DIP-SV-FILTER

**Cluster-aware diploid hypothesis modeling for structural variant filtering.**

DIP-SV-FILTER refines long-read SV callsets by finding the pair of local haplotypes that best explains the reads. It clusters nearby candidates, constructs alternative pseudo-haplotypes, prunes diploid pairs with k-mer evidence, and realigns reads to the retained haplotypes. A time-distributed CNN–Transformer scores residual SV signal; the best-supported pair determines updated genotypes and the filtered callset.

![DIP-SV-FILTER pipeline: haplotype construction, k-mer prefiltering, read realignment, residual-signal scoring, and VCF export.](docs/images/pipeline.png)

## Installation

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and `minimap2` and `samtools` on `PATH`.

```bash
uv sync --locked
```

## Filtering

Run from the repository root with a sorted, single-sample candidate VCF, a sorted/indexed BAM, an indexed reference FASTA, and a checkpoint matching the current model. Reference contig names must agree across inputs. Data and trained checkpoints are not bundled.

```bash
mkdir -p output

# Prepare candidates and construct local haplotypes.
uv run python -m construct_haplotypes.preprocess_variants \
  --variant-file-path data/candidates.vcf.gz \
  --output-variant-file-path output/candidates.vcf
uv run python -m construct_haplotypes.construct_haplotypes \
  --variant-file-path output/candidates.vcf \
  --reference-file-path data/reference.fa \
  --output-directory output/haplotypes

# Retain candidate pairs and align reads to their haplotypes.
uv run python -m filter_contig_pairs \
  --fasta output/haplotypes --bam data/reads.bam \
  --output-directory output/prefilter --keep-top-pairs 30
uv run python -m align_cluster_reads_to_haplotypes output/prefilter \
  --preset map-hifi --jobs 2 --minimap2-threads 4

# Encode evidence, rank pairs, and export the filtered callset.
uv run python -m encode_pair_target_windows output/prefilter --threads 2
uv run python -m classify_contig_pairs \
  --model models/best_model.pt --cluster-dir output/prefilter
uv run python -m export_filtered_vcf \
  --cluster-dir output/prefilter --input-vcf output/candidates.vcf \
  --output-vcf output/filtered.vcf --decision-tsv output/decisions.tsv
```

Use `--preset map-ont` for Oxford Nanopore reads. Export must use the preprocessed VCF so variant IDs match the haplotype metadata. It normally removes `0/0` calls and preserves unevaluated records from that VCF. Supply only one sample: the exporter applies the selected genotype to every sample column.

**Current scope:** preprocessing retains sequence-resolved, single-alternate insertions and deletions. Haplotype construction excludes overlapping active variants on one haplotype and skips clusters exceeding 1,024 genotype combinations before overlap filtering.

## Training and inference

The model consumes `(2000, 9)` alignment-feature matrices and predicts ten SV-signal probabilities, one per 200 bp subwindow. Channels encode mismatches, deletions, clipping, insertions, mean/maximum insertion and deletion lengths, and depth. A shared CNN feeds three transformer blocks with 100-dimensional embeddings and four attention heads.

<details>
<summary>Generate features, train a model, and predict windows</summary>

```bash
uv run python -m featurizers.generate_labeled_data \
  --bam data/reads.bam --vcf data/truth.vcf.gz \
  --fai data/reference.fa.fai --output-directory output/features --threads 4

# Prepare separate training, validation, and test sets before training.
uv run python -m models.train \
  --train-directory data/features/training/matrices \
  --validation-directory data/features/validation/matrices \
  --test-directory data/features/test/matrices \
  --output-directory output/model --wandb-mode disabled

uv run python -m models.inference \
  --checkpoint-file-path output/model/best_model.pt \
  --split-directory data/features/test/matrices \
  --output-file-path output/predictions.tsv
```

The generator writes matrices and `labels.txt`; it does not split the dataset. Each label row contains a matrix filename/path, a tab, and ten comma-separated binary labels. Training resolves labels in each matrix directory or its parent.

Training uses binary cross-entropy, AdamW, and cosine annealing by default, selecting `best_model.pt` by validation elementwise F1. It also saves a final checkpoint and JSON metrics. W&B logging is enabled unless disabled as above. Window inference writes filenames, ten probabilities, and ten binary predictions to TSV.

Checkpoints from the former 128-dimensional, four-block architecture are incompatible. Feature order, normalization, and label semantics must match training. Use each command's `--help` for options.

</details>

## Repository layout

| Location | Purpose |
| --- | --- |
| `src/construct_haplotypes/` | VCF preprocessing and pseudo-haplotype construction |
| `src/filter_contig_pairs.py` | Read-local k-mer prefiltering |
| `src/align_cluster_reads_to_haplotypes.py` | Per-haplotype alignment |
| `src/encode_pair_target_windows.py` | Pair-aware read assignment and focused encoding |
| `src/classify_contig_pairs.py` | Residual scoring and pair ranking |
| `src/export_filtered_vcf.py` | Genotype updates and VCF export |
| `src/featurizers/` | Alignment features and labeled-data generation |
| `src/models/` | Architecture, training, inference, and diagnostics |

## Development

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pre-commit run --all-files
```

## Paper

Manuscript in preparation. The bioRxiv link and citation will be added when available.
