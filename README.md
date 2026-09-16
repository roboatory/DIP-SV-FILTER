# DIP-SV-FILTER

A post-calling structural variant (SV) filtering tool for long-read sequencing data. DIP-SV-FILTER reduces false positives in SV callsets by jointly modeling clustered SV candidates as alternative diploid sequence hypotheses, using a time-distributed CNN-Transformer classifier to score residual alignment signal after local realignment.

## Method overview

DIP-SV-FILTER treats filtering as a local diploid sequence explanation problem: which pair of candidate haplotypes best explains the observed reads? This supports joint evaluation and re-genotyping of nearby calls, where overlapping events, alignment artifacts, and alternative variant representations can make independent SV scoring unreliable.

![DIP-SV-FILTER pipeline: train an SV-signal model, cluster candidate variants, construct and prefilter haplotype pairs, realign reads, and select the pair with the least residual SV signal.](docs/images/dip-sv-filter-pipeline.png)

*Pipeline overview extracted from Figure 1 of the project poster, “DipSVFilter: filtering false-positive structural variants through cluster-aware diploid hypothesis modeling,” by Yichen Henry Liu, Rohit Khurana, and Xin Maizie Zhou. The figure describes the full method; implementation coverage is listed below.*

1. **Cluster candidates.** Group nearby SVs from an input VCF into local regions.
2. **Construct pseudo-haplotypes.** Apply alternative allele combinations to the reference sequence and enumerate unordered diploid pairs, including self pairs.
3. **Prefilter with k-mers.** Compare reference-state and alternate-state-specific k-mers with local reads. Each informative read supports its more compatible haplotype within a pair; retain the strongest pairs for alignment-based evaluation.
4. **Realign and partition reads.** In the full method, assign reads to their better-aligned haplotype within each retained pair and encode SV-overlapping windows.
5. **Score residual signal.** Predict SV-signal probabilities in 200 bp subwindows. The poster describes focus masks, top-fraction averaging within haplotype–SV combinations, and an event-balanced mean across SVs.
6. **Select and export.** Choose the pair with the lowest residual signal, update diploid genotypes, and normally remove variants assigned `0/0`, preserving unevaluated variants.

The classifier measures remaining SV-like alignment signal. A lower residual score indicates that a candidate diploid sequence better explains the reads.

## Implementation status

This repository provides separate command-line components for haplotype construction, k-mer prefiltering, realignment, feature generation, model training, and inference. It does **not yet provide an end-to-end command that exports a filtered VCF**.

The poster's pair-aware read partitioning, focused residual-score aggregation, final pair selection, and genotype export are not implemented in this checkout. The current realignment helper aligns reads to a cluster FASTA while retaining secondary hits; model inference outputs per-window predictions.

Current scope and limits:

- VCF preprocessing retains sequence-resolved, single-alternate insertions and deletions and regenerates IDs; it does not perform general VCF normalization.
- Haplotype construction uses input genotypes to generate alternatives, excludes overlapping active variants on the same haplotype, and skips clusters with more than 1,024 genotype combinations before overlap filtering.
- The k-mer prefilter defaults to retaining 30 pairs when filtering is applied and bypasses filtering for small candidate sets.
- Data, trained checkpoints, and generated outputs are not bundled with the package.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked  # install the package and dependencies from the lockfile
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

Realignment also requires `minimap2` on `PATH`. Use a coordinate-sorted, indexed BAM and matching reference FASTA/FAI and VCF contig names. BAM region arguments are 0-based, half-open.

## Usage

Run commands from the repository root. Paths below are examples; supply your own data and checkpoints. These commands expose individual components, not a complete filtering workflow.

### Haplotype construction and k-mer prefiltering

Preprocess a sorted candidate VCF, construct cluster FASTAs, and rank diploid pairs using the original reference-aligned reads:

```bash
mkdir -p output

uv run python src/construct_haplotypes/preprocess_variants.py \
  --variant-file-path data/variants/HG002.vcf.gz \
  --output-variant-file-path output/candidates.vcf

uv run python src/construct_haplotypes/construct_haplotypes.py \
  --variant-file-path output/candidates.vcf \
  --reference-file-path data/reference.fa \
  --output-directory output/haplotypes

uv run python src/filter_contig_pairs.py \
  --fasta output/haplotypes \
  --bam data/HG002_chr21.bam \
  --output-directory output/prefilter \
  --keep-top-pairs 30
```

The constructor writes cluster FASTAs named `chrom_start-end.fasta`. The prefilter exports retained-pair tables, haplotype FASTAs, local reads, and evidence summaries.

### Cluster realignment

For one generated cluster FASTA (replace the example filename with an actual output):

```bash
uv run python src/realign_with_secondary.py \
  --fasta output/haplotypes/chr21_1000000-1010000.fasta \
  --bam data/HG002_chr21.bam \
  --output-directory output/realigned \
  --preset map-hifi \
  --threads 4
```

Use `--preset map-ont` for Oxford Nanopore reads. This helper produces a sorted BAM and, by default, its index; it does not partition reads for individual diploid pairs.

### Feature extraction

Extract MAMNET-style alignment features from a BAM region into per-window `.npy` matrices (one `(window_size, 9)` matrix per 200 bp window by default) plus heatmaps:

```bash
uv run python src/featurizers/extract_features.py \
  --bam data/HG002_chr21.bam \
  --contig chr21 \
  --start 1000000 \
  --end 1002000 \
  --window-size 2000 \
  --output-directory output/features
```

The example sets a 2 kb window for model-compatible input. The standalone extractor defaults to 200 bp windows, which cannot be passed directly to the classifier.

### Labeled data generation

Generate the labeled `(2000, 9)` feature matrices consumed by the model, along with a `labels.txt` of 10 per-subwindow binary labels:

```bash
uv run python src/featurizers/generate_labeled_data.py \
  --bam data/HG002_chr21.bam \
  --vcf data/variants/HG002.vcf.gz \
  --fai data/reference.fa.fai \
  --output-directory output/features \
  --threads 4
```

### Training

Train the CNN-Transformer classifier on pre-extracted feature matrices:

```bash
uv run python src/models/train.py \
  --train-directory data/baseline/features/training/matrices \
  --validation-directory data/baseline/features/validation/matrices \
  --test-directory data/baseline/features/test/matrices \
  --output-directory output \
  --wandb-mode disabled
```

Training logs to [Weights & Biases](https://wandb.ai) by default (`--wandb-mode disabled` to turn off).

Prepare training, validation, and test directories separately; the labeled-data generator does not create these splits automatically. Training defaults to 20 epochs, batch size 64, AdamW with learning rate `2e-4` and weight decay `1e-3`, and cosine annealing. The best checkpoint is selected by validation elementwise F1. Outputs include `best_model.pt`, `final_model.pt`, `history.json`, `test_metrics.json`, and `run_summary.json`.

### Inference

Run a trained model on new feature matrices:

```bash
uv run python src/models/inference.py \
  --checkpoint-file-path output/best_model.pt \
  --split-directory data/baseline/features/test/matrices \
  --output-file-path output/predictions.tsv
```

The output TSV contains `file_name`, `predicted_probabilities`, and `predicted_labels`, with ten comma-separated values in each prediction vector. The default probability threshold is 0.5.

## Features and model

Model inputs are logarithmically transformed NumPy arrays of shape `(2000, 9)`. Channel order is fixed:

1. Mismatch count
2. Deletion count
3. Soft/hard clipping count
4. Insertion count
5. Mean insertion length
6. Maximum insertion length
7. Mean deletion length
8. Maximum deletion length
9. Read depth

Each window is divided into ten contiguous 200 bp subwindows. `labels.txt` associates each matrix filename/path with ten comma-separated binary labels, separated from the filename by a tab. Training searches for this file in the matrix directory or its parent. Labels are generated from VCF overlap and alignment-based checks.

`SVHunterModel` applies input LayerNorm, appends a learnable positional channel, and encodes each subwindow with a shared CNN. The resulting embeddings are projected to 128 dimensions, augmented with subwindow positional embeddings, and processed by four transformer blocks with four attention heads. An MLP predicts one logit per subwindow, trained with `BCEWithLogitsLoss`.

Evaluation reports elementwise precision, recall, F1, and accuracy; exact-match accuracy; and any-SV precision, recall, and F1. These are window/subwindow classification metrics, not final VCF benchmarking metrics. Changing feature order, normalization, window dimensions, or label semantics requires coordinated model changes and retraining.

## Project structure

```
src/
  construct_haplotypes/
    preprocess_variants.py          # explicit-sequence INS/DEL selection and ID assignment
    construct_haplotypes.py         # clustered pseudo-haplotype FASTA generation
  featurizers/
    extract_features.py             # BAM region -> per-window (200, 9) .npy matrices + heatmaps
    extract_indels.py               # intra/inter-alignment SV signature extraction
    generate_labeled_data.py        # BAM+VCF+FAI -> (2000, 9) matrices + labels.txt
    parse_sample_specific_strings.py # sample-specific string analysis
  models/
    architecture.py                 # CNN-Transformer model definition
    train.py                        # training loop, metrics, checkpointing
    inference.py                    # batch inference from checkpoint
  filter_contig_pairs.py            # read-local k-mer prefilter for haplotype pairs
  realign_with_secondary.py         # minimap2 realignment retaining secondary hits
  utils.py                          # VCF/reference helpers
docs/images/
  dip-sv-filter-pipeline.png         # pipeline overview extracted from the poster
plans/                              # local planning documents (ignored)
data/
  baseline/features/{training,validation,test}/
    labels.txt                      # per-split label file
    matrices/                       # .npy feature matrices
  mixed-coverage/                    # symlink to the mixed-coverage training data
```

The `data/` entries describe the local workspace layout, not versioned datasets.

## Research context

The project abstract reports preliminary F1 improvements in complex SV clusters relative to CSV-FILTER and SAMPLOT-ML. The poster presents HG002 benchmarking against the GIAB high-confidence SV set using PacBio HiFi and Oxford Nanopore reads, with comparisons to CSV-Filter, SVJedi-graph, and Kanpig. Performance varies by caller, variant type, and sequencing platform; these reported experiments do not establish an improvement for every setting.

The method description and pipeline figure above draw on the supplied project abstract and poster. Their full experimental workflow extends beyond the components currently connected in this repository.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
```

Pre-commit and pre-push hooks check file hygiene, run Ruff lint and formatting, and strip notebook outputs. To apply formatting manually, run `uv run ruff format .`.

The existing script paths remain supported. `uv sync` also installs the modules under `src/`, so commands such as `uv run python -m models.train --help` work. Realignment additionally requires `minimap2` on `PATH`.

Personal agent instructions, agent configuration, skills, and `plans/` are ignored for new files. Already tracked instruction files remain tracked until explicitly removed from the index.
