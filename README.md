# DIP-SV-FILTER

A post-calling structural variant (SV) filtering tool for long-read sequencing data. DIP-SV-FILTER reduces false positives in SV callsets by jointly modeling clustered SV candidates as alternative diploid sequence hypotheses, using a time-distributed CNN-Transformer classifier to score residual alignment signal after local realignment.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # install dependencies from lockfile
uv run pre-commit install   # set up linting hooks
```

## Usage

### Feature extraction

Extract MAMNET-style alignment features from a BAM region into per-window `.npy` matrices (one `(window_size, 9)` matrix per 200 bp window by default) plus heatmaps:

```bash
uv run python src/featurizers/extract_features.py \
  --bam data/HG002_chr21.bam \
  --contig chr21 \
  --start 1000000 \
  --end 1002000 \
  --output-directory output/features
```

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
  --train-directory data/features/training/matrices \
  --validation-directory data/features/validation/matrices \
  --test-directory data/features/test/matrices \
  --output-directory output \
  --wandb-mode disabled
```

Training logs to [Weights & Biases](https://wandb.ai) by default (`--wandb-mode disabled` to turn off).

### Inference

Run a trained model on new feature matrices:

```bash
uv run python src/models/inference.py \
  --checkpoint-file-path output/best_model.pt \
  --split-directory data/features/test/matrices \
  --output-file-path output/predictions.tsv
```

## Project structure

```
src/
  construct_haplotypes/
    preprocess_variants.py          # VCF normalization/preprocessing helper
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
tests/
data/
  features/{training,validation,test}/
    labels.txt                      # per-split label file
    matrices/                       # .npy feature matrices
```

## Linting

```bash
uv run ruff check .
uv run ruff format .
```

Pre-commit hooks run ruff lint, ruff format, and nbstripout automatically.
