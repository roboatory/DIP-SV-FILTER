## Project Overview

DIP-SV-FILTER is a post-calling structural-variant (SV) filtering framework for long-read sequencing data. It clusters nearby SV calls, constructs alternative diploid sequence hypotheses, realigns or scores read evidence around those hypotheses, and trains a time-distributed CNN-Transformer classifier over MAMNET-style alignment features.

The repository currently contains:

- Feature extraction and labeled-data generation from BAM/VCF inputs
- Pseudo-haplotype construction, realignment helpers, and k-mer prefiltering utilities
- PyTorch training and inference for the SVHunter-style CNN-Transformer model

## Environment

```bash
uv sync
uv run pre-commit install
```

Requires Python 3.13+ and uses `uv.lock` as the dependency lockfile. Prefer `uv run ...` for project commands so imports and dependencies resolve in the managed environment.

Large generated artifacts are intentionally ignored: `data/`, `wandb/`, `*output*/`, caches, and the local `.venv/`. In this workspace, sibling directories such as `/data/maiziezhou_lab/rohit/data` and `/data/maiziezhou_lab/rohit/models` may contain generated features and trained checkpoints, but they are not part of the Git-tracked package.

## Commands

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Run pre-commit hooks
uv run pre-commit run --all-files

# Extract MAMNET-style windows from a BAM region
uv run python src/featurizers/extract_features.py \
  --bam data/HG002_chr21.bam \
  --contig chr21 \
  --start 1000000 \
  --end 1002000 \
  --output-directory output/features

# Generate labeled 2 kb feature windows from BAM + VCF + FAI
uv run python src/featurizers/generate_labeled_data.py \
  --bam data/HG002_chr21.bam \
  --vcf data/variants/HG002.vcf.gz \
  --fai data/reference.fa.fai \
  --output-directory output/features \
  --threads 4

# Train the classifier
uv run python src/models/train.py \
  --train-directory data/features/training/matrices \
  --validation-directory data/features/validation/matrices \
  --test-directory data/features/test/matrices \
  --output-directory output/model \
  --wandb-mode disabled

# Run inference
uv run python src/models/inference.py \
  --checkpoint-file-path output/model/best_model.pt \
  --split-directory data/features/test/matrices \
  --output-file-path output/predictions.tsv
```

Training logs to Weights & Biases by default. Use `--wandb-mode disabled` for local or automated runs that should not contact W&B.

## Repository Layout

```text
src/
  construct_haplotypes/
    preprocess_variants.py          # VCF normalization/preprocessing helper
    construct_haplotypes.py         # clustered pseudo-haplotype FASTA generation
  featurizers/
    extract_features.py             # BAM region encoding and per-window .npy matrices/plots
    extract_indels.py               # CUTE-SV-style INS/DEL signature extraction
    generate_labeled_data.py        # BAM+VCF+FAI -> matrices + labels.txt
    parse_sample_specific_strings.py # sample-specific string signature analysis
  models/
    architecture.py                 # SVHunterModel and attention/CNN blocks
    train.py                        # dataset loading, metrics, training, checkpoints
    inference.py                    # batch inference TSV writer
  filter_contig_pairs.py            # read-local k-mer prefilter for haplotype pairs
  realign_with_secondary.py         # minimap2 realignment retaining secondary hits
  utils.py                          # VCF/reference helpers
```

The codebase was recently flattened from `src/dipsvfilter/...` to `src/...`; avoid reintroducing `dipsvfilter` imports or paths.

## Data Contracts

Feature matrices:

- Model inputs are `.npy` arrays with shape `(2000, 9)`.
- The 2 kb window is split into 10 contiguous 200 bp subwindows.
- Feature channel order is fixed: mismatch count, deletion count, soft/hard clip count, insertion count, insertion mean, insertion max, deletion mean, deletion max, depth.
- Changing channel order, window length, subwindow count, or normalization changes model semantics and requires coordinated retraining.

Labels:

- `labels.txt` is tab-separated: `<npy_basename>\t<v0,v1,...,v9>`.
- Each label vector must contain exactly 10 binary values, one per 200 bp subwindow.
- `train.py` looks for `labels.txt` either inside the matrix directory or in its parent directory.

Inference output:

- TSV columns are `file_name`, `predicted_probabilities`, and `predicted_labels`.
- Probabilities and labels are comma-separated 10-value vectors.

## Model and Training Notes

`SVHunterModel` expects `(batch, 2000, 9)` tensors. It applies input `LayerNorm`, appends a learnable 1D positional channel, divides each example into ten 200 bp subwindows, encodes subwindows through a shared CNN, projects to a 128-dimensional embedding, adds subwindow positional embeddings, applies 4 transformer blocks with 4 attention heads and key dimension 32, and predicts 10 binary logits through an MLP head. The loss is `BCEWithLogitsLoss`.

Training defaults:

- AdamW, learning rate `2e-4`, weight decay `1e-3`
- Cosine annealing scheduler with `eta_min=1e-6`
- Batch size `64`, epochs `20`, seed `42`
- Best checkpoint selected by validation elementwise F1
- Outputs: `best_model.pt`, `final_model.pt`, `history.json`, `test_metrics.json`, `run_summary.json`

Metrics include elementwise precision/recall/F1/accuracy, exact-match accuracy, and any-SV precision/recall/F1.

## Haplotype and Realignment Utilities

The haplotype path is separate from the classifier path but supports the same SV-filtering goal:

1. `construct_haplotypes/preprocess_variants.py` preprocesses VCF records.
2. `construct_haplotypes/construct_haplotypes.py` clusters nearby variants and writes pseudo-haplotype FASTA files named like `chrom_start-end.fasta`.
3. `realign_with_secondary.py` extracts reads overlapping the original reference interval encoded in the FASTA name and realigns them to the constructed contigs with minimap2 while retaining secondary alignments.
4. `filter_contig_pairs.py` can prefilter large haplotype-pair search spaces using read-local k-mer evidence.

These utilities use script-style imports from `src/`, so run them from the repository root with `uv run python src/...` unless you intentionally adjust `PYTHONPATH`.

## Personal Coding Standards

Apply these standards to all code changes and code-review feedback in this repository.

Language and tooling:

- Use Python 3 and manage environments/dependencies with `uv`.
- Use `uv sync` to install from the lockfile and `uv run <command>` for project commands.
- Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for development dependencies.
- Use Ruff and ruff-format through the configured pre-commit hooks.

Naming:

- Spell out variable and function names fully; avoid abbreviations.
- Mathematical or statistical code may use conventional notation such as `mu`, `sigma`, `x`, `y`, `n`, `p`, and `alpha`.
- Prefer names that express intent over names that describe type or structure.

Comments and docstrings:

- Add comments only for complex or non-obvious blocks.
- Do not comment self-explanatory code; improve naming or structure instead.
- Use single-line docstrings for functions.
- Use multi-line docstrings only when behavior is non-obvious and cannot fit in one sentence.
- Do not use Google, NumPy, or Sphinx/RST docstring sections such as `Args:`, `Returns:`, or `:param`.

Design priorities, in order:

1. Simplicity: prefer straightforward solutions; avoid cleverness unless it is the more performant option.
2. Readability: optimize for code that is easy to read and maintain.
3. Performance: optimize where it measurably matters.

Python conventions:

- Add type hints to all public functions and methods.
- In new or substantially edited function signatures, put each argument on its own line.
- Format function calls according to normal Python practice: short calls on one line, longer calls across lines as needed.
- Prefer `pathlib.Path` over `os.path` for filesystem work.
- Prefer f-strings over `.format()` or `%` formatting.
- Prefer list, dict, and set comprehensions where the result remains readable.
- Raise specific exceptions; do not raise bare `Exception`.
- Do not use wildcard imports.

Project layout:

- Keep root-level `main.py`, when present, as a driver only: parse CLI arguments or config, do minimal wiring, and delegate to `src/`.
- Put implementation code in `src/`, including modules, subpackages, domain logic, services, adapters, and architecture.
- Put tests under `tests/`; keep tooling config and documentation at the project root.

Git conventions:

- Commit prefixes must be one of `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, or `style`.
- Commit messages use `<prefix>: short_description`; the description is lowercase, imperative, short, and has no punctuation.
- Add a commit body only when the why is not obvious.
- Add `closes #<number>` on its own line when closing a GitHub issue.
- Do not add co-author trailers.
- Branch names use `<prefix>/<short_description>`, lowercase with words separated by underscores.
- Pull requests are drafts by default; titles follow commit-message format, and bodies include summary, changes, testing notes, and issue references.

## Development Guidance

- Keep generated data, checkpoints, plots, and W&B outputs out of Git.
- Prefer small, focused changes; this repository mixes polished model code with exploratory genomics scripts.
- Preserve CLI compatibility unless intentionally changing a workflow.
- Be careful with coordinate systems: BAM/pysam region operations are 0-based half-open, while VCF display positions are often 1-based. Existing code generally converts to 0-based half-open internally.
- Do not silently change matrix shape, channel order, label length, threshold semantics, or checkpoint format.
- Before committing, run `uv run ruff check .` and `uv run ruff format .`. For broader validation, run a small `--maximum-samples` training smoke test if feature data is available.
