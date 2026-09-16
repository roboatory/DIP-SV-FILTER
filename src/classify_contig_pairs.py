#!/usr/bin/env python3
"""
Classify retained contig pairs from targeted alignment feature windows.

The model predicts one residual-SV probability for each of the ten 200 bp
subwindows in a 2000 bp feature matrix. Only focused subwindows recorded by
``encode_pair_target_windows.py`` contribute to scoring. Scores are aggregated
with an event-balanced mean so that each SV contributes equal weight,
independent of its length or number of overlapping feature windows.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from models.architecture import SVHunterModel

EXPECTED_INPUT_SHAPE = (2000, 9)
EXPECTED_SUBWINDOWS = 10


@dataclass(frozen=True)
class Haplotype:
    """Haplotype state vector and ordered SV identifiers."""

    states: Tuple[int, ...]
    sv_ids: Tuple[str, ...]


@dataclass(frozen=True)
class Pair:
    """One retained haplotype pair."""

    prefilter_rank: int
    pair_id: str
    hap1_id: str
    hap2_id: str


@dataclass(frozen=True)
class PairInputs:
    """Validated targeted-window metadata for one pair."""

    cluster_name: str
    pair: Pair
    focus_rows: Tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class PairResult:
    """Final event-balanced score and genotype for one pair."""

    pair: Pair
    score: float
    genotype_text: str
    sv_hap_scores: Mapping[Tuple[str, str], float]


@dataclass(frozen=True)
class ClusterInputs:
    """Validated classification inputs for one cluster."""

    cluster_dir: Path
    haplotypes: Mapping[str, Haplotype]
    pairs: Tuple[PairInputs, ...]


class FeatureDataset(Dataset[Tuple[Tensor, str]]):
    """Load validated targeted feature matrices for inference."""

    def __init__(
        self,
        feature_paths: Sequence[Path],
    ) -> None:
        """Initialize and validate the dataset or accumulator."""

        self.feature_paths = tuple(feature_paths)

    def __len__(
        self,
    ) -> int:
        """Return the number of feature windows."""

        return len(self.feature_paths)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Tensor, str]:
        """Load a feature window and its associated metadata."""

        path = self.feature_paths[index]
        array = np.load(path).astype(np.float32, copy=False)
        return torch.from_numpy(array), str(path)


def get_default_device_name() -> str:
    """Choose the best available PyTorch device."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run residual-SV classification on pair-aware targeted windows, "
            "rank retained contig pairs within each cluster, and write "
            "pair_classification.tsv files for downstream VCF export."
        )
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="PyTorch model checkpoint."
    )
    parser.add_argument(
        "--cluster-dir",
        type=Path,
        required=True,
        help="Root directory containing all k-mer prefilter cluster directories.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Inference batch size. Default: 64."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader worker processes. Default: 0.",
    )
    parser.add_argument(
        "--device",
        default=get_default_device_name(),
        help="Inference device, for example cuda or cpu. Default: automatically detected.",
    )
    parser.add_argument(
        "--feature-subdir",
        default="pair_target_windows",
        help="Per-cluster targeted feature subdirectory. Default: pair_target_windows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pair_classification.tsv files.",
    )
    parser.add_argument(
        "--top-subwindow-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of the highest focused subwindow scores used to summarize "
            "each SV-haplotype score, with the count rounded up. Default: 0.5."
        ),
    )
    return parser.parse_args()


def read_tsv(
    path: Path,
) -> List[Dict[str, str]]:
    """Read a TSV file and validate that it has a header."""

    if not path.is_file():
        raise FileNotFoundError(f"Required TSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV file has no header: {path}")
        return list(reader)


def require_columns(
    path: Path,
    columns: Sequence[str],
) -> None:
    """Require TSV columns, including for header-only files."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        fieldnames = csv.DictReader(handle, delimiter="\t").fieldnames or []
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def parse_bool(
    value: str,
) -> bool:
    """Parse boolean text from existing TSV output."""

    return value.lower() in {"1", "true", "t", "yes", "y"}


def parse_contig_metadata(
    contig_name: str,
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Parse ordered SV IDs and haplotype states from a contig name."""

    fields: Dict[str, str] = {}
    for item in contig_name.split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value

    if "SVs" not in fields or "GT" not in fields:
        raise ValueError(f"Contig name must contain SVs= and GT= fields: {contig_name}")

    sv_ids = []
    for entry in fields["SVs"].split(";"):
        if not entry:
            continue
        try:
            sv_id, _ = entry.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid SV metadata entry {entry!r}: {contig_name}"
            ) from exc
        sv_ids.append(sv_id)

    try:
        states = tuple(int(value) for value in fields["GT"].split(":"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid GT state vector in contig name: {contig_name}"
        ) from exc

    if len(sv_ids) != len(states):
        raise ValueError(
            f"SV and GT vector lengths differ in contig name: "
            f"{len(sv_ids)} SVs versus {len(states)} states"
        )
    if any(state not in (0, 1) for state in states):
        raise ValueError(
            f"Only biallelic haplotype states 0/1 are supported: {contig_name}"
        )

    return tuple(sv_ids), states


def load_haplotypes(
    cluster_dir: Path,
) -> Dict[str, Haplotype]:
    """Load and cross-check haplotype state vectors."""

    path = cluster_dir / "haplotypes.tsv"
    rows = read_tsv(path)
    require_columns(path, ("hap_id", "gt", "name"))
    if not rows:
        raise ValueError(f"No haplotypes found in {path}")

    haplotypes: Dict[str, Haplotype] = {}
    expected_sv_ids: Tuple[str, ...] | None = None
    for row in rows:
        sv_ids, states = parse_contig_metadata(row["name"])
        tsv_states = tuple(int(value) for value in row["gt"].split(":"))
        if tsv_states != states:
            raise ValueError(
                f"{cluster_dir.name}: haplotype {row['hap_id']} GT differs between "
                "haplotypes.tsv and contig metadata"
            )
        if expected_sv_ids is None:
            expected_sv_ids = sv_ids
        elif sv_ids != expected_sv_ids:
            raise ValueError(
                f"{cluster_dir.name}: haplotypes do not share the same ordered SV IDs"
            )

        haplotypes[row["hap_id"]] = Haplotype(
            states=states,
            sv_ids=sv_ids,
        )

    return haplotypes


def load_retained_pairs(
    cluster_dir: Path,
) -> List[Pair]:
    """Load retained pairs and their original prefilter ranks."""

    path = cluster_dir / "prefilter_pairs.tsv"
    rows = read_tsv(path)
    require_columns(path, ("rank", "pair_id", "hap1_id", "hap2_id", "retained"))

    pairs = [
        Pair(
            prefilter_rank=int(row["rank"]),
            pair_id=row["pair_id"],
            hap1_id=row["hap1_id"],
            hap2_id=row["hap2_id"],
        )
        for row in rows
        if parse_bool(row["retained"])
    ]
    if not pairs:
        raise ValueError(f"No retained pairs found in {path}")
    return sorted(pairs, key=lambda pair: pair.prefilter_rank)


def expected_pair_dir(
    feature_root: Path,
    pair: Pair,
) -> Path:
    """Return the directory name used by pair-aware targeted encoding."""

    return feature_root / f"{pair.prefilter_rank:02d}_{pair.hap1_id}__{pair.hap2_id}"


def validate_pair_inputs(
    cluster_dir: Path,
    pair: Pair,
    feature_root: Path,
) -> Tuple[PairInputs, Tuple[Path, ...]]:
    """Validate metadata and feature matrices for one retained pair."""

    pair_dir = expected_pair_dir(feature_root, pair)
    if not pair_dir.is_dir():
        raise FileNotFoundError(
            f"{cluster_dir.name}: targeted feature directory missing for pair "
            f"{pair.pair_id}: {pair_dir}"
        )

    windows_path = pair_dir / "windows.tsv"
    focus_path = pair_dir / "focus_subwindows.tsv"
    window_rows = read_tsv(windows_path)
    focus_rows = read_tsv(focus_path)
    require_columns(
        windows_path,
        (
            "feature_path",
            "pair_rank",
            "pair_id",
            "hap_id",
            "window_start",
            "window_end",
        ),
    )
    require_columns(
        focus_path,
        (
            "feature_path",
            "pair_rank",
            "pair_id",
            "hap_id",
            "sv_id",
            "subwindow_index",
            "subwindow_start",
            "subwindow_end",
        ),
    )
    if not window_rows:
        raise ValueError(
            f"{cluster_dir.name} pair {pair.pair_id}: windows.tsv is empty"
        )
    if not focus_rows:
        raise ValueError(
            f"{cluster_dir.name} pair {pair.pair_id}: focus_subwindows.tsv is empty"
        )

    feature_paths: Dict[str, Path] = {}
    for row in window_rows:
        if (
            int(row["pair_rank"]) != pair.prefilter_rank
            or row["pair_id"] != pair.pair_id
        ):
            raise ValueError(
                f"{windows_path}: pair metadata does not match {pair.pair_id}"
            )
        relative_path = row["feature_path"]
        if relative_path in feature_paths:
            raise ValueError(f"{windows_path}: duplicate feature_path {relative_path}")
        feature_path = pair_dir / relative_path
        if not feature_path.is_file():
            raise FileNotFoundError(f"Feature matrix not found: {feature_path}")
        array = np.load(feature_path, mmap_mode="r")
        if array.shape != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"{feature_path} has shape {array.shape}; expected {EXPECTED_INPUT_SHAPE}"
            )
        feature_paths[relative_path] = feature_path.resolve()

    for row in focus_rows:
        if (
            int(row["pair_rank"]) != pair.prefilter_rank
            or row["pair_id"] != pair.pair_id
        ):
            raise ValueError(
                f"{focus_path}: pair metadata does not match {pair.pair_id}"
            )
        if row["feature_path"] not in feature_paths:
            raise ValueError(
                f"{focus_path}: feature_path {row['feature_path']} is absent from windows.tsv"
            )
        subwindow_index = int(row["subwindow_index"])
        if not 0 <= subwindow_index < EXPECTED_SUBWINDOWS:
            raise ValueError(
                f"{focus_path}: subwindow index {subwindow_index} is outside 0-9"
            )
        if int(row["subwindow_end"]) <= int(row["subwindow_start"]):
            raise ValueError(f"{focus_path}: invalid physical subwindow coordinates")

    resolved_focus_rows = []
    for row in focus_rows:
        resolved = dict(row)
        resolved["feature_path"] = str(feature_paths[row["feature_path"]])
        resolved_focus_rows.append(resolved)

    return (
        PairInputs(
            cluster_name=cluster_dir.name,
            pair=pair,
            focus_rows=tuple(resolved_focus_rows),
        ),
        tuple(feature_paths.values()),
    )


def discover_cluster_inputs(
    cluster_root: Path,
    feature_subdir: str,
) -> Tuple[List[ClusterInputs], Tuple[Path, ...]]:
    """Discover and validate every cluster before model inference."""

    if not cluster_root.is_dir():
        raise NotADirectoryError(f"Cluster root is not a directory: {cluster_root}")

    clusters: List[ClusterInputs] = []
    all_feature_paths = set()
    for cluster_dir in sorted(path for path in cluster_root.iterdir() if path.is_dir()):
        if not (cluster_dir / "prefilter_pairs.tsv").is_file():
            continue
        haplotypes = load_haplotypes(cluster_dir)
        retained_pairs = load_retained_pairs(cluster_dir)
        feature_root = cluster_dir / feature_subdir
        if not feature_root.is_dir():
            raise FileNotFoundError(
                f"{cluster_dir.name}: targeted feature subdirectory not found: {feature_root}"
            )

        pair_inputs = []
        for pair in retained_pairs:
            if pair.hap1_id not in haplotypes or pair.hap2_id not in haplotypes:
                raise ValueError(
                    f"{cluster_dir.name}: pair {pair.pair_id} references an unknown haplotype"
                )
            validated_pair, feature_paths = validate_pair_inputs(
                cluster_dir, pair, feature_root
            )
            pair_inputs.append(validated_pair)
            all_feature_paths.update(feature_paths)

        clusters.append(
            ClusterInputs(
                cluster_dir=cluster_dir,
                haplotypes=haplotypes,
                pairs=tuple(pair_inputs),
            )
        )

    if not clusters:
        raise ValueError(f"No cluster directories found under {cluster_root}")
    if not all_feature_paths:
        raise ValueError(f"No targeted feature matrices found under {cluster_root}")

    return clusters, tuple(sorted(all_feature_paths))


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Strictly load the checkpoint into the local architecture."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain model_state_dict: {checkpoint_path}"
        )

    model = SVHunterModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def run_inference(
    model: nn.Module,
    feature_paths: Sequence[Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run batched inference once across all cluster feature matrices."""

    dataset = FeatureDataset(feature_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    predictions: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for features, path_strings in dataloader:
            features = features.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            logits = model(features)
            if logits.ndim != 2 or logits.shape[1] != EXPECTED_SUBWINDOWS:
                raise ValueError(
                    f"Model returned shape {tuple(logits.shape)}; expected "
                    f"(batch, {EXPECTED_SUBWINDOWS})"
                )
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for path_string, probability_vector in zip(
                path_strings, probabilities, strict=True
            ):
                predictions[path_string] = probability_vector

    if len(predictions) != len(feature_paths):
        raise RuntimeError(
            f"Inference returned {len(predictions)} predictions for "
            f"{len(feature_paths)} feature matrices"
        )
    return predictions


def pair_genotypes(
    pair: Pair,
    haplotypes: Mapping[str, Haplotype],
) -> Tuple[Tuple[str, str], ...]:
    """Convert two haplotype state vectors to unordered diploid genotypes."""

    hap1 = haplotypes[pair.hap1_id]
    hap2 = haplotypes[pair.hap2_id]
    if hap1.sv_ids != hap2.sv_ids:
        raise ValueError(f"Pair {pair.pair_id} haplotypes have different SV ordering")

    genotypes = []
    for sv_id, state1, state2 in zip(
        hap1.sv_ids, hap1.states, hap2.states, strict=True
    ):
        low, high = sorted((state1, state2))
        genotypes.append((sv_id, f"{low}/{high}"))
    return tuple(genotypes)


def top_fraction_mean(
    values: Sequence[float],
    fraction: float,
) -> float:
    """Average the highest ceil(n * fraction) values."""

    if not values:
        raise ValueError("Cannot summarize an empty score vector")
    if not 0 < fraction <= 1:
        raise ValueError("Top-score fraction must be greater than 0 and at most 1")

    count = max(1, math.ceil(len(values) * fraction))
    top_values = sorted(values, reverse=True)[:count]
    return float(np.mean(top_values))


def score_pair(
    pair_inputs: PairInputs,
    haplotypes: Mapping[str, Haplotype],
    predictions: Mapping[str, np.ndarray],
    top_subwindow_fraction: float,
) -> PairResult:
    """Compute the hierarchical event-balanced score for one pair."""

    pair = pair_inputs.pair
    expected_haps = (
        (pair.hap1_id,)
        if pair.hap1_id == pair.hap2_id
        else (pair.hap1_id, pair.hap2_id)
    )
    expected_sv_ids = haplotypes[pair.hap1_id].sv_ids

    repeated_scores: Dict[Tuple[str, str, int, int], List[float]] = defaultdict(list)
    for row in pair_inputs.focus_rows:
        feature_path = row["feature_path"]
        if feature_path not in predictions:
            raise KeyError(f"Missing model prediction for {feature_path}")
        subwindow_index = int(row["subwindow_index"])
        key = (
            row["hap_id"],
            row["sv_id"],
            int(row["subwindow_start"]),
            int(row["subwindow_end"]),
        )
        repeated_scores[key].append(float(predictions[feature_path][subwindow_index]))

    unique_subwindow_scores = {
        key: float(np.mean(values)) for key, values in repeated_scores.items()
    }

    sv_hap_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for (hap_id, sv_id, _start, _end), score in unique_subwindow_scores.items():
        sv_hap_values[(hap_id, sv_id)].append(score)

    sv_hap_scores = {
        key: top_fraction_mean(values, top_subwindow_fraction)
        for key, values in sv_hap_values.items()
    }

    sv_pair_scores = []
    for sv_id in expected_sv_ids:
        hap_scores = []
        for hap_id in expected_haps:
            key = (hap_id, sv_id)
            if key not in sv_hap_scores:
                raise ValueError(
                    f"{pair_inputs.cluster_name} pair {pair.pair_id}: no focused "
                    f"score for SV {sv_id} on haplotype {hap_id}"
                )
            hap_scores.append(sv_hap_scores[key])
        sv_pair_scores.append(float(np.mean(hap_scores)))

    if not sv_pair_scores:
        raise ValueError(
            f"{pair_inputs.cluster_name} pair {pair.pair_id}: no SV scores generated"
        )

    genotype_text = ";".join(
        f"{sv_id}={genotype}" for sv_id, genotype in pair_genotypes(pair, haplotypes)
    )
    return PairResult(
        pair=pair,
        score=float(np.mean(sv_pair_scores)),
        genotype_text=genotype_text,
        sv_hap_scores=sv_hap_scores,
    )


def classify_clusters(
    clusters: Sequence[ClusterInputs],
    predictions: Mapping[str, np.ndarray],
    top_subwindow_fraction: float,
) -> Dict[str, List[PairResult]]:
    """Score and rank all retained pairs within every cluster."""

    results: Dict[str, List[PairResult]] = {}
    for cluster in clusters:
        pair_results = [
            score_pair(
                pair_inputs,
                cluster.haplotypes,
                predictions,
                top_subwindow_fraction,
            )
            for pair_inputs in cluster.pairs
        ]
        results[cluster.cluster_dir.name] = sorted(
            pair_results,
            key=lambda result: (result.score, result.pair.prefilter_rank),
        )
    return results


def format_sv_scores(
    result: PairResult,
    hap_id: str,
    haplotypes: Mapping[str, Haplotype],
) -> str:
    """Format per-SV scores for one haplotype in a retained pair."""

    scores = []
    for sv_id in haplotypes[hap_id].sv_ids:
        key = (hap_id, sv_id)
        if key not in result.sv_hap_scores:
            raise ValueError(
                f"Pair {result.pair.pair_id}: missing score for {hap_id}/{sv_id}"
            )
        scores.append(f"{sv_id}={result.sv_hap_scores[key]:.6f}")
    return ";".join(scores)


def atomic_write_tsv(
    path: Path,
    rows: Iterable[Sequence[object]],
    header: Sequence[str],
) -> None:
    """Write a TSV through a temporary sibling file, then replace atomically."""

    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_output_paths(
    clusters: Sequence[ClusterInputs],
    overwrite: bool,
) -> None:
    """Refuse to overwrite any requested output unless explicitly allowed."""

    output_paths = [
        cluster.cluster_dir / "pair_classification.tsv" for cluster in clusters
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        preview = "\n".join(str(path) for path in existing[:10])
        extra = "" if len(existing) <= 10 else f"\n... and {len(existing) - 10} more"
        raise FileExistsError(
            f"Classification outputs already exist; use --overwrite:\n{preview}{extra}"
        )


def write_results(
    clusters: Sequence[ClusterInputs],
    cluster_results: Mapping[str, Sequence[PairResult]],
) -> None:
    """Write per-cluster pair rankings."""

    for cluster in clusters:
        rows = [
            (
                result.pair.pair_id,
                f"{result.score:.6f}",
                result.genotype_text,
                format_sv_scores(result, result.pair.hap1_id, cluster.haplotypes),
                format_sv_scores(result, result.pair.hap2_id, cluster.haplotypes),
            )
            for result in cluster_results[cluster.cluster_dir.name]
        ]
        atomic_write_tsv(
            cluster.cluster_dir / "pair_classification.tsv",
            rows,
            header=("pair", "score", "genotype", "hap1_sv_scores", "hap2_sv_scores"),
        )


def main() -> None:
    """Entry point."""

    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not 0 < args.top_subwindow_fraction <= 1:
        raise ValueError(
            "--top-subwindow-fraction must be greater than 0 and at most 1"
        )

    cluster_root = args.cluster_dir.resolve()
    clusters, feature_paths = discover_cluster_inputs(cluster_root, args.feature_subdir)
    validate_output_paths(clusters, args.overwrite)

    device = torch.device(args.device)
    print(f"Clusters: {len(clusters)}")
    print(f"Feature matrices: {len(feature_paths)}")
    print(f"Device: {device}")

    model = load_model(args.model.resolve(), device)
    predictions = run_inference(
        model=model,
        feature_paths=feature_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    cluster_results = classify_clusters(
        clusters,
        predictions,
        args.top_subwindow_fraction,
    )
    write_results(clusters, cluster_results)

    print(f"Pair classifications written: {len(clusters)}")
    print(cluster_root)


if __name__ == "__main__":
    main()
