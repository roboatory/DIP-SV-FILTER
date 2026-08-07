#!/usr/bin/env python3
"""
Export the final filtered VCF from contig-pair classification results.

This script is the lightweight decision layer after ``classify_contig_pairs.py``.
It does not run the neural network. Instead, it reads the best pair from each
cluster's ``pair_classification.tsv``, applies haplotype-specific allele
gate/rescue thresholds, and projects the resulting genotypes back onto the
original VCF records.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pysam


@dataclass(frozen=True)
class Haplotype:
    """One haplotype state vector from haplotypes.tsv."""

    hap_id: str
    states: Tuple[int, ...]
    sv_ids: Tuple[str, ...]


@dataclass(frozen=True)
class VariantDecision:
    """Final decision for one evaluated variant."""

    variant_id: str
    genotype: str
    original_pair_genotype: str
    cluster: str
    pair: str
    pair_score: float
    hap1_score: float
    hap2_score: float
    reason: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Apply contig-pair classification results to an original VCF and "
            "export the final filtered callset."
        )
    )
    parser.add_argument(
        "--cluster-dir",
        type=Path,
        required=True,
        help="Root directory containing per-cluster pair_classification.tsv files.",
    )
    parser.add_argument("--input-vcf", type=Path, required=True, help="Original input VCF.")
    parser.add_argument("--output-vcf", type=Path, required=True, help="Filtered output VCF.")
    parser.add_argument(
        "--decision-tsv",
        type=Path,
        help="Optional audit TSV describing the decision for each input VCF record.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output VCF and decision TSV.",
    )
    parser.add_argument(
        "--keep-absent-genotypes",
        action="store_true",
        help=(
            "Keep evaluated variants whose final genotype is 0/0 in the output "
            "VCF. By default these records are filtered out."
        ),
    )
    parser.add_argument(
        "--max-present-ins-allele-score",
        type=float,
        default=1.0,
        help=(
            "Gate original present INS alleles with residual score above this "
            "threshold to 0. Default: 1.0, effectively disabled for sigmoid scores."
        ),
    )
    parser.add_argument(
        "--max-present-del-allele-score",
        type=float,
        default=1.0,
        help=(
            "Gate original present DEL alleles with residual score above this "
            "threshold to 0. Default: 1.0, effectively disabled for sigmoid scores."
        ),
    )
    parser.add_argument(
        "--min-absent-ins-allele-score",
        type=float,
        default=1.0,
        help=(
            "Rescue original absent INS alleles with residual score at or above "
            "this threshold to 1. Default: 1.0, effectively disabled for sigmoid scores."
        ),
    )
    parser.add_argument(
        "--min-absent-del-allele-score",
        type=float,
        default=1.0,
        help=(
            "Rescue original absent DEL alleles with residual score at or above "
            "this threshold to 1. Default: 1.0, effectively disabled for sigmoid scores."
        ),
    )
    return parser.parse_args()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a tab-delimited file with a required header."""

    if not path.is_file():
        raise FileNotFoundError(f"Required TSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV file has no header: {path}")
        return list(reader)


def require_columns(path: Path, columns: Sequence[str]) -> None:
    """Require TSV columns, including for header-only files."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        fieldnames = csv.DictReader(handle, delimiter="\t").fieldnames or []
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def parse_contig_metadata(contig_name: str) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Parse ordered SV IDs and haplotype states from a constructed contig name."""

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
            raise ValueError(f"Invalid SV metadata entry {entry!r}: {contig_name}") from exc
        sv_ids.append(sv_id)

    try:
        states = tuple(int(value) for value in fields["GT"].split(":"))
    except ValueError as exc:
        raise ValueError(f"Invalid GT state vector in contig name: {contig_name}") from exc

    if len(sv_ids) != len(states):
        raise ValueError(
            f"SV and GT vector lengths differ in contig name: "
            f"{len(sv_ids)} SVs versus {len(states)} states"
        )
    if any(state not in (0, 1) for state in states):
        raise ValueError(f"Only biallelic haplotype states 0/1 are supported: {contig_name}")

    return tuple(sv_ids), states


def load_haplotypes(cluster_dir: Path) -> Dict[str, Haplotype]:
    """Load haplotypes for a cluster."""

    path = cluster_dir / "haplotypes.tsv"
    rows = read_tsv(path)
    require_columns(path, ("hap_id", "gt", "name"))
    if not rows:
        raise ValueError(f"No haplotypes found in {path}")

    haplotypes: Dict[str, Haplotype] = {}
    expected_sv_ids: Optional[Tuple[str, ...]] = None
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
            raise ValueError(f"{cluster_dir.name}: haplotypes do not share ordered SV IDs")

        haplotypes[row["hap_id"]] = Haplotype(row["hap_id"], states, sv_ids)
    return haplotypes


def parse_score_field(text: str) -> Dict[str, float]:
    """Parse a semicolon-separated SV score field."""

    scores: Dict[str, float] = {}
    if not text:
        return scores
    for item in text.split(";"):
        if not item:
            continue
        try:
            sv_id, value = item.rsplit("=", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid SV score entry: {item!r}") from exc
        scores[sv_id] = float(value)
    return scores


def genotype_text_to_map(text: str) -> Dict[str, str]:
    """Parse a semicolon-separated SV genotype field."""

    genotypes: Dict[str, str] = {}
    if not text:
        return genotypes
    for item in text.split(";"):
        if not item:
            continue
        try:
            sv_id, genotype = item.rsplit("=", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid SV genotype entry: {item!r}") from exc
        genotypes[sv_id] = genotype
    return genotypes


def score_thresholds_for_sv(
    sv_id: str,
    max_present_ins: float,
    max_present_del: float,
    min_absent_ins: float,
    min_absent_del: float,
) -> Tuple[float, float]:
    """Return present-gate and absent-rescue thresholds for one SV."""

    if ".INS." in sv_id:
        return max_present_ins, min_absent_ins
    if ".DEL." in sv_id:
        return max_present_del, min_absent_del
    return 1.0, 1.0


def update_allele_state(
    state: int,
    score: float,
    max_present_score: float,
    min_absent_score: float,
) -> int:
    """Gate original present alleles and rescue original absent alleles."""

    if state == 1 and score > max_present_score:
        return 0
    if state == 0 and score >= min_absent_score:
        return 1
    return state


def parse_pair_ids(pair_id: str) -> Tuple[str, str]:
    """Parse pair_id into two haplotype IDs."""

    parts = pair_id.split("__")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid pair ID, expected hap1__hap2: {pair_id}")
    return parts[0], parts[1]


def build_cluster_decisions(
    cluster_dir: Path,
    max_present_ins: float,
    max_present_del: float,
    min_absent_ins: float,
    min_absent_del: float,
) -> List[VariantDecision]:
    """Build final genotype decisions for one cluster's best pair."""

    classification_path = cluster_dir / "pair_classification.tsv"
    if not classification_path.is_file():
        return []
    rows = read_tsv(classification_path)
    require_columns(
        classification_path,
        ("pair", "score", "genotype", "hap1_sv_scores", "hap2_sv_scores"),
    )
    if not rows:
        raise ValueError(f"{classification_path} is empty")

    best = rows[0]
    pair_id = best["pair"]
    hap1_id, hap2_id = parse_pair_ids(pair_id)
    haplotypes = load_haplotypes(cluster_dir)
    if hap1_id not in haplotypes or hap2_id not in haplotypes:
        raise ValueError(f"{cluster_dir.name}: pair {pair_id} references unknown haplotypes")

    hap1 = haplotypes[hap1_id]
    hap2 = haplotypes[hap2_id]
    if hap1.sv_ids != hap2.sv_ids:
        raise ValueError(f"{cluster_dir.name}: pair {pair_id} haplotypes have different SV IDs")

    pair_genotypes = genotype_text_to_map(best["genotype"])
    hap1_scores = parse_score_field(best["hap1_sv_scores"])
    hap2_scores = parse_score_field(best["hap2_sv_scores"])
    pair_score = float(best["score"])

    decisions = []
    for sv_id, state1, state2 in zip(hap1.sv_ids, hap1.states, hap2.states, strict=True):
        if sv_id not in pair_genotypes:
            raise ValueError(f"{classification_path}: genotype field missing {sv_id}")
        if sv_id not in hap1_scores or sv_id not in hap2_scores:
            raise ValueError(f"{classification_path}: score fields missing {sv_id}")

        max_present_score, min_absent_score = score_thresholds_for_sv(
            sv_id,
            max_present_ins,
            max_present_del,
            min_absent_ins,
            min_absent_del,
        )
        final_state1 = update_allele_state(
            state1, hap1_scores[sv_id], max_present_score, min_absent_score
        )
        final_state2 = update_allele_state(
            state2, hap2_scores[sv_id], max_present_score, min_absent_score
        )
        low, high = sorted((final_state1, final_state2))
        final_genotype = f"{low}/{high}"
        reasons = []
        if final_state1 != state1:
            reasons.append(f"{hap1_id}:{state1}->{final_state1}")
        if final_state2 != state2:
            reasons.append(f"{hap2_id}:{state2}->{final_state2}")
        reason = ",".join(reasons) if reasons else "unchanged"

        decisions.append(
            VariantDecision(
                variant_id=sv_id,
                genotype=final_genotype,
                original_pair_genotype=pair_genotypes[sv_id],
                cluster=cluster_dir.name,
                pair=pair_id,
                pair_score=pair_score,
                hap1_score=hap1_scores[sv_id],
                hap2_score=hap2_scores[sv_id],
                reason=reason,
            )
        )
    return decisions


def collect_decisions(
    cluster_root: Path,
    max_present_ins: float,
    max_present_del: float,
    min_absent_ins: float,
    min_absent_del: float,
) -> Dict[str, VariantDecision]:
    """Collect final decisions from every cluster and stop on conflicts."""

    if not cluster_root.is_dir():
        raise NotADirectoryError(f"Cluster directory not found: {cluster_root}")

    decisions: Dict[str, VariantDecision] = {}
    for cluster_dir in sorted(path for path in cluster_root.iterdir() if path.is_dir()):
        for decision in build_cluster_decisions(
            cluster_dir,
            max_present_ins,
            max_present_del,
            min_absent_ins,
            min_absent_del,
        ):
            previous = decisions.get(decision.variant_id)
            if previous is not None and previous.genotype != decision.genotype:
                raise ValueError(
                    f"Conflicting genotype for {decision.variant_id}: "
                    f"{previous.genotype} from {previous.cluster}/{previous.pair}, "
                    f"but {decision.genotype} from {decision.cluster}/{decision.pair}"
                )
            decisions[decision.variant_id] = decision
    return decisions


def validate_threshold(name: str, value: float) -> None:
    """Require a sigmoid probability threshold."""

    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def validate_output_paths(
    output_vcf: Path,
    decision_tsv: Optional[Path],
    overwrite: bool,
) -> None:
    """Refuse to overwrite outputs unless requested."""

    existing = [path for path in (output_vcf, decision_tsv) if path is not None and path.exists()]
    if existing and not overwrite:
        preview = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists; use --overwrite:\n{preview}")


def parse_gt_text(genotype: str) -> Tuple[int, int]:
    """Convert 0/0-style genotype text to a pysam GT tuple."""

    parts = genotype.replace("|", "/").split("/")
    if len(parts) != 2:
        raise ValueError(f"Only diploid genotypes are supported: {genotype}")
    return int(parts[0]), int(parts[1])


def record_gt_text(record: pysam.VariantRecord) -> str:
    """Return the first sample genotype as text, or NA for sample-less VCFs."""

    if not record.samples:
        return "NA"
    sample = next(iter(record.samples))
    gt = record.samples[sample].get("GT")
    if gt is None:
        return "NA"
    return "/".join("." if allele is None else str(allele) for allele in gt)


def set_record_gt(record: pysam.VariantRecord, genotype: str) -> None:
    """Set all sample GT fields to the final genotype."""

    if not record.samples:
        return
    gt_tuple = parse_gt_text(genotype)
    for sample in record.samples:
        record.samples[sample]["GT"] = gt_tuple


def vcf_mode_for_output(path: Path) -> str:
    """Return a pysam output mode based on the requested suffix."""

    if path.suffix == ".gz":
        return "wz"
    if path.suffix == ".bcf":
        return "wb"
    return "w"


def write_decision_rows(
    path: Path,
    rows: Iterable[Sequence[object]],
) -> None:
    """Write the optional audit TSV."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "variant_id",
                "vcf_status",
                "original_gt",
                "final_gt",
                "cluster",
                "pair",
                "pair_score",
                "original_pair_gt",
                "hap1_score",
                "hap2_score",
                "reason",
            )
        )
        writer.writerows(rows)


def export_vcf(
    input_vcf: Path,
    output_vcf: Path,
    decisions: Mapping[str, VariantDecision],
    keep_absent_genotypes: bool,
) -> List[Tuple[object, ...]]:
    """Write the filtered VCF and return optional audit rows."""

    decision_rows: List[Tuple[object, ...]] = []
    with pysam.VariantFile(str(input_vcf)) as in_vcf:
        with pysam.VariantFile(str(output_vcf), vcf_mode_for_output(output_vcf), header=in_vcf.header) as out_vcf:
            for record in in_vcf:
                variant_id = record.id
                original_gt = record_gt_text(record)
                decision = decisions.get(variant_id) if variant_id else None

                if decision is None:
                    out_vcf.write(record)
                    decision_rows.append(
                        (
                            variant_id or ".",
                            "kept_skipped_or_not_evaluated",
                            original_gt,
                            original_gt,
                            "NA",
                            "NA",
                            "NA",
                            "NA",
                            "NA",
                            "NA",
                            "not_evaluated",
                        )
                    )
                    continue

                if decision.genotype == "0/0" and not keep_absent_genotypes:
                    decision_rows.append(
                        (
                            decision.variant_id,
                            "filtered_absent",
                            original_gt,
                            decision.genotype,
                            decision.cluster,
                            decision.pair,
                            f"{decision.pair_score:.6f}",
                            decision.original_pair_genotype,
                            f"{decision.hap1_score:.6f}",
                            f"{decision.hap2_score:.6f}",
                            decision.reason,
                        )
                    )
                    continue

                set_record_gt(record, decision.genotype)
                out_vcf.write(record)
                status = "kept_absent" if decision.genotype == "0/0" else "kept_present"
                decision_rows.append(
                    (
                        decision.variant_id,
                        status,
                        original_gt,
                        decision.genotype,
                        decision.cluster,
                        decision.pair,
                        f"{decision.pair_score:.6f}",
                        decision.original_pair_genotype,
                        f"{decision.hap1_score:.6f}",
                        f"{decision.hap2_score:.6f}",
                        decision.reason,
                    )
                )
    return decision_rows


def main() -> None:
    """Entry point."""

    args = parse_args()
    for name in (
        "max_present_ins_allele_score",
        "max_present_del_allele_score",
        "min_absent_ins_allele_score",
        "min_absent_del_allele_score",
    ):
        validate_threshold(f"--{name.replace('_', '-')}", getattr(args, name))

    validate_output_paths(args.output_vcf, args.decision_tsv, args.overwrite)
    decisions = collect_decisions(
        args.cluster_dir.resolve(),
        args.max_present_ins_allele_score,
        args.max_present_del_allele_score,
        args.min_absent_ins_allele_score,
        args.min_absent_del_allele_score,
    )
    decision_rows = export_vcf(
        args.input_vcf.resolve(),
        args.output_vcf,
        decisions,
        args.keep_absent_genotypes,
    )
    if args.decision_tsv:
        write_decision_rows(args.decision_tsv, decision_rows)

    print(f"Evaluated variants: {len(decisions)}")
    print(f"VCF records processed: {len(decision_rows)}")
    print(args.output_vcf)
    if args.decision_tsv:
        print(args.decision_tsv)


if __name__ == "__main__":
    main()
