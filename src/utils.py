import csv
import gzip
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TextIO

import pysam


def read_fasta_index(
    reference_index_path: str,
) -> dict[str, list[int]]:
    """Read a FASTA index file into a dictionary keyed by chromosome."""

    reference_index = {}
    with Path(reference_index_path).open("r", encoding="utf-8") as reference_index_file:
        for line in reference_index_file:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            reference_index[fields[0]] = [int(value) for value in fields[1:]]
    return reference_index


def convert_reference_position_to_file_offset(
    position: int,
    file_sequence_start: int,
    bases_per_line: int,
    characters_per_line: int,
) -> int:
    """Convert a reference coordinate to its byte offset in a FASTA file."""

    return (
        file_sequence_start
        + (position // bases_per_line) * characters_per_line
        + position % bases_per_line
    )


def get_reference_sequence(
    reference_file: TextIO,
    reference_index: dict[str, list[int]],
    chromosome: str,
    start: int,
    end: int | None = None,
) -> str:
    """Return reference sequence for 0-based half-open coordinates."""

    if end is None:
        end = reference_index[chromosome][0]

    if end > reference_index[chromosome][0]:
        raise ValueError("end point exceeds the chromosome length")
    if start > reference_index[chromosome][0]:
        raise ValueError("start point exceeds the chromosome length")
    if start > end:
        raise ValueError("start point larger than end point")

    sequence_start_offset = convert_reference_position_to_file_offset(
        start,
        *reference_index[chromosome][1:],
    )
    sequence_end_offset = convert_reference_position_to_file_offset(
        end,
        *reference_index[chromosome][1:],
    )

    reference_file.seek(sequence_start_offset)
    sequence = (
        reference_file.read(sequence_end_offset - sequence_start_offset)
        .replace("\n", "")
        .upper()
    )

    return sequence


def open_variant_file(
    variant_file_path: str,
    mode: str = "r",
    header: Any = None,
) -> pysam.VariantFile:
    """Open a VCF/BCF file with an optional header."""

    if header is None:
        return pysam.VariantFile(variant_file_path, mode)
    return pysam.VariantFile(variant_file_path, mode, header=header)


def open_text_auto(
    path: str | PathLike[str],
    mode: str = "rt",
) -> Any:
    """Open plain text or gzip-compressed text based on the file suffix."""

    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    if "b" in mode:
        return Path(path).open(mode)
    return Path(path).open(mode, encoding="utf-8")


def infer_structural_variant_type(
    reference_allele: str,
    alternate_alleles: list[str],
    info: dict[str, Any],
) -> str | None:
    """Infer structural variant type from VCF INFO or allele lengths."""

    sv_type = info.get("SVTYPE")
    if sv_type:
        return sv_type.upper()

    symbolic_types = {
        allele[1:-1].upper()
        for allele in alternate_alleles
        if allele.startswith("<") and allele.endswith(">")
    }
    if len(symbolic_types) == 1:
        return next(iter(symbolic_types))

    if len(alternate_alleles) == 1:
        if len(alternate_alleles[0]) > len(reference_allele):
            return "INS"
        if len(reference_allele) > len(alternate_alleles[0]):
            return "DEL"

    return None


def format_genotype(
    sample_data: Any,
) -> list[str]:
    """Format a pysam sample genotype as alternating allele and separator tokens."""

    genotype = sample_data.get("GT", (".", "."))
    alleles = [str(allele) if allele is not None else "." for allele in genotype]
    separator = "|" if sample_data.phased else "/"
    formatted_genotype = []
    for index, allele in enumerate(alleles):
        if index > 0:
            formatted_genotype.append(separator)
        formatted_genotype.append(allele)
    return formatted_genotype


def parse_pysam_variant_record(
    variant_record: pysam.VariantRecord,
) -> dict[str, Any]:
    """Convert a pysam VCF record to the simplified project record dictionary."""

    alternate_alleles = list(variant_record.alts or [])
    info = dict(variant_record.info)
    samples = list(variant_record.samples)
    genotype = None
    if samples:
        genotype = format_genotype(variant_record.samples[samples[0]])

    return {
        "chrom": variant_record.chrom,
        "id": variant_record.id or ".",
        "ref": variant_record.ref,
        "alts": list(variant_record.alts or []),
        "sv_type": infer_structural_variant_type(
            variant_record.ref,
            alternate_alleles,
            info,
        ),
        "start": variant_record.start,
        "end": variant_record.stop
        if variant_record.stop is not None
        else variant_record.start + max(1, len(variant_record.ref)),
        "alleles": variant_record.alleles,
        "gt": genotype,
    }


def parse_variant_records(
    variant_source: str | bytes | PathLike[str] | pysam.VariantFile,
) -> Iterator[dict[str, Any]]:
    """Yield simplified record dictionaries from a path or pysam variant file."""

    if isinstance(variant_source, (str, bytes, PathLike)):
        with open_variant_file(variant_source, "r") as variant_file:
            for variant_record in variant_file:
                yield parse_pysam_variant_record(variant_record)
    else:
        for variant_record in variant_source:
            yield parse_pysam_variant_record(variant_record)


def read_tsv(
    path: Path,
    require_header: bool = True,
) -> list[dict[str, str]]:
    """Read a TSV file and validate that it has a header."""

    if not path.is_file():
        raise FileNotFoundError(f"Required TSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if require_header and reader.fieldnames is None:
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
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Parse ordered SV IDs and haplotype states from a contig name."""

    fields: dict[str, str] = {}
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


@dataclass(frozen=True)
class Haplotype:
    """One haplotype state vector from haplotypes.tsv."""

    hap_id: str
    states: tuple[int, ...]
    sv_ids: tuple[str, ...]


def load_haplotypes(
    cluster_dir: Path,
) -> dict[str, Haplotype]:
    """Load haplotypes for a cluster."""

    path = cluster_dir / "haplotypes.tsv"
    rows = read_tsv(path)
    require_columns(path, ("hap_id", "gt", "name"))
    if not rows:
        raise ValueError(f"No haplotypes found in {path}")

    haplotypes: dict[str, Haplotype] = {}
    expected_sv_ids: tuple[str, ...] | None = None
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
                f"{cluster_dir.name}: haplotypes do not share ordered SV IDs"
            )

        haplotypes[row["hap_id"]] = Haplotype(row["hap_id"], states, sv_ids)
    return haplotypes
