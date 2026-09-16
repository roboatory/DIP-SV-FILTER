import gzip
from collections.abc import Iterator
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
