from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pysam


def parse_cluster_from_fasta(
    fasta_path: str,
) -> tuple[str, int, int]:
    """Parse ``chrom_start-end`` cluster coordinates from a FASTA basename."""

    stem = Path(fasta_path).stem
    try:
        chromosome, interval = stem.rsplit("_", 1)
        start_text, end_text = interval.split("-", 1)
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise ValueError(
            "Could not parse cluster coordinates from FASTA basename. "
            "Expected a name like chrom_start-end.fasta, for example "
            "chr1_10000-11000.fasta."
        ) from exc

    if start < 0 or end <= start:
        raise ValueError(f"Invalid cluster interval parsed from FASTA: {stem}")

    return chromosome, start, end


def cluster_name_from_fasta(
    fasta_path: str,
) -> str:
    """Return the output basename derived from the input FASTA."""

    return Path(fasta_path).stem


def count_fasta_records(
    fasta_path: str,
) -> int:
    """Count records in a plain-text FASTA file."""

    count = 0
    with Path(fasta_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1

    if count == 0:
        raise ValueError(f"No contigs found in FASTA: {fasta_path}")

    return count


def sequence_priority(
    record: pysam.AlignedSegment,
) -> tuple[int, int, int]:
    """Rank candidate records for choosing one sequence per read."""

    if not record.is_secondary and not record.is_supplementary:
        record_type_priority = 2
    elif record.is_supplementary:
        record_type_priority = 1
    else:
        record_type_priority = 0

    sequence_length = len(record.query_sequence) if record.query_sequence else 0
    return (record_type_priority, sequence_length, record.mapping_quality)


def collect_region_read_sequences(
    bam_file_path: str,
    chromosome: str,
    start: int,
    end: int,
) -> dict[str, str]:
    """Collect one sequence per read with any alignment overlapping a region."""

    read_sequences: dict[str, str] = {}
    read_priorities: dict[str, tuple[int, int, int]] = {}

    with pysam.AlignmentFile(bam_file_path, "rb") as bam:
        try:
            records = bam.fetch(chromosome, start, end)
        except ValueError as exc:
            raise ValueError(
                f"Region {chromosome}:{start}-{end} is not fetchable from BAM"
            ) from exc

        for record in records:
            if record.is_unmapped or not record.query_sequence:
                continue

            priority = sequence_priority(record)
            if priority > read_priorities.get(record.query_name, (-1, -1, -1)):
                read_sequences[record.query_name] = record.query_sequence.upper()
                read_priorities[record.query_name] = priority

    return read_sequences


def write_reads_fasta(
    read_sequences: dict[str, str],
    output_file_path: str,
    line_width: int = 80,
) -> None:
    """Write deduplicated read sequences to FASTA."""

    with Path(output_file_path).open("w", encoding="utf-8") as handle:
        for read_name in sorted(read_sequences):
            sequence = read_sequences[read_name]
            handle.write(f">{read_name}\n")
            for sequence_start in range(0, len(sequence), line_width):
                sequence_end = sequence_start + line_width
                handle.write(sequence[sequence_start:sequence_end] + "\n")


def run_minimap2(
    minimap2_executable: str,
    contig_fasta_path: str,
    reads_fasta_path: str,
    sam_file_path: str,
    contig_count: int,
    threads: int,
    preset: str,
    soft_clip_supplementary: bool,
) -> None:
    """Run minimap2 and write SAM output to a file."""

    command = [
        minimap2_executable,
        "-a",
        "-x",
        preset,
        "-N",
        str(contig_count),
        "-f",
        "0.05",
        "-t",
        str(threads),
    ]
    if soft_clip_supplementary:
        command.append("-Y")
    command.extend([contig_fasta_path, reads_fasta_path])

    with Path(sam_file_path).open("w", encoding="utf-8") as sam_file:
        subprocess.run(command, stdout=sam_file, check=True)


def sort_and_index_alignment(
    sam_file_path: str,
    bam_file_path: str,
    threads: int,
    index: bool,
) -> None:
    """Sort SAM into BAM and optionally create a BAM index."""

    pysam.sort("-@", str(threads), "-o", bam_file_path, sam_file_path)
    if index:
        pysam.index(bam_file_path)


def realign_cluster(
    arguments: argparse.Namespace,
) -> str:
    """Extract region-overlapping reads and realign them to constructed contigs."""

    if arguments.threads <= 0:
        raise ValueError("--threads must be a positive integer")

    chromosome, start, end = parse_cluster_from_fasta(arguments.fasta_path)
    contig_count = count_fasta_records(arguments.fasta_path)
    cluster_name = cluster_name_from_fasta(arguments.fasta_path)

    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_bam = output_directory / f"{cluster_name}.bam"
    reads_fasta = output_directory / f"{cluster_name}.reads.fasta"

    read_sequences = collect_region_read_sequences(
        arguments.bam_file, chromosome, start, end
    )
    if not read_sequences:
        raise ValueError(f"No reads with sequences found in {chromosome}:{start}-{end}")

    write_reads_fasta(read_sequences, str(reads_fasta))

    with tempfile.TemporaryDirectory(
        prefix=f"{cluster_name}.",
        dir=arguments.temporary_directory or str(output_directory),
    ) as temporary_directory:
        sam_file_path = Path(temporary_directory) / f"{cluster_name}.sam"
        run_minimap2(
            arguments.minimap2_executable,
            arguments.fasta_path,
            str(reads_fasta),
            str(sam_file_path),
            contig_count,
            arguments.threads,
            arguments.preset,
            arguments.soft_clip_supplementary,
        )
        sort_and_index_alignment(
            str(sam_file_path),
            str(output_bam),
            arguments.threads,
            not arguments.no_index,
        )

    if not arguments.keep_reads:
        reads_fasta.unlink()

    print(f"Cluster: {chromosome}:{start}-{end}")
    print(f"Constructed contigs: {contig_count}")
    print(f"Extracted unique reads: {len(read_sequences)}")
    print(f"Wrote BAM: {output_bam}")
    if not arguments.no_index:
        print(f"Wrote BAM index: {output_bam}.bai")

    return str(output_bam)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Extract one sequence per read from an original BAM over the cluster encoded in a constructed-contig FASTA name, then realign those reads to the constructed contigs with minimap2 secondary alignments enabled."
    )
    parser.add_argument(
        "--fasta", dest="fasta_path", required=True, help="Constructed contig FASTA"
    )
    parser.add_argument(
        "--bam", dest="bam_file", required=True, help="Original read alignment BAM"
    )
    parser.add_argument(
        "--output-directory",
        dest="output_directory",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=1, help="Threads for minimap2 and sorting"
    )
    parser.add_argument(
        "--minimap2",
        dest="minimap2_executable",
        default="minimap2",
        help="Path to minimap2 executable",
    )
    parser.add_argument(
        "--preset",
        "-x",
        default="map-hifi",
        help="minimap2 -x preset (default: map-hifi)",
    )
    parser.add_argument(
        "--soft-clip-supplementary",
        action="store_true",
        help="Add minimap2 -Y to soft-clip supplementary alignments",
    )
    parser.add_argument(
        "--keep-reads",
        action="store_true",
        help="Keep the intermediate deduplicated reads FASTA",
    )
    parser.add_argument(
        "--no-index", action="store_true", help="Do not create a BAM index"
    )
    parser.add_argument(
        "--temporary-directory",
        dest="temporary_directory",
        default=None,
        help="Directory for temporary SAM files (default: output directory)",
    )
    return parser


if __name__ == "__main__":
    realign_cluster(build_argument_parser().parse_args())
