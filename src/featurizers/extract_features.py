import argparse
import re
from pathlib import Path

import numpy as np
import pysam

FEATURE_CHANNEL_LABELS = [
    "MISMATCHCOUNT",
    "DELETIONCOUNT",
    "SOFTHARDCOUNT",
    "INSERTIONCOUNT",
    "INSERTIONMEAN",
    "INSERTIONMAX",
    "DELETIONMEAN",
    "DELETIONMAX",
    "DEPTH",
]

REFERENCE_CONSUMING_CIGAR_OPS = {0, 2, 3, 7, 8}
QUERY_ONLY_CIGAR_OPS = {1, 4, 5}
IGNORED_CIGAR_OPS = {6, 9}


def logify_numpy(
    values: np.ndarray,
) -> np.ndarray:
    """Apply the logarithmic transformation used in MAMNET normalization."""

    return np.log(((values > 0) * values) + 1.0) - np.log(
        (np.abs(values) * (values < 0)) + 1.0
    )


def trim_cigar(
    segment: pysam.AlignedSegment,
    region_start: int,
    region_end: int,
) -> list[tuple[int, int]]:
    """Return the CIGAR operations overlapping a target reference region."""

    reference_position = segment.reference_start
    trimmed_cigar = []

    for operation, length in segment.cigartuples:
        if operation in REFERENCE_CONSUMING_CIGAR_OPS:
            reference_end = reference_position + length

            if reference_end > region_start and reference_position < region_end:
                if reference_position < region_start:
                    length -= region_start - reference_position
                if reference_end > region_end:
                    length -= reference_end - region_end
                trimmed_cigar.append((operation, length))

            reference_position = reference_end
        elif operation in QUERY_ONLY_CIGAR_OPS:
            if region_start <= reference_position < region_end:
                trimmed_cigar.append((operation, length))
        elif operation in IGNORED_CIGAR_OPS:
            continue
        else:
            raise ValueError(f"Unsupported CIGAR operation: {operation}")

    return trimmed_cigar


def parse_mdtag(
    segment: pysam.AlignedSegment,
) -> list[tuple[str, int]]:
    """Parse a pysam MD tag into operation tuples."""

    if not segment.has_tag("MD"):
        return []

    md_pattern = re.compile(r"(\d+)|(\^[A-Z]+)|([A-Z])")
    md_string = segment.get_tag("MD")

    md_tokens = []
    for match in md_pattern.finditer(md_string):
        if match.group(1):
            md_tokens.append(("=", int(match.group(1))))
        elif match.group(2):
            deletion_sequence = match.group(2)[1:]
            md_tokens.append(("D", len(deletion_sequence)))
        elif match.group(3):
            md_tokens.append(("X", 1))

    return md_tokens


def trim_mdtag(
    segment: pysam.AlignedSegment,
    region_start: int,
    region_end: int,
) -> list[tuple[str, int]]:
    """Return MD tag operations overlapping a target reference region."""

    md_tokens = parse_mdtag(segment)
    if not md_tokens:
        return []

    reference_position = segment.reference_start
    trimmed_md = []

    for operation, length in md_tokens:
        if operation not in {"=", "X", "D"}:
            raise ValueError(f"Unsupported MD operation: {operation}")

        reference_end = reference_position + length

        if reference_end > region_start and reference_position < region_end:
            if reference_position < region_start:
                length -= region_start - reference_position
            if reference_end > region_end:
                length -= reference_end - region_end
            trimmed_md.append((operation, length))

        reference_position = reference_end

    return trimmed_md


def update_feature_matrix(
    feature_matrix: np.ndarray,
    trimmed_cigar: list[tuple[int, int]],
    trimmed_md: list[tuple[str, int]],
    related_start: int,
    related_end: int,
) -> None:
    """Update a feature matrix from trimmed CIGAR and MD operations."""

    feature_matrix[related_start:related_end, 8] += 1

    position = related_start

    for operation, length in trimmed_cigar:
        if operation in REFERENCE_CONSUMING_CIGAR_OPS:
            end = position + length
        elif operation in QUERY_ONLY_CIGAR_OPS:
            end = position
        else:
            raise ValueError(f"Unsupported CIGAR operation: {operation}")

        if operation == 2:
            feature_matrix[position:end, 1] += 1
            feature_matrix[position:end, 6] += length
            feature_matrix[position:end, 7] = np.maximum(
                feature_matrix[position:end, 7], length
            )
        elif operation == 1:
            feature_matrix[position, 3] += 1
            feature_matrix[position, 4] += length
            feature_matrix[position, 5] = np.maximum(
                feature_matrix[position, 5], length
            )
        elif operation in (4, 5):
            feature_matrix[position, 2] += 1

        position = end

    position = related_start
    for operation, length in trimmed_md:
        if operation not in {"=", "X", "D"}:
            raise ValueError(f"Unsupported MD operation: {operation}")

        end = position + length
        if operation == "X":
            feature_matrix[position:end, 0] += 1
        position = end


def build_feature_matrix(
    bam: pysam.AlignmentFile,
    contig: str,
    region_start: int,
    region_end: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build an unnormalized MAMNET-style feature matrix for one BAM region."""

    region_length = region_end - region_start
    feature_matrix = np.zeros((region_length, 9), dtype=dtype)

    for segment in bam.fetch(contig, region_start, region_end):
        if segment.is_unmapped:
            continue

        trimmed_cigar = trim_cigar(segment, region_start, region_end)
        trimmed_md = trim_mdtag(segment, region_start, region_end)

        related_start = max(0, segment.reference_start - region_start)
        related_end = min(region_length, segment.reference_end - region_start)

        update_feature_matrix(
            feature_matrix, trimmed_cigar, trimmed_md, related_start, related_end
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        insertion_counts = feature_matrix[:, 3]
        deletion_counts = feature_matrix[:, 1]

        feature_matrix[:, 4] = np.where(
            insertion_counts > 0, feature_matrix[:, 4] / insertion_counts, 0
        )
        feature_matrix[:, 6] = np.where(
            deletion_counts > 0, feature_matrix[:, 6] / deletion_counts, 0
        )

    return feature_matrix


def encode_region(
    bam: pysam.AlignmentFile,
    contig: str,
    region_start: int,
    region_end: int,
    output_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Encode one BAM region as a normalized MAMNET-style feature matrix."""

    feature_matrix = build_feature_matrix(
        bam=bam,
        contig=contig,
        region_start=region_start,
        region_end=region_end,
        dtype=np.float32,
    )
    return logify_numpy(feature_matrix).astype(output_dtype)


def write_feature_heatmap(
    feature_matrix: np.ndarray,
    contig: str,
    window_start: int,
    window_end: int,
    output_path: Path,
    dpi: int | None = None,
) -> None:
    """Write a heatmap for one feature matrix if plotting dependencies exist."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("seaborn or matplotlib not installed, skipping heatmap generation.")
        return

    figure, axes = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        feature_matrix.T,
        cmap="viridis",
        ax=axes,
        cbar_kws={"label": "Log-normalized feature value"},
        yticklabels=FEATURE_CHANNEL_LABELS,
    )
    axes.set_xlabel("Position in Window")
    axes.set_title(f"Features Heatmap: {contig}:{window_start}-{window_end}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def write_feature_windows(
    bam_file: str,
    contig: str,
    region_start: int,
    region_end: int,
    window_size: int = 200,
    output_directory: str = "./features/",
    output_dtype: np.dtype = np.float32,
    plot_dpi: int | None = None,
) -> None:
    """Generate normalized feature windows and optional plots for one BAM region."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        bam = pysam.AlignmentFile(bam_file, "rb")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"error opening bam file: {error}") from error

    try:
        feature_matrix = build_feature_matrix(
            bam=bam,
            contig=contig,
            region_start=region_start,
            region_end=region_end,
            dtype=np.float32,
        )
    finally:
        bam.close()

    region_length = region_end - region_start
    for start in range(0, region_length + 1, window_size):
        end = start + window_size
        if end > region_length:
            continue

        window_features = feature_matrix[start:end]
        if np.sum(window_features) == 0:
            continue

        normalized_features = logify_numpy(window_features).astype(output_dtype)
        window_start = region_start + start
        window_end = window_start + (end - start)
        output_stem = f"{contig}_{window_start}_{window_end}"

        np.save(output_path / f"{output_stem}.npy", normalized_features)
        write_feature_heatmap(
            feature_matrix=normalized_features,
            contig=contig,
            window_start=window_start,
            window_end=window_end,
            output_path=output_path / f"{output_stem}.png",
            dpi=plot_dpi,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate MAMNET-style alignment features from a BAM region."
    )
    parser.add_argument(
        "positional_bam_file", nargs="?", help="path to sorted, indexed BAM file"
    )
    parser.add_argument("positional_contig", nargs="?", help="chromosome/contig name")
    parser.add_argument("positional_start", nargs="?", type=int, help="start position")
    parser.add_argument("positional_end", nargs="?", type=int, help="end position")
    parser.add_argument(
        "-b",
        "--bam",
        dest="bam_file",
        default="data/HG002_chr21.bam",
        help="input BAM file (default: data/HG002_chr21.bam)",
    )
    parser.add_argument(
        "-c",
        "--contig",
        default="chr21",
        help="chromosome / contig name (default: chr21)",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=11019054,
        help="start position (0-based, default: 11019054)",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=11020031,
        help="end position (0-based, exclusive; default: 11020031)",
    )
    parser.add_argument(
        "-w",
        "--window-size",
        dest="window_size",
        type=int,
        default=200,
        help="window size for feature tiling (default: 200)",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        dest="output_directory",
        default="output/features",
        help="output directory for feature matrices and plots (default: output/features)",
    )
    parser.add_argument(
        "--output-data-type",
        dest="output_data_type",
        choices=["float16", "float32"],
        default="float16",
        help="NumPy dtype for saved feature windows (default: float16)",
    )
    parser.add_argument(
        "--plot-dots-per-inch",
        dest="plot_dots_per_inch",
        type=int,
        default=300,
        help="heatmap DPI; use 0 for matplotlib default (default: 300)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the feature-window command-line interface."""

    arguments = parse_arguments()
    output_dtype = np.float16 if arguments.output_data_type == "float16" else np.float32
    plot_dpi = (
        None if arguments.plot_dots_per_inch == 0 else arguments.plot_dots_per_inch
    )
    write_feature_windows(
        bam_file=arguments.positional_bam_file or arguments.bam_file,
        contig=arguments.positional_contig or arguments.contig,
        region_start=(
            arguments.positional_start
            if arguments.positional_start is not None
            else arguments.start
        ),
        region_end=(
            arguments.positional_end
            if arguments.positional_end is not None
            else arguments.end
        ),
        window_size=arguments.window_size,
        output_directory=arguments.output_directory,
        output_dtype=output_dtype,
        plot_dpi=plot_dpi,
    )


if __name__ == "__main__":
    main()
