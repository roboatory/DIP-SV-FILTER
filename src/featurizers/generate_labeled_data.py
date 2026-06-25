import pysam

import numpy as np
from collections import defaultdict

from extract_features import encode_region
from multiprocessing.pool import Pool
from pathlib import Path
import tqdm

pysam.set_verbosity(0)


def get_chromosome_lengths(
    reference_index_path: str,
) -> dict[str, int]:
    """Read chromosome lengths from a FASTA index file."""

    chromosome_lengths = {}
    with open(reference_index_path, "r", encoding="utf-8") as reference_index_file:
        for line in reference_index_file:
            fields = line.strip().split("\t")
            chromosome_lengths[fields[0]] = int(fields[1])
    return chromosome_lengths


def extract_structural_variant_regions(
    variant_file_path: str,
) -> defaultdict[str, list[tuple[int, int]]]:
    """Extract structural variant regions from a VCF."""

    structural_variant_regions = defaultdict(list)
    variant_file = pysam.VariantFile(variant_file_path)
    for record in variant_file.fetch():
        chromosome = record.chrom
        start = record.start
        end = record.stop if record.stop else record.start + 1
        structural_variant_regions[chromosome].append((start, end))

    for chromosome in structural_variant_regions.keys():
        structural_variant_regions[chromosome] = sorted(
            structural_variant_regions[chromosome],
            key=lambda region: (region[0], region[1]),
        )

    return structural_variant_regions


def cluster_structural_variants(
    structural_variant_regions: dict[str, list[tuple[int, int]]],
    chromosome_lengths: dict[str, int],
    distance_threshold: int = 1999,
) -> defaultdict[str, list[list[object]]]:
    """Cluster nearby structural variant regions by chromosome."""

    clustered_regions = defaultdict(list)

    for chromosome, regions in structural_variant_regions.items():
        if not regions:
            continue

        chromosome_length = chromosome_lengths[chromosome]

        current_start, current_end = regions[0]
        variants_in_cluster = [regions[0]]

        for start, end in regions[1:]:
            if start - current_end <= distance_threshold:
                current_end = max(current_end, end)
                variants_in_cluster.append((start, end))
            else:
                clustered_regions[chromosome].append(
                    [
                        (
                            max(0, current_start - distance_threshold),
                            min(current_end + distance_threshold, chromosome_length),
                        ),
                        variants_in_cluster,
                    ]
                )
                current_start, current_end = start, end
                variants_in_cluster = [(start, end)]

        clustered_regions[chromosome].append(
            [
                (
                    max(0, current_start - distance_threshold),
                    min(current_end + distance_threshold, chromosome_length),
                ),
                variants_in_cluster,
            ]
        )

    return clustered_regions


def sample_non_structural_variant_windows(
    clustered_structural_variant_regions: dict[str, list[list[object]]],
    chromosome_lengths: dict[str, int],
    window_size: int = 2000,
    samples_per_chromosome: int = 500,
) -> defaultdict[str, list[list[object]]]:
    """Sample windows away from clustered structural variants."""

    non_structural_variant_windows = defaultdict(list)

    for chromosome, variant_clusters in clustered_structural_variant_regions.items():
        chromosome_length = chromosome_lengths[chromosome]

        non_structural_variant_intervals = list()

        for index in range(len(variant_clusters) + 1):
            if len(variant_clusters) == 0:
                non_structural_variant_intervals.append((0, chromosome_length))
            elif index == 0:
                if variant_clusters[0][0][0] == 0:
                    continue
                else:
                    non_structural_variant_intervals.append(
                        (0, variant_clusters[0][0][0])
                    )
            elif (
                index == len(variant_clusters)
                and variant_clusters[-1][0][1] == chromosome_length
            ):
                continue
            elif index == len(variant_clusters):
                non_structural_variant_intervals.append(
                    (variant_clusters[-1][0][1], chromosome_length)
                )
            else:
                non_structural_variant_intervals.append(
                    (
                        variant_clusters[index - 1][0][1],
                        variant_clusters[index][0][0],
                    )
                )

        while len(non_structural_variant_windows[chromosome]) < samples_per_chromosome:
            sampled_intervals = np.random.choice(
                range(len(non_structural_variant_intervals)),
                size=min(
                    samples_per_chromosome,
                    len(non_structural_variant_intervals),
                ),
                replace=False,
            )

            for sampled_interval in sampled_intervals:
                interval_start, interval_end = non_structural_variant_intervals[
                    sampled_interval
                ]
                if interval_end - interval_start < window_size:
                    continue
                max_start = interval_end - window_size
                sampled_start = np.random.randint(interval_start, max_start + 1)
                non_structural_variant_windows[chromosome].append(
                    [(sampled_start, sampled_start + window_size), []]
                )

    return non_structural_variant_windows


def check_bam_region(
    bam: pysam.AlignmentFile,
    chromosome: str,
    start: int,
    end: int,
    structural_variant_threshold: int = 50,
) -> bool:
    """Return whether a BAM region has little structural variant evidence."""

    try:
        read_iterator = bam.fetch(chromosome, start, end)
    except ValueError:
        return False

    suspicious_signals = 0
    total_checked = 0

    for read in read_iterator:
        if read.is_unmapped or read.is_secondary or read.is_duplicate:
            continue

        total_checked += 1

        if read.is_supplementary or read.has_tag("SA"):
            read_start = read.reference_start
            read_end = read.reference_end

            if (start <= read_start < end) or (start <= read_end < end):
                suspicious_signals += 1
                continue

        current_reference_position = read.reference_start
        has_local_structural_variant = False

        if read.cigartuples:
            for operation, length in read.cigartuples:
                consumes_reference = operation in [0, 2, 3, 7, 8]

                if operation in {1, 2} and length >= structural_variant_threshold:
                    if operation == 1:
                        if start <= current_reference_position < end:
                            has_local_structural_variant = True
                    if operation == 2:
                        deletion_start = current_reference_position
                        deletion_end = current_reference_position + length
                        if max(start, deletion_start) < min(end, deletion_end):
                            has_local_structural_variant = True

                if has_local_structural_variant:
                    suspicious_signals += 1
                    break

                if consumes_reference:
                    current_reference_position += length

    if total_checked > 0:
        ratio = suspicious_signals / total_checked
        if ratio > 0.10:
            return False

    return True


def label_patches(
    chromosome: str,
    region: tuple[int, int],
    structural_variants: list[tuple[int, int]],
    bam: pysam.AlignmentFile,
    patch_size: int = 200,
) -> np.ndarray:
    """Label 200 bp patches that overlap or show evidence of structural variants."""

    patch_count = (region[1] - region[0]) // patch_size

    assert patch_count == 10

    labels = np.zeros(patch_count, dtype=int)

    for patch_index in range(patch_count):
        patch_start = region[0] + patch_index * patch_size
        patch_end = patch_start + patch_size

        for variant_start, variant_end in structural_variants:
            if patch_end > variant_start and patch_start < variant_end:
                labels[patch_index] = 1
                break
        else:
            if not check_bam_region(bam, chromosome, patch_start, patch_end):
                labels[patch_index] = 1

    return labels


def generate_labeled_windows(
    chromosome: str,
    target_region: list[object],
    bam_file: str,
    output_directory: str,
    width: int = 2000,
    stride: int = 500,
    non_structural_variant: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """Generate encoded feature windows and labels for one target region."""

    bam = pysam.AlignmentFile(bam_file, "rb")

    feature_file_and_labels = list()

    region, structural_variants = target_region
    if (region[1] - region[0]) % width == 0:
        window_starts = list(range(region[0], region[1] - width + 1, stride))
        window_ends = list(range(region[0] + width, region[1] + 1, stride))
    else:
        window_starts = list(range(region[0], region[1] - width + 1, stride)) + [
            region[1] - width
        ]
        window_ends = list(range(region[0] + width, region[1] + 1, stride)) + [
            region[1]
        ]

    for window_start, window_end in zip(window_starts, window_ends):
        labels = label_patches(
            chromosome,
            (window_start, window_end),
            structural_variants,
            bam,
        )

        if labels.sum() != 0 and non_structural_variant:
            continue

        output_path = str(
            Path(output_directory) / f"{chromosome}_{window_start}_{window_end}.npy"
        )

        encoded_window = encode_region(bam, chromosome, window_start, window_end)

        if non_structural_variant and not np.any(encoded_window[:, 8]):
            continue

        np.save(output_path, encoded_window)

        feature_file_and_labels.append((output_path, labels))

    return feature_file_and_labels


def main(
    bam_file: str,
    variant_file_path: str,
    reference_index_path: str,
    output_directory_path: str,
    threads: int = 4,
) -> None:
    """Generate labeled feature matrices and a labels file."""

    output_directory = Path(output_directory_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    chromosome_lengths = get_chromosome_lengths(reference_index_path)
    structural_variant_regions = extract_structural_variant_regions(variant_file_path)
    clustered_structural_variant_regions = cluster_structural_variants(
        structural_variant_regions,
        chromosome_lengths,
    )
    non_structural_variant_windows = sample_non_structural_variant_windows(
        clustered_structural_variant_regions,
        chromosome_lengths,
    )

    labels_file_path = output_directory / "labels.txt"

    with (
        Pool(threads) as pool,
        labels_file_path.open("w", encoding="utf-8") as labels_file,
    ):
        print("Processing SV regions...")
        results = []
        for chromosome, regions in clustered_structural_variant_regions.items():
            for region_info in regions:
                result = pool.apply_async(
                    generate_labeled_windows,
                    args=(
                        chromosome,
                        region_info,
                        bam_file,
                        output_directory_path,
                        2000,
                        500,
                        False,
                    ),
                )
                results.append(result)

        for result in tqdm.tqdm(results):
            feature_file_and_labels = result.get()
            for feature_file, labels in feature_file_and_labels:
                labels_str = ",".join(map(str, labels.tolist()))
                labels_file.write(f"{feature_file}\t{labels_str}\n")

        print("Processing non-SV regions...")
        results = []
        for chromosome, regions in non_structural_variant_windows.items():
            for region_info in regions:
                result = pool.apply_async(
                    generate_labeled_windows,
                    args=(
                        chromosome,
                        region_info,
                        bam_file,
                        output_directory_path,
                        2000,
                        500,
                        True,
                    ),
                )
                results.append(result)

        for result in tqdm.tqdm(results):
            feature_file_and_labels = result.get()
            for feature_file, labels in feature_file_and_labels:
                labels_str = ",".join(map(str, labels.tolist()))
                labels_file.write(f"{feature_file}\t{labels_str}\n")


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser(description="Generate labeled data for SV filtering")
    parser.add_argument("--bam", dest="bam_file", required=True, help="Input BAM file")
    parser.add_argument("--vcf", dest="variant_file_path", required=True, help="Input VCF file with SV calls")
    parser.add_argument("--fai", dest="reference_index_path", required=True, help="FAI index file for the reference genome")
    parser.add_argument("--output-directory", dest="output_directory", required=True, help="Directory to save the output feature files and labels")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing")
    # fmt: on
    arguments = parser.parse_args()

    main(
        arguments.bam_file,
        arguments.variant_file_path,
        arguments.reference_index_path,
        arguments.output_directory,
        arguments.threads,
    )
