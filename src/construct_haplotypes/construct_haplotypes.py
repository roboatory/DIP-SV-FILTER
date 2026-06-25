import itertools
from pathlib import Path
from typing import Any, TextIO

from utils import (
    get_reference_sequence,
    parse_variant_records,
    read_fasta_index,
)


class StructuralVariantCluster:
    reference_file_handle: TextIO | None = None
    reference_index: dict[str, list[int]] | None = None

    def __init__(
        self,
        structural_variant: dict[str, Any],
    ) -> None:
        """Initialize a cluster from the first structural variant."""

        if (
            StructuralVariantCluster.reference_file_handle is None
            or StructuralVariantCluster.reference_index is None
        ):
            raise ValueError(
                "Missing reference file and/or FAI dictionary for StructuralVariantCluster"
            )

        self.chromosome = structural_variant["chrom"]
        self.variant_ids = [structural_variant["id"]]
        self.breakpoints = [(structural_variant["start"], structural_variant["end"])]
        self.alleles = [structural_variant["alleles"]]
        self.genotypes = [
            [
                "0" if genotype_value == "." else genotype_value
                for genotype_value in structural_variant["gt"]
            ]
        ]

    def add_structural_variant(
        self,
        structural_variant: dict[str, Any],
    ) -> None:
        """Add another structural variant to this cluster."""

        self.variant_ids.append(structural_variant["id"])
        self.breakpoints.append(
            (structural_variant["start"], structural_variant["end"])
        )
        self.alleles.append(structural_variant["alleles"])
        self.genotypes.append(
            [
                "0" if genotype_value == "." else genotype_value
                for genotype_value in structural_variant["gt"]
            ]
        )

    def find_overlaps(
        self,
        haplotype_genotypes: list[tuple[int, ...]],
    ) -> bool:
        """Return whether any active variants overlap within a haplotype."""

        for haplotype_genotype in haplotype_genotypes:
            end = -1
            for genotype_state, variant_breakpoint in zip(
                haplotype_genotype, self.breakpoints
            ):
                if genotype_state != 0:
                    if variant_breakpoint[0] < end:
                        return True
                    end = variant_breakpoint[1]
        return False

    def get_genotype_combinations(self) -> list[list[tuple[int, int]]]:
        """Return possible diploid genotype combinations for each variant."""

        # Mirrors hap-eval genotype-combination handling.
        genotype_combinations = []
        for genotype in self.genotypes:
            allele_pair = tuple(
                int(allele) for index, allele in enumerate(genotype) if index % 2 == 0
            )
            if all(allele == allele_pair[0] for allele in allele_pair):
                genotype_combination = [allele_pair, (0, 0)]
            elif genotype[1] == "/":
                genotype_combination = list(itertools.permutations(allele_pair))
                genotype_combination.append((0, 0))
            elif genotype[1] == "|":
                genotype_combination = [allele_pair, (0, 0)]
            else:
                genotype_combination = []
            genotype_combinations.append(genotype_combination)
        return genotype_combinations

    def cluster_bounds(
        self,
        flank: int,
    ) -> tuple[int, int]:
        """Return reference-coordinate bounds for this cluster and flank."""

        cluster_start = max(0, self.breakpoints[0][0] - flank)
        cluster_end = (
            sorted(
                self.breakpoints,
                key=lambda variant_breakpoint: variant_breakpoint[1],
            )[-1][-1]
            + flank
        )
        chromosome_length = StructuralVariantCluster.reference_index[self.chromosome][0]
        return cluster_start, min(cluster_end, chromosome_length)

    def make_header(
        self,
        haplotype_genotype: tuple[int, ...],
        variant_intervals: list[tuple[int, int, int]],
    ) -> str:
        """Build a FASTA header describing variant intervals and genotype state."""

        variant_names = ";".join(
            f"{self.variant_ids[variant_index]}:{sequence_start}-{sequence_end}"
            for variant_index, sequence_start, sequence_end in variant_intervals
        )
        genotype_name = ":".join(
            str(genotype_state) for genotype_state in haplotype_genotype
        )
        return f">SVs={variant_names}|GT={genotype_name}\n"

    @staticmethod
    def project_reference_boundary(
        reference_position: int,
        active_mappings: list[tuple[int, int, int, int]],
        cluster_start: int,
        boundary: str,
    ) -> int:
        """Project a reference boundary onto the constructed haplotype sequence."""

        shift = 0
        for (
            reference_start,
            reference_end,
            haplotype_start,
            haplotype_end,
        ) in active_mappings:
            if reference_position < reference_start:
                break
            if reference_position == reference_start:
                return haplotype_start
            if reference_position < reference_end:
                return haplotype_start if boundary == "start" else haplotype_end
            if reference_position == reference_end:
                return haplotype_end
            shift += (haplotype_end - haplotype_start) - (
                reference_end - reference_start
            )

        return reference_position - cluster_start + shift

    @staticmethod
    def one_base_anchor(
        position: int,
        sequence_length: int,
    ) -> tuple[int, int]:
        """Return a one-base interval constrained to the sequence length."""

        anchor_start = min(max(position, 0), sequence_length - 1)
        return anchor_start, anchor_start + 1

    def build_haplotype_sequence(
        self,
        cluster_start: int,
        cluster_end: int,
        haplotype_genotype: tuple[int, ...],
    ) -> tuple[str, list[tuple[int, int, int]]]:
        """Build one haplotype sequence and its projected variant intervals."""

        haplotype_breakpoints = [
            self.breakpoints[index]
            for index in range(len(haplotype_genotype))
            if haplotype_genotype[index] != 0
        ]
        haplotype_sequences = [
            self.alleles[index][haplotype_genotype[index]]
            for index in range(len(haplotype_genotype))
            if haplotype_genotype[index] != 0
        ]
        haplotype_indexes = [
            index
            for index in range(len(haplotype_genotype))
            if haplotype_genotype[index] != 0
        ]

        if len(haplotype_breakpoints) == 0:
            sequence = get_reference_sequence(
                StructuralVariantCluster.reference_file_handle,
                StructuralVariantCluster.reference_index,
                self.chromosome,
                cluster_start,
                cluster_end,
            )
            variant_intervals = []
            for variant_index, (start, _) in enumerate(self.breakpoints):
                sequence_start, sequence_end = self.one_base_anchor(
                    start - cluster_start, len(sequence)
                )
                variant_intervals.append((variant_index, sequence_start, sequence_end))
            return sequence, variant_intervals

        prefix = get_reference_sequence(
            StructuralVariantCluster.reference_file_handle,
            StructuralVariantCluster.reference_index,
            self.chromosome,
            cluster_start,
            haplotype_breakpoints[0][0],
        )
        parts = [prefix]
        active_mappings = []
        current_position = len(prefix)

        for index, (
            variant_index,
            haplotype_breakpoint,
            haplotype_sequence,
        ) in enumerate(
            zip(haplotype_indexes, haplotype_breakpoints, haplotype_sequences)
        ):
            sequence_start = current_position
            parts.append(haplotype_sequence)
            current_position += len(haplotype_sequence)
            active_mappings.append(
                (
                    haplotype_breakpoint[0],
                    haplotype_breakpoint[1],
                    sequence_start,
                    current_position,
                )
            )

            if index < len(haplotype_breakpoints) - 1:
                connector = get_reference_sequence(
                    StructuralVariantCluster.reference_file_handle,
                    StructuralVariantCluster.reference_index,
                    self.chromosome,
                    haplotype_breakpoint[1],
                    haplotype_breakpoints[index + 1][0],
                )
                parts.append(connector)
                current_position += len(connector)

        suffix = get_reference_sequence(
            StructuralVariantCluster.reference_file_handle,
            StructuralVariantCluster.reference_index,
            self.chromosome,
            haplotype_breakpoints[-1][1],
            cluster_end,
        )
        parts.append(suffix)
        sequence = "".join(parts)
        active_intervals = {
            variant_index: (haplotype_start, haplotype_end)
            for variant_index, (_, _, haplotype_start, haplotype_end) in zip(
                haplotype_indexes, active_mappings
            )
        }
        variant_intervals = []
        for variant_index, (start, _) in enumerate(self.breakpoints):
            if variant_index in active_intervals:
                sequence_start, sequence_end = active_intervals[variant_index]
            else:
                projected_start = self.project_reference_boundary(
                    start, active_mappings, cluster_start, "start"
                )
                sequence_start, sequence_end = self.one_base_anchor(
                    projected_start, len(sequence)
                )
            variant_intervals.append((variant_index, sequence_start, sequence_end))

        return sequence, variant_intervals

    def write(
        self,
        output_directory: Path,
        flank: int,
    ) -> None:
        """Write this cluster's valid pseudo-haplotype FASTA records."""

        cluster_start, cluster_end = self.cluster_bounds(flank)

        genotype_combinations = self.get_genotype_combinations()
        combination_count = 1
        for combination in genotype_combinations:
            combination_count *= len(combination)
        if combination_count > 1024:
            print(
                "%s:%d-%d too many phasing combos %d"
                % (
                    self.chromosome,
                    self.breakpoints[0][0],
                    self.breakpoints[-1][1],
                    combination_count,
                )
            )
            return

        valid_combinations = []

        for combination in itertools.product(*genotype_combinations):
            if not combination:
                continue

            haplotype_genotypes = list(zip(*combination))
            if self.find_overlaps(haplotype_genotypes):
                continue

            valid_combinations += haplotype_genotypes

        written = False
        output_fasta_path = (
            output_directory / f"{self.chromosome}_{cluster_start}-{cluster_end}.fasta"
        )
        with output_fasta_path.open("w", encoding="utf-8") as output_fasta:
            for haplotype_genotype in sorted(set(valid_combinations)):
                sequence, variant_intervals = self.build_haplotype_sequence(
                    cluster_start, cluster_end, haplotype_genotype
                )
                output_fasta.write(
                    self.make_header(haplotype_genotype, variant_intervals)
                )
                output_fasta.write(sequence + "\n")
                written = True

        if not written:
            print("[WARNING]: No valid SV combination found in Cluster:")
            for variant_breakpoint, genotype in zip(self.breakpoints, self.genotypes):
                print(
                    "\t"
                    + self.chromosome
                    + "\t"
                    + str(variant_breakpoint)
                    + "\t"
                    + "".join(genotype)
                )
            output_fasta_path.unlink()


def construct_haplotypes(
    variant_file_path: str,
    reference_file_path: str,
    flank: int,
    output_directory_path: str,
) -> None:
    """Construct pseudo-haplotype FASTA files from a sorted variant file."""

    reference_index_path = reference_file_path + ".fai"
    reference_index = read_fasta_index(reference_index_path)
    output_directory = Path(output_directory_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    with open(reference_file_path, "r", encoding="utf-8") as reference_file_handle:
        StructuralVariantCluster.reference_file_handle = reference_file_handle
        StructuralVariantCluster.reference_index = reference_index

        variant_records = parse_variant_records(variant_file_path)

        for structural_variant in variant_records:
            if structural_variant["gt"] is None:
                continue
            if (
                structural_variant["gt"][0] == "0"
                and structural_variant["gt"][-1] == "0"
            ):
                continue
            chromosome = structural_variant["chrom"]
            last_end = structural_variant["end"]
            structural_variant_cluster = StructuralVariantCluster(structural_variant)
            break
        else:
            raise ValueError("No valid structural variant cluster found in input VCF")

        for structural_variant in variant_records:
            if structural_variant["gt"] is None:
                continue
            if (
                structural_variant["gt"][0] == "0"
                and structural_variant["gt"][-1] == "0"
            ):
                continue

            if structural_variant["chrom"] != chromosome:
                structural_variant_cluster.write(output_directory, flank)
                chromosome = structural_variant["chrom"]
                last_end = structural_variant["end"]
                structural_variant_cluster = StructuralVariantCluster(
                    structural_variant
                )
            elif structural_variant["start"] - last_end < flank:
                structural_variant_cluster.add_structural_variant(structural_variant)
                if structural_variant["end"] > last_end:
                    last_end = structural_variant["end"]
            else:
                structural_variant_cluster.write(output_directory, flank)
                last_end = structural_variant["end"]
                structural_variant_cluster = StructuralVariantCluster(
                    structural_variant
                )

        structural_variant_cluster.write(output_directory, flank)


if __name__ == "__main__":
    from argparse import ArgumentParser

    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--variant-file-path", dest="variant_file_path", help="SORTED VCF")
    parser.add_argument("--output-directory", dest="output_directory", help="Output directory")
    parser.add_argument("--flank", "-f", default=5000, type=int, help="length of flanking reference sequence")
    parser.add_argument("--reference-file-path", dest="reference_file_path", help="reference file")
    # fmt: on

    arguments = parser.parse_args()

    construct_haplotypes(
        arguments.variant_file_path,
        arguments.reference_file_path,
        arguments.flank,
        arguments.output_directory,
    )
