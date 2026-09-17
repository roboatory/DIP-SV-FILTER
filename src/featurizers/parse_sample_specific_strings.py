import argparse
import ast
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("agg")


def generate_index_file(
    fragments: str,
) -> None:
    """Generate an offset index for a sample-specific-signature file."""

    output_file = fragments.replace("txt", "index")

    with (
        Path(fragments).open("r", encoding="utf-8") as source,
        Path(output_file).open("w", encoding="utf-8") as destination,
    ):
        buffer = ("", 0)

        while signature := source.readline():
            read = signature.split("\t")[0]

            if read != "*":
                offset = source.tell() - len(signature)

                if buffer[0] != "":
                    destination.write(f"{buffer[0]}\t{buffer[1]}\t{offset}\n")

                buffer = (read, offset)

        destination.write(f"{buffer[0]}\t{buffer[1]}\t{source.tell()}\n")


def gather_sample_specific_signatures(
    bed: str,
    fragments: str,
    offsets: pd.DataFrame,
    signatures: str,
    extension: int = 50,
) -> None:
    """Gather sample-specific signatures overlapping one variant BED file."""

    bed_path = Path(bed)
    destination = Path(fragments) / bed_path.name

    variant = bed_path.stem.split("_")
    variant_start_position = int(variant[1])
    variant_end_position = int(variant[2])

    with (
        Path(bed).open("r", encoding="utf-8") as bed_file,
        Path(signatures).open("r", encoding="utf-8") as signatures_file,
        destination.open("w", encoding="utf-8") as fragments_file,
    ):
        _ = bed_file.readline()

        fragments_file.write("CHROMOSOME\tSTART\tEND\tREAD\tTYPE\n")

        reads_encountered = set()

        for read in bed_file:
            read = read.rstrip().split("\t")

            if read[3] not in reads_encountered:
                start_offset = offsets.loc[read[3], "start"]
                end_offset = offsets.loc[read[3], "end"]

                signatures_file.seek(start_offset)

                for read_fragment in signatures_file.read(
                    end_offset - start_offset
                ).split("\n"):
                    tuples = [
                        ast.literal_eval(expression)
                        for expression in re.findall(r"\([^()]*\)", read_fragment)
                    ]

                    if len(tuples) >= 2:
                        for reference_tuple in tuples[1:]:
                            if all(reference_tuple) and reference_tuple[0] == read[0]:
                                if max(
                                    reference_tuple[1],
                                    variant_start_position - extension,
                                ) <= min(
                                    reference_tuple[2], variant_end_position + extension
                                ):
                                    fragments_file.write(
                                        f"{read[0]}\t{reference_tuple[1]}\t"
                                        f"{reference_tuple[2]}\t{read[3]}\tSFS\n"
                                    )

                reads_encountered.add(read[3])


def visualize_fragments(
    fragments_file: str,
    images: str,
) -> None:
    """Write signature-alignment and fragment-length plots for one variant."""

    fragments_path = Path(fragments_file)
    images_path = Path(images)
    variant = fragments_path.stem.split("_")

    chromosome = variant[0]
    variant_start_position = int(variant[1])
    variant_end_position = int(variant[2])

    fragments = pd.read_csv(fragments_file, sep="\t")
    fragments["HEIGHT"] = fragments.groupby("READ").ngroup() + 2
    fragments["LENGTH"] = fragments["END"] - fragments["START"]

    def plot_signature_alignment() -> None:
        """Plot SFS alignments around the variant."""

        plt.plot([variant_start_position, variant_end_position], [1, 1], color="blue")
        for _, row in fragments.iterrows():
            plt.plot(
                [row["START"], row["END"]], [row["HEIGHT"], row["HEIGHT"]], color="red"
            )

        plt.xticks(rotation="vertical")
        plt.tick_params(left=False, labelleft=False)
        plt.ticklabel_format(axis="x", style="sci", useOffset=False)

        plt.tight_layout()
        plt.savefig(
            images_path
            / "signatures"
            / f"{chromosome}_{variant_start_position}_{variant_end_position}.png"
        )
        plt.clf()

    def plot_sample_specific_signature_distribution() -> None:
        """Plot the SFS fragment-length distribution."""

        fragments["LENGTH"].plot.hist()

        plt.xlabel("length")
        plt.ylabel("number of fragments")

        plt.tight_layout()
        plt.savefig(
            images_path
            / "distributions"
            / f"{chromosome}_{variant_start_position}_{variant_end_position}.png"
        )
        plt.clf()

    plot_signature_alignment()
    plot_sample_specific_signature_distribution()


def launch_chromosome_analysis(
    chromosome: str,
    base_bed: str,
    base_images: str,
    base_fragments: str,
    offsets: pd.DataFrame,
    signatures: str,
) -> None:
    """Run sample-specific-signature analysis for one chromosome."""

    bed = Path(base_bed) / chromosome
    images = Path(base_images) / chromosome
    fragments = Path(base_fragments) / chromosome

    (images / "signatures").mkdir(parents=True, exist_ok=True)
    (images / "distributions").mkdir(parents=True, exist_ok=True)
    fragments.mkdir(parents=True, exist_ok=True)

    for bed_file in bed.iterdir():
        gather_sample_specific_signatures(
            str(bed_file),
            str(fragments),
            offsets,
            signatures,
        )
        visualize_fragments(str(fragments / bed_file.name), str(images))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="sample-specific string (SFS) analysis"
    )
    # fmt: off
    parser.add_argument("-c", "--chromosomes", default="chr21", help="limits SFS analysis to particular chromosomes; specify as a comma separated list or using the keyword 'all' for the entire genome (default: chr21)")
    parser.add_argument("-d", "--bed", default="data/bed", help="path to variant BED directory (default: data/bed)")
    parser.add_argument("-f", "--fragments", default="data/fragments", help="output location for SFS binned by chromosomal variant (default: data/fragments)")
    parser.add_argument("-g", "--generate_index", action="store_true", help="generate a .index file for fast lookup")
    parser.add_argument("-i", "--images", default="data/fragment-images", help="output image directory (default: data/fragment-images)")
    parser.add_argument("-s", "--signatures", default="data/SFS_signatures.txt", help="path to .txt file containing the extracted SFS (default: data/SFS_signatures.txt)")
    # fmt: on

    return parser.parse_args()


def main() -> None:
    """Run sample-specific-signature analysis."""

    arguments = parse_arguments()
    chromosomes = (
        [f"chr{chromosome_number}" for chromosome_number in range(1, 23)]
        if arguments.chromosomes == "all"
        else arguments.chromosomes.split(",")
    )
    base_bed = arguments.bed
    base_fragments = arguments.fragments
    generate_index = arguments.generate_index
    base_images = arguments.images
    signatures = arguments.signatures

    if generate_index:
        generate_index_file(signatures)

    if Path(signatures.replace("txt", "index")).exists():
        offsets = pd.read_csv(
            signatures.replace("txt", "index"),
            delimiter="\t",
            index_col=0,
            names=["start", "end"],
        )

        for chromosome in chromosomes:
            launch_chromosome_analysis(
                chromosome, base_bed, base_images, base_fragments, offsets, signatures
            )

    else:
        print("please generate a .index file by adding the flag --generate_index")


main()
