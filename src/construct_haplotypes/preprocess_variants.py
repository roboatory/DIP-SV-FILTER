from argparse import ArgumentParser
from collections import defaultdict
import sys

from utils import open_text_auto, parse_variant_records


ALLOWED_SV_TYPES = {"INS", "DEL"}


def has_symbolic_alt(
    alternate_alleles: list[str],
) -> bool:
    """Return whether any alternate allele is symbolic."""

    return any(
        alternate_allele.startswith("<") and alternate_allele.endswith(">")
        for alternate_allele in alternate_alleles
    )


def preprocess_vcf(
    input_variant_file_path: str,
    output_variant_file_path: str,
) -> tuple[int, int]:
    """Write a filtered VCF containing only supported insertion and deletion records."""

    counts = defaultdict(int)
    kept_count = 0
    skipped_count = 0

    records = parse_variant_records(input_variant_file_path)
    with (
        open_text_auto(input_variant_file_path, "rt") as input_variant_file,
        open_text_auto(output_variant_file_path, "wt") as output_variant_file,
    ):
        for line in input_variant_file:
            if line.startswith("#"):
                output_variant_file.write(line)
                continue

            record = next(records)
            sv_type = record["sv_type"]

            if (
                len(record["alts"]) != 1
                or has_symbolic_alt(record["alts"])
                or sv_type not in ALLOWED_SV_TYPES
            ):
                skipped_count += 1
                continue

            chromosome = record["chrom"]
            counts[(chromosome, sv_type)] += 1
            columns = line.rstrip("\n").split("\t")
            columns[2] = f"{chromosome}.{sv_type}.{counts[(chromosome, sv_type)]}"
            output_variant_file.write("\t".join(columns) + "\n")
            kept_count += 1

    return kept_count, skipped_count


def build_parser() -> ArgumentParser:
    """Build the command-line parser."""

    # fmt: off
    parser = ArgumentParser(description="Keep INS/DEL records from a VCF and regenerate IDs as <chrom>.<type>.<index>.")
    parser.add_argument("--variant-file-path", dest="variant_file_path", required=True, help="Input VCF")
    parser.add_argument("--output-variant-file-path", dest="output_variant_file_path", required=True, help="Output preprocessed VCF")
    # fmt: on
    return parser


def main() -> int:
    """Run the command-line interface."""

    arguments = build_parser().parse_args()
    kept_count, skipped_count = preprocess_vcf(
        arguments.variant_file_path,
        arguments.output_variant_file_path,
    )
    print(
        f"Done. Kept {kept_count} INS/DEL records; skipped {skipped_count} other records.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
