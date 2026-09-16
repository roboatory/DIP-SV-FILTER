from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pysam

DNA_ALPHABET = set("ACGT")
REVCOMP_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


@dataclass(frozen=True)
class StructuralVariantInterval:
    """Pseudo-haplotype interval for one SV state."""

    structural_variant_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Haplotype:
    """One generated pseudo-haplotype sequence."""

    haplotype_id: str
    name: str
    genotype_states: tuple[int, ...]
    structural_variant_intervals: tuple[StructuralVariantInterval, ...]
    sequence: str


@dataclass(frozen=True)
class HaplotypePair:
    """Unordered diploid pseudo-haplotype pair."""

    pair_id: str
    first_haplotype: Haplotype
    second_haplotype: Haplotype


@dataclass(frozen=True)
class Evidence:
    """One k-mer vote for an SV state."""

    structural_variant_index: int
    structural_variant_id: str
    state: int
    weight: float


@dataclass
class ReadEvidence:
    """Read-local evidence accumulated over informative k-mers."""

    support: defaultdict[int, list[float]] = field(
        default_factory=lambda: defaultdict(lambda: [0.0, 0.0])
    )
    informative_kmers: int = 0

    @property
    def weight(
        self,
    ) -> float:
        """Cap read contribution so highly repetitive evidence cannot dominate."""

        return min(1.0, self.informative_kmers / 20.0)


@dataclass
class ClusterReads:
    """Read sequences and localized query intervals for one cluster."""

    sequences: dict[str, str]
    intervals: dict[str, list[tuple[int, int]]]


@dataclass(frozen=True)
class PairScore:
    """Prefilter result for one candidate pair."""

    rank: int
    pair: HaplotypePair
    score: float | None
    informative_reads: int
    total_read_weight: float
    unexplained_present: float
    unexplained_absent: float
    retained: bool
    reason: str


def reverse_complement(
    sequence: str,
) -> str:
    """Return reverse complement sequence."""

    return sequence.translate(REVCOMP_TABLE)[::-1].upper()


def canonical_kmer(
    kmer: str,
) -> str:
    """Return canonical representation of a k-mer."""

    kmer = kmer.upper()
    return min(kmer, reverse_complement(kmer))


def homopolymer_compress(
    sequence: str,
) -> str:
    """Collapse consecutive identical bases."""

    if not sequence:
        return sequence

    compressed = [sequence[0]]
    for base in sequence[1:]:
        if base != compressed[-1]:
            compressed.append(base)
    return "".join(compressed)


def max_homopolymer_run(
    sequence: str,
) -> int:
    """Return the longest run of one repeated base."""

    if not sequence:
        return 0

    best = 1
    current = 1
    for previous, base in zip(sequence, sequence[1:]):
        if base == previous:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def is_low_complexity(
    kmer: str,
) -> bool:
    """Heuristic low-complexity filter for first-pass informative k-mers."""

    unique_bases = set(kmer)
    if len(unique_bases) <= 1:
        return True
    if max(kmer.count(base) for base in unique_bases) / len(kmer) >= 0.8:
        return True
    if max_homopolymer_run(kmer) / len(kmer) >= 0.6:
        return True
    return False


def iter_kmers(
    sequence: str,
    k: int,
    canonical: bool = True,
    filter_low_complexity: bool = True,
) -> Iterator[str]:
    """Yield valid k-mers from a sequence."""

    sequence = sequence.upper()
    if len(sequence) < k:
        return

    for start in range(0, len(sequence) - k + 1):
        kmer = sequence[start : start + k]
        if any(base not in DNA_ALPHABET for base in kmer):
            continue
        if filter_low_complexity and is_low_complexity(kmer):
            continue
        yield canonical_kmer(kmer) if canonical else kmer


def read_fasta(
    fasta_path: str,
) -> list[tuple[str, str]]:
    """Read FASTA records as ``(name, sequence)`` tuples."""

    records: list[tuple[str, str]] = []
    current_name: str | None = None
    current_parts: list[str] = []

    with Path(fasta_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    records.append((current_name, "".join(current_parts).upper()))
                current_name = line[1:].split()[0]
                current_parts = []
            elif current_name is None:
                raise ValueError(f"FASTA sequence found before header in {fasta_path}")
            else:
                current_parts.append(line)

    if current_name is not None:
        records.append((current_name, "".join(current_parts).upper()))
    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    return records


def parse_header_fields(
    name: str,
) -> dict[str, str]:
    """Parse pipe-delimited FASTA header fields."""

    fields: dict[str, str] = {}
    for part in name.split("|"):
        key, sep, value = part.partition("=")
        if not sep:
            continue
        fields[key] = value
    return fields


def parse_structural_variant_intervals(
    value: str,
) -> tuple[StructuralVariantInterval, ...]:
    """Parse ``SVs=sv_id:start-end;...`` header field."""

    intervals: list[StructuralVariantInterval] = []
    if not value:
        return tuple(intervals)

    for item in value.split(";"):
        if not item:
            continue
        structural_variant_id, sep, coords = item.rpartition(":")
        if not sep:
            raise ValueError(f"Malformed SV interval item: {item}")
        start_text, dash, end_text = coords.partition("-")
        if not dash:
            raise ValueError(f"Malformed SV interval coordinates: {item}")
        intervals.append(
            StructuralVariantInterval(
                structural_variant_id=structural_variant_id,
                start=int(start_text),
                end=int(end_text),
            )
        )
    return tuple(intervals)


def parse_genotype_states(
    value: str,
) -> tuple[int, ...]:
    """Parse ``GT=0:1:...`` state vector."""

    if not value:
        raise ValueError("Missing GT value")
    return tuple(int(item) for item in value.split(":"))


def contig_genotype_label(
    genotype_states: Sequence[int],
) -> str:
    """Return compact genotype-state label used as haplotype ID."""

    return "".join(str(state) for state in genotype_states)


def load_haplotypes(
    fasta_path: str,
) -> list[Haplotype]:
    """Load pseudo-haplotypes and validate header consistency."""

    haplotypes: list[Haplotype] = []
    structural_variant_ids: tuple[str, ...] | None = None
    seen_haplotype_ids: set[str] = set()

    for name, sequence in read_fasta(fasta_path):
        fields = parse_header_fields(name)
        if "SVs" not in fields or "GT" not in fields:
            raise ValueError(f"FASTA header must contain SVs= and GT= fields: {name}")

        intervals = parse_structural_variant_intervals(fields["SVs"])
        genotype_states = parse_genotype_states(fields["GT"])
        if len(intervals) != len(genotype_states):
            raise ValueError(
                f"Header has {len(intervals)} SV intervals but {len(genotype_states)} GT states: {name}"
            )

        current_structural_variant_ids = tuple(
            interval.structural_variant_id for interval in intervals
        )
        if structural_variant_ids is None:
            structural_variant_ids = current_structural_variant_ids
        elif current_structural_variant_ids != structural_variant_ids:
            raise ValueError(
                "All haplotypes in one FASTA must list SVs in the same order"
            )

        haplotype_id = contig_genotype_label(genotype_states)
        if haplotype_id in seen_haplotype_ids:
            raise ValueError(f"Duplicate haplotype GT label in FASTA: {haplotype_id}")
        seen_haplotype_ids.add(haplotype_id)

        haplotypes.append(
            Haplotype(
                haplotype_id=haplotype_id,
                name=name,
                genotype_states=genotype_states,
                structural_variant_intervals=intervals,
                sequence=sequence,
            )
        )

    return haplotypes


def cluster_interval_from_fasta(
    fasta_path: str,
) -> tuple[str, int, int]:
    """Parse ``chrom_start-end`` from FASTA basename."""

    stem = Path(fasta_path).stem
    match = re.match(r"^(.+)_(\d+)-(\d+)$", stem)
    if not match:
        raise ValueError(
            "FASTA basename must encode cluster interval as chrom_start-end: "
            f"{Path(fasta_path).name}"
        )
    chromosome, start, end = match.groups()
    return chromosome, int(start), int(end)


def make_pairs(
    haplotypes: Sequence[Haplotype],
) -> list[HaplotypePair]:
    """Generate unordered haplotype pairs, including homozygous/self pairs."""

    pairs: list[HaplotypePair] = []
    for first_haplotype, second_haplotype in itertools.combinations_with_replacement(
        haplotypes, 2
    ):
        pair_id = f"{first_haplotype.haplotype_id}__{second_haplotype.haplotype_id}"
        pairs.append(
            HaplotypePair(
                pair_id=pair_id,
                first_haplotype=first_haplotype,
                second_haplotype=second_haplotype,
            )
        )
    return pairs


def interval_context(
    sequence: str,
    start: int,
    end: int,
    window: int,
) -> str:
    """Return sequence window around a pseudo-haplotype SV interval."""

    left = max(0, start - window)
    right = min(len(sequence), end + window)
    if left >= right:
        return ""
    return sequence[left:right]


def build_structural_variant_state_kmers(
    haplotypes: Sequence[Haplotype],
    k: int,
    window: int,
    canonical: bool,
    filter_low_complexity: bool,
    homopolymer_compression: bool,
) -> dict[int, dict[int, set[str]]]:
    """
    Build state-differentiating k-mers for each SV.

    This first implementation uses pseudo-haplotype intervals from FASTA headers.
    For each SV, k-mers observed in state 0 contexts are contrasted with k-mers
    observed in nonzero/present state contexts.
    """

    if not haplotypes:
        raise ValueError("No haplotypes supplied")

    structural_variant_count = len(haplotypes[0].genotype_states)
    raw: dict[int, dict[int, set[str]]] = {
        structural_variant_index: {0: set(), 1: set()}
        for structural_variant_index in range(structural_variant_count)
    }

    for haplotype in haplotypes:
        for structural_variant_index, interval in enumerate(
            haplotype.structural_variant_intervals
        ):
            state = 0 if haplotype.genotype_states[structural_variant_index] == 0 else 1
            context = interval_context(
                haplotype.sequence, interval.start, interval.end, window
            )
            if homopolymer_compression:
                context = homopolymer_compress(context)
            raw[structural_variant_index][state].update(
                iter_kmers(
                    context,
                    k,
                    canonical=canonical,
                    filter_low_complexity=filter_low_complexity,
                )
            )

    state_kmers: dict[int, dict[int, set[str]]] = {}
    for structural_variant_index in range(structural_variant_count):
        absent = raw[structural_variant_index][0] - raw[structural_variant_index][1]
        present = raw[structural_variant_index][1] - raw[structural_variant_index][0]
        state_kmers[structural_variant_index] = {0: absent, 1: present}

    return state_kmers


def build_kmer_to_evidence(
    haplotypes: Sequence[Haplotype],
    state_kmers: dict[int, dict[int, set[str]]],
    absent_weight: float,
    present_weight: float,
) -> dict[str, list[Evidence]]:
    """Build inverted k-mer evidence index."""

    structural_variant_ids = [
        interval.structural_variant_id
        for interval in haplotypes[0].structural_variant_intervals
    ]
    kmer_to_evidence: defaultdict[str, list[Evidence]] = defaultdict(list)

    for structural_variant_index, by_state in state_kmers.items():
        for state, kmers in by_state.items():
            weight = absent_weight if state == 0 else present_weight
            for kmer in kmers:
                kmer_to_evidence[kmer].append(
                    Evidence(
                        structural_variant_index=structural_variant_index,
                        structural_variant_id=structural_variant_ids[
                            structural_variant_index
                        ],
                        state=state,
                        weight=weight,
                    )
                )

    return dict(kmer_to_evidence)


def query_interval_from_record(
    record: pysam.AlignedSegment,
    reference_start: int,
    reference_end: int,
) -> tuple[int, int] | None:
    """Map a reference interval overlap to query coordinates for one alignment."""

    if record.cigartuples is None or record.reference_start is None:
        return None

    reference_position = record.reference_start
    query_position = 0
    query_starts: list[int] = []
    query_ends: list[int] = []

    for operation, length in record.cigartuples:
        consumes_reference = operation in (0, 2, 3, 7, 8)
        consumes_query = operation in (0, 1, 4, 5, 7, 8)

        if operation in (0, 7, 8):
            block_start = reference_position
            block_end = reference_position + length
            overlap_start = max(block_start, reference_start)
            overlap_end = min(block_end, reference_end)
            if overlap_start < overlap_end:
                query_starts.append(query_position + (overlap_start - block_start))
                query_ends.append(query_position + (overlap_end - block_start))

        if consumes_reference:
            reference_position += length
        if consumes_query:
            query_position += length

    if not query_starts:
        return None
    return min(query_starts), max(query_ends)


def nearby_large_query_segments(
    record: pysam.AlignedSegment,
    reference_start: int,
    reference_end: int,
    large_insertion_threshold: int,
    soft_clip_threshold: int,
) -> list[tuple[int, int]]:
    """Collect large insertion or soft-clip query segments near the cluster interval."""

    if record.cigartuples is None or record.reference_start is None:
        return []

    reference_position = record.reference_start
    query_position = 0
    segments: list[tuple[int, int]] = []

    for operation, length in record.cigartuples:
        consumes_reference = operation in (0, 2, 3, 7, 8)
        consumes_query = operation in (0, 1, 4, 5, 7, 8)

        if operation == 1 and length >= large_insertion_threshold:
            if reference_start <= reference_position <= reference_end:
                segments.append((query_position, query_position + length))
        elif operation == 4 and length >= soft_clip_threshold:
            clip_is_near_left = (
                abs(reference_position - reference_start) <= soft_clip_threshold
            )
            clip_is_near_right = (
                abs(reference_position - reference_end) <= soft_clip_threshold
            )
            clip_inside = reference_start <= reference_position <= reference_end
            if clip_inside or clip_is_near_left or clip_is_near_right:
                segments.append((query_position, query_position + length))

        if consumes_reference:
            reference_position += length
        if consumes_query:
            query_position += length

    return segments


def merge_intervals(
    intervals: Iterable[tuple[int, int]],
    flank: int,
    sequence_length: int,
) -> list[tuple[int, int]]:
    """Merge query intervals after adding a flank."""

    expanded = [
        (max(0, start - flank), min(sequence_length, end + flank))
        for start, end in intervals
        if start < end
    ]
    if not expanded:
        return []

    expanded.sort()
    merged = [expanded[0]]
    for start, end in expanded[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def record_sequence_priority(
    record: pysam.AlignedSegment,
) -> tuple[int, int, int]:
    """Rank records for recovering one representative read sequence."""

    sequence_length = len(record.query_sequence) if record.query_sequence else 0
    primary_rank = 1 if not record.is_secondary and not record.is_supplementary else 0
    return primary_rank, sequence_length, record.mapping_quality


def collect_cluster_reads(
    bam_file_path: str,
    chromosome: str,
    reference_start: int,
    reference_end: int,
    large_insertion_threshold: int,
    soft_clip_threshold: int,
) -> ClusterReads:
    """Collect full read sequences and local query intervals from the original BAM."""

    read_sequences: dict[str, str] = {}
    read_priorities: dict[str, tuple[int, int, int]] = {}
    read_intervals: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)

    with pysam.AlignmentFile(bam_file_path, "rb") as bam:
        for record in bam.fetch(chromosome, reference_start, reference_end):
            if record.is_unmapped or not record.query_name:
                continue

            sequence = record.query_sequence
            if sequence:
                priority = record_sequence_priority(record)
                if priority > read_priorities.get(record.query_name, (-1, -1, -1)):
                    read_sequences[record.query_name] = sequence.upper()
                    read_priorities[record.query_name] = priority

            mapped_interval = query_interval_from_record(
                record,
                reference_start,
                reference_end,
            )
            if mapped_interval is not None:
                read_intervals[record.query_name].append(mapped_interval)
            read_intervals[record.query_name].extend(
                nearby_large_query_segments(
                    record,
                    reference_start,
                    reference_end,
                    large_insertion_threshold=large_insertion_threshold,
                    soft_clip_threshold=soft_clip_threshold,
                )
            )

    return ClusterReads(sequences=read_sequences, intervals=dict(read_intervals))


def build_read_evidence(
    cluster_reads: ClusterReads,
    kmer_to_evidence: dict[str, list[Evidence]],
    k: int,
    query_flank: int,
    minimum_information_kmers: int,
    canonical: bool,
    filter_low_complexity: bool,
    homopolymer_compression: bool,
) -> dict[str, ReadEvidence]:
    """Extract local read k-mers and accumulate SV-state evidence."""

    read_evidence: dict[str, ReadEvidence] = {}
    for read_name, intervals in cluster_reads.intervals.items():
        sequence = cluster_reads.sequences.get(read_name)
        if not sequence:
            continue
        local_intervals = merge_intervals(intervals, query_flank, len(sequence))
        if not local_intervals:
            continue

        local_kmers: set[str] = set()
        for start, end in local_intervals:
            local_sequence = sequence[start:end]
            if homopolymer_compression:
                local_sequence = homopolymer_compress(local_sequence)
            local_kmers.update(
                iter_kmers(
                    local_sequence,
                    k,
                    canonical=canonical,
                    filter_low_complexity=filter_low_complexity,
                )
            )

        evidence = ReadEvidence()
        for kmer in local_kmers:
            votes = kmer_to_evidence.get(kmer)
            if not votes:
                continue
            evidence.informative_kmers += 1
            for vote in votes:
                evidence.support[vote.structural_variant_index][vote.state] += (
                    vote.weight
                )

        if evidence.informative_kmers >= minimum_information_kmers:
            read_evidence[read_name] = evidence

    return read_evidence


def haplotype_compatibility(
    haplotype: Haplotype,
    evidence: ReadEvidence,
) -> float:
    """Compute positive read compatibility for one haplotype."""

    score = 0.0
    for structural_variant_index, support in evidence.support.items():
        state = 0 if haplotype.genotype_states[structural_variant_index] == 0 else 1
        score += support[state]
    return score


def unexplained_evidence(
    pair: HaplotypePair,
    evidence: ReadEvidence,
) -> tuple[float, float]:
    """Summarize evidence contradicted by both haplotypes in a pair."""

    unexplained_present = 0.0
    unexplained_absent = 0.0
    for structural_variant_index, support in evidence.support.items():
        dosage = int(
            pair.first_haplotype.genotype_states[structural_variant_index] != 0
        ) + int(pair.second_haplotype.genotype_states[structural_variant_index] != 0)
        if dosage == 0:
            unexplained_present += support[1]
        elif dosage == 2:
            unexplained_absent += support[0]
    return unexplained_present, unexplained_absent


def score_pair(
    pair: HaplotypePair,
    read_evidence: dict[str, ReadEvidence],
) -> PairScore:
    """Score one candidate pair by read-level best-haplotype compatibility."""

    score = 0.0
    informative_reads = 0
    total_read_weight = 0.0
    unexplained_present = 0.0
    unexplained_absent = 0.0

    for evidence in read_evidence.values():
        read_weight = evidence.weight
        first_haplotype_score = haplotype_compatibility(
            pair.first_haplotype,
            evidence,
        )
        second_haplotype_score = haplotype_compatibility(
            pair.second_haplotype,
            evidence,
        )
        best_score = max(first_haplotype_score, second_haplotype_score)
        if best_score == 0:
            continue

        present, absent = unexplained_evidence(pair, evidence)
        score += read_weight * best_score
        unexplained_present += read_weight * present
        unexplained_absent += read_weight * absent
        informative_reads += 1
        total_read_weight += read_weight

    return PairScore(
        rank=0,
        pair=pair,
        score=score,
        informative_reads=informative_reads,
        total_read_weight=total_read_weight,
        unexplained_present=unexplained_present,
        unexplained_absent=unexplained_absent,
        retained=False,
        reason="scored",
    )


def select_scored_pairs(
    scores: Sequence[PairScore],
    keep_top: int,
) -> list[PairScore]:
    """Select retained pairs after scoring."""

    keep = min(keep_top, len(scores))
    selected: list[PairScore] = []
    for rank, score in enumerate(scores[:keep], start=1):
        selected.append(
            PairScore(
                rank=rank,
                pair=score.pair,
                score=score.score,
                informative_reads=score.informative_reads,
                total_read_weight=score.total_read_weight,
                unexplained_present=score.unexplained_present,
                unexplained_absent=score.unexplained_absent,
                retained=True,
                reason=f"top_{keep}",
            )
        )
    return selected


def bypass_pairs(
    pairs: Sequence[HaplotypePair],
) -> list[PairScore]:
    """Return all pairs without k-mer scoring."""

    return [
        PairScore(
            rank=rank,
            pair=pair,
            score=None,
            informative_reads=0,
            total_read_weight=0.0,
            unexplained_present=0.0,
            unexplained_absent=0.0,
            retained=True,
            reason="bypass_total_pairs_le_threshold",
        )
        for rank, pair in enumerate(pairs, start=1)
    ]


def score_pairs(
    pairs: Sequence[HaplotypePair],
    read_evidence: dict[str, ReadEvidence],
) -> list[PairScore]:
    """Score and sort all candidate pairs."""

    scores = [score_pair(pair, read_evidence) for pair in pairs]
    scores.sort(
        key=lambda result: (
            -float(result.score or 0.0),
            result.unexplained_present,
            result.unexplained_absent,
            -result.informative_reads,
            result.pair.pair_id,
        )
    )
    return scores


def summarize_structural_variant_evidence(
    haplotypes: Sequence[Haplotype],
    state_kmers: dict[int, dict[int, set[str]]],
    read_evidence: dict[str, ReadEvidence],
) -> list[dict[str, object]]:
    """Build per-SV evidence summary rows."""

    rows: list[dict[str, object]] = []
    structural_variant_ids = [
        interval.structural_variant_id
        for interval in haplotypes[0].structural_variant_intervals
    ]
    for structural_variant_index, structural_variant_id in enumerate(
        structural_variant_ids
    ):
        absent_support = 0.0
        present_support = 0.0
        absent_reads = 0
        present_reads = 0
        for evidence in read_evidence.values():
            support = evidence.support.get(structural_variant_index)
            if not support:
                continue
            absent_support += support[0]
            present_support += support[1]
            absent_reads += int(support[0] > 0)
            present_reads += int(support[1] > 0)

        rows.append(
            {
                "sv_index": structural_variant_index,
                "sv_id": structural_variant_id,
                "absent_kmers": len(state_kmers[structural_variant_index][0]),
                "present_kmers": len(state_kmers[structural_variant_index][1]),
                "absent_support": f"{absent_support:.6g}",
                "present_support": f"{present_support:.6g}",
                "absent_reads": absent_reads,
                "present_reads": present_reads,
            }
        )
    return rows


def write_tsv(
    path: str,
    rows: Sequence[dict[str, object]],
    fields: Sequence[str],
) -> None:
    """Write rows to a tab-delimited file."""

    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def wrap_sequence(
    sequence: str,
    width: int = 80,
) -> Iterator[str]:
    """Yield wrapped FASTA sequence lines."""

    for start in range(0, len(sequence), width):
        yield sequence[start : start + width]


def write_cluster_reads_fasta(
    cluster_reads: ClusterReads,
    path: str,
) -> None:
    """Write one deduplicated full-read FASTA for downstream alignment."""

    with Path(path).open("w", encoding="utf-8") as handle:
        for read_name in sorted(cluster_reads.sequences):
            handle.write(f">{read_name}\n")
            for line in wrap_sequence(cluster_reads.sequences[read_name]):
                handle.write(line + "\n")


def retained_haplotypes(
    scores: Sequence[PairScore],
) -> list[Haplotype]:
    """Return unique haplotypes appearing in retained pairs, preserving rank order."""

    retained: list[Haplotype] = []
    seen_haplotype_ids: set[str] = set()
    for score in scores:
        for haplotype in (
            score.pair.first_haplotype,
            score.pair.second_haplotype,
        ):
            if haplotype.haplotype_id in seen_haplotype_ids:
                continue
            retained.append(haplotype)
            seen_haplotype_ids.add(haplotype.haplotype_id)
    return retained


def write_retained_haplotype_fastas(
    haplotypes: Sequence[Haplotype],
    cluster_dir: str,
) -> list[dict[str, object]]:
    """Write one single-contig FASTA per retained haplotype."""

    cluster_path = Path(cluster_dir)
    haplotype_fasta_directory = cluster_path / "hap_fastas"
    haplotype_fasta_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for haplotype in haplotypes:
        fasta_path = haplotype_fasta_directory / f"{haplotype.haplotype_id}.fasta"
        with fasta_path.open("w", encoding="utf-8") as handle:
            handle.write(f">{haplotype.name}\n")
            for line in wrap_sequence(haplotype.sequence):
                handle.write(line + "\n")

        rows.append(
            {
                "hap_id": haplotype.haplotype_id,
                "gt": ":".join(str(state) for state in haplotype.genotype_states),
                "hap_fasta": str(fasta_path.relative_to(cluster_path)),
                "contig_name": haplotype.name,
                "length": len(haplotype.sequence),
            }
        )

    return rows


def pair_score_rows(
    scores: Sequence[PairScore],
    total_pairs: int,
    prefilter_applied: bool,
) -> list[dict[str, object]]:
    """Convert pair scores to TSV rows."""

    rows: list[dict[str, object]] = []
    for result in scores:
        rows.append(
            {
                "rank": result.rank,
                "pair_id": result.pair.pair_id,
                "hap1_id": result.pair.first_haplotype.haplotype_id,
                "hap2_id": result.pair.second_haplotype.haplotype_id,
                "hap1_gt": ":".join(
                    str(state) for state in result.pair.first_haplotype.genotype_states
                ),
                "hap2_gt": ":".join(
                    str(state) for state in result.pair.second_haplotype.genotype_states
                ),
                "score": "NA" if result.score is None else f"{result.score:.8g}",
                "informative_reads": result.informative_reads,
                "total_read_weight": f"{result.total_read_weight:.6g}",
                "unexplained_present": f"{result.unexplained_present:.6g}",
                "unexplained_absent": f"{result.unexplained_absent:.6g}",
                "retained": str(result.retained).lower(),
                "reason": result.reason,
                "prefilter_applied": str(prefilter_applied).lower(),
                "total_pairs": total_pairs,
            }
        )
    return rows


def haplotype_rows(
    haplotypes: Sequence[Haplotype],
) -> list[dict[str, object]]:
    """Convert haplotypes to TSV rows."""

    rows: list[dict[str, object]] = []
    for haplotype in haplotypes:
        rows.append(
            {
                "hap_id": haplotype.haplotype_id,
                "gt": ":".join(str(state) for state in haplotype.genotype_states),
                "name": haplotype.name,
                "length": len(haplotype.sequence),
            }
        )
    return rows


def read_evidence_rows(
    read_evidence: dict[str, ReadEvidence],
    haplotypes: Sequence[Haplotype],
) -> list[dict[str, object]]:
    """Convert read evidence to long-form TSV rows."""

    structural_variant_ids = [
        interval.structural_variant_id
        for interval in haplotypes[0].structural_variant_intervals
    ]
    rows: list[dict[str, object]] = []
    for read_name, evidence in sorted(read_evidence.items()):
        for structural_variant_index, support in sorted(evidence.support.items()):
            rows.append(
                {
                    "read_id": read_name,
                    "sv_index": structural_variant_index,
                    "sv_id": structural_variant_ids[structural_variant_index],
                    "absent_support": f"{support[0]:.6g}",
                    "present_support": f"{support[1]:.6g}",
                    "informative_kmers": evidence.informative_kmers,
                    "read_weight": f"{evidence.weight:.6g}",
                }
            )
    return rows


def discover_fasta_paths(
    path: str,
) -> list[str]:
    """Return FASTA input paths from one file or a directory."""

    input_path = Path(path)
    if input_path.is_dir():
        fasta_paths = [
            str(fasta_path)
            for fasta_path in input_path.iterdir()
            if fasta_path.name.endswith((".fa", ".fasta", ".fna"))
        ]
        return sorted(fasta_paths)
    return [path]


def process_cluster(
    fasta_path: str,
    arguments: argparse.Namespace,
) -> None:
    """Run prefilter for one pseudo-haplotype FASTA cluster."""

    cluster_name = Path(fasta_path).stem
    cluster_directory = Path(arguments.output_directory) / cluster_name
    cluster_directory.mkdir(parents=True, exist_ok=True)

    haplotypes = load_haplotypes(fasta_path)
    pairs = make_pairs(haplotypes)
    total_pairs = len(pairs)
    chromosome, reference_start, reference_end = cluster_interval_from_fasta(fasta_path)
    cluster_reads = collect_cluster_reads(
        arguments.bam_file,
        chromosome,
        reference_start,
        reference_end,
        large_insertion_threshold=arguments.large_insertion_threshold,
        soft_clip_threshold=arguments.soft_clip_threshold,
    )
    reads_fasta_path = cluster_directory / "cluster_reads.fasta"
    write_cluster_reads_fasta(cluster_reads, str(reads_fasta_path))

    write_tsv(
        str(cluster_directory / "haplotypes.tsv"),
        haplotype_rows(haplotypes),
        fields=("hap_id", "gt", "name", "length"),
    )

    if total_pairs <= arguments.prefilter_pair_threshold:
        retained = bypass_pairs(pairs)
        prefilter_applied = False
        read_evidence: dict[str, ReadEvidence] = {}
        state_kmers = {
            structural_variant_index: {0: set(), 1: set()}
            for structural_variant_index in range(len(haplotypes[0].genotype_states))
        }
    else:
        state_kmers = build_structural_variant_state_kmers(
            haplotypes,
            k=arguments.k,
            window=arguments.structural_variant_window,
            canonical=not arguments.no_canonical,
            filter_low_complexity=not arguments.keep_low_complexity,
            homopolymer_compression=arguments.homopolymer_compress,
        )
        kmer_to_evidence = build_kmer_to_evidence(
            haplotypes,
            state_kmers,
            absent_weight=arguments.absent_weight,
            present_weight=arguments.present_weight,
        )
        read_evidence = build_read_evidence(
            cluster_reads,
            kmer_to_evidence,
            k=arguments.k,
            query_flank=arguments.query_flank,
            minimum_information_kmers=arguments.minimum_information_kmers,
            canonical=not arguments.no_canonical,
            filter_low_complexity=not arguments.keep_low_complexity,
            homopolymer_compression=arguments.homopolymer_compress,
        )
        scores = score_pairs(pairs, read_evidence)
        retained = select_scored_pairs(
            scores,
            keep_top=arguments.keep_top_pairs,
        )
        prefilter_applied = True

    pair_rows = pair_score_rows(
        retained, total_pairs=total_pairs, prefilter_applied=prefilter_applied
    )
    write_tsv(
        str(cluster_directory / "prefilter_pairs.tsv"),
        pair_rows,
        fields=(
            "rank",
            "pair_id",
            "hap1_id",
            "hap2_id",
            "hap1_gt",
            "hap2_gt",
            "score",
            "informative_reads",
            "total_read_weight",
            "unexplained_present",
            "unexplained_absent",
            "retained",
            "reason",
            "prefilter_applied",
            "total_pairs",
        ),
    )

    retained_haplotypes_list = retained_haplotypes(retained)
    retained_hap_rows = write_retained_haplotype_fastas(
        retained_haplotypes_list, str(cluster_directory)
    )
    write_tsv(
        str(cluster_directory / "retained_haplotypes.tsv"),
        retained_hap_rows,
        fields=("hap_id", "gt", "hap_fasta", "contig_name", "length"),
    )

    write_tsv(
        str(cluster_directory / "sv_evidence.tsv"),
        summarize_structural_variant_evidence(haplotypes, state_kmers, read_evidence),
        fields=(
            "sv_index",
            "sv_id",
            "absent_kmers",
            "present_kmers",
            "absent_support",
            "present_support",
            "absent_reads",
            "present_reads",
        ),
    )

    if arguments.write_read_evidence and read_evidence:
        write_tsv(
            str(cluster_directory / "read_evidence.tsv"),
            read_evidence_rows(read_evidence, haplotypes),
            fields=(
                "read_id",
                "sv_index",
                "sv_id",
                "absent_support",
                "present_support",
                "informative_kmers",
                "read_weight",
            ),
        )

    with (cluster_directory / "prefilter.log").open("w", encoding="utf-8") as handle:
        handle.write(f"fasta\t{fasta_path}\n")
        handle.write(f"bam\t{arguments.bam_file}\n")
        handle.write(f"cluster_reads_fasta\t{reads_fasta_path}\n")
        handle.write(f"haplotypes\t{len(haplotypes)}\n")
        handle.write(f"total_pairs\t{total_pairs}\n")
        handle.write(
            f"prefilter_pair_threshold\t{arguments.prefilter_pair_threshold}\n"
        )
        handle.write(f"keep_top_pairs\t{arguments.keep_top_pairs}\n")
        handle.write(f"prefilter_applied\t{str(prefilter_applied).lower()}\n")
        handle.write(f"retained_pairs\t{len(retained)}\n")
        handle.write(f"retained_haplotypes\t{len(retained_haplotypes_list)}\n")
        handle.write(f"cluster_reads\t{len(cluster_reads.sequences)}\n")
        handle.write(f"informative_reads\t{len(read_evidence)}\n")
        handle.write(f"k\t{arguments.k}\n")

    print(
        f"{cluster_name}: haplotypes={len(haplotypes)} total_pairs={total_pairs} "
        f"prefilter={str(prefilter_applied).lower()} retained={len(retained)}"
    )


def process_cluster_worker(
    task: tuple[str, argparse.Namespace],
) -> None:
    """Multiprocessing worker for one cluster FASTA."""

    fasta_path, arguments = task
    process_cluster(fasta_path, arguments)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="High-recall read-local k-mer prefilter for pseudo-haplotype pairs. Input FASTA can be one cluster FASTA or a directory of cluster FASTAs."
    )
    parser.add_argument(
        "--fasta",
        dest="fasta_path",
        required=True,
        help="Pseudo-haplotype FASTA file or directory",
    )
    parser.add_argument(
        "--bam", dest="bam_file", required=True, help="Original reference-aligned BAM"
    )
    parser.add_argument(
        "--output-directory",
        dest="output_directory",
        required=True,
        help="Output directory",
    )
    parser.add_argument("--k", type=int, default=21, help="k-mer length (default: 21)")
    parser.add_argument(
        "--structural-variant-window",
        dest="structural_variant_window",
        type=int,
        default=250,
        help="Sequence flank around SV interval for state k-mers",
    )
    parser.add_argument(
        "--query-flank",
        type=int,
        default=3000,
        help="Query flank around localized cluster interval",
    )
    parser.add_argument(
        "--minimum-information-kmers",
        dest="minimum_information_kmers",
        type=int,
        default=5,
        help="Minimum informative k-mers required per read",
    )
    parser.add_argument(
        "--large-insertion-threshold",
        dest="large_insertion_threshold",
        type=int,
        default=50,
        help="Large insertion CIGAR threshold",
    )
    parser.add_argument(
        "--soft-clip-threshold",
        dest="soft_clip_threshold",
        type=int,
        default=50,
        help="Large soft-clip CIGAR threshold",
    )
    parser.add_argument(
        "--absent-weight",
        type=float,
        default=0.5,
        help="Weight for absent/reference evidence",
    )
    parser.add_argument(
        "--present-weight",
        type=float,
        default=1.0,
        help="Weight for present/ALT evidence",
    )
    parser.add_argument(
        "--prefilter-pair-threshold",
        type=int,
        default=30,
        help="Bypass prefilter at or below this pair count",
    )
    parser.add_argument(
        "--keep-top-pairs",
        type=int,
        default=30,
        help="Number of top scored pairs to retain after prefiltering",
    )
    parser.add_argument(
        "--homopolymer-compress",
        action="store_true",
        help="Apply homopolymer compression before k-mer extraction",
    )
    parser.add_argument(
        "--no-canonical",
        action="store_true",
        help="Do not canonicalize reverse-complement k-mers",
    )
    parser.add_argument(
        "--keep-low-complexity", action="store_true", help="Keep low-complexity k-mers"
    )
    parser.add_argument(
        "--write-read-evidence",
        action="store_true",
        help="Write long-form read evidence TSV",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of cluster-level worker processes",
    )
    return parser


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Validate CLI arguments."""

    if arguments.k <= 0:
        raise ValueError("--k must be positive")
    if arguments.prefilter_pair_threshold < 0:
        raise ValueError("--prefilter-pair-threshold must be non-negative")
    if arguments.keep_top_pairs <= 0:
        raise ValueError("--keep-top-pairs must be positive")
    if arguments.threads <= 0:
        raise ValueError("--threads must be positive")


def main() -> None:
    """CLI entry point."""

    arguments = build_argument_parser().parse_args()
    validate_arguments(arguments)
    Path(arguments.output_directory).mkdir(parents=True, exist_ok=True)

    fasta_paths = discover_fasta_paths(arguments.fasta_path)
    if not fasta_paths:
        raise ValueError(f"No FASTA files found from input: {arguments.fasta_path}")

    if arguments.threads == 1:
        for fasta_path in fasta_paths:
            process_cluster(fasta_path, arguments)
    else:
        tasks = [(fasta_path, arguments) for fasta_path in fasta_paths]
        with ProcessPoolExecutor(max_workers=arguments.threads) as executor:
            list(executor.map(process_cluster_worker, tasks))


if __name__ == "__main__":
    main()
