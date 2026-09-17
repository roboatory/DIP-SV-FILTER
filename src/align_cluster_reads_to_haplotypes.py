#!/usr/bin/env python3
"""
Align cluster reads to retained pseudo-haplotype FASTA files.

This helper consumes the output directory produced by
``src/filter_contig_pairs.py``. For each cluster directory, it expects:

    cluster_reads.fasta
    hap_fastas/<hap_id>.fasta

For every retained haplotype FASTA, reads are aligned with minimap2, converted
to a coordinate-sorted BAM, and indexed. Output is written as:

    hap_bams/<hap_id>.bam
    hap_bams/<hap_id>.bam.bai
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AlignmentTask:
    """One read-to-haplotype alignment job."""

    cluster_name: str
    reads_fasta: Path
    hap_fasta: Path
    output_bam: Path


@dataclass(frozen=True)
class AlignmentResult:
    """Summary of one completed alignment job."""

    cluster_name: str
    hap_id: str
    output_bam: Path
    skipped: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Align each cluster's extracted reads to each retained haplotype "
            "FASTA, then write sorted/indexed BAMs. Total minimap2 CPU threads "
            "used at once is approximately --jobs * --minimap2-threads."
        )
    )
    # fmt: off
    parser.add_argument("cluster_dir", help="Directory containing per-cluster k-mer prefilter outputs. Each cluster should contain cluster_reads.fasta and hap_fastas/*.fasta.")
    parser.add_argument("--preset", default="map-hifi", help="minimap2 -x preset. Default: map-hifi. Short aliases hifi and ont are accepted and converted to map-hifi and map-ont.")
    parser.add_argument("--jobs", type=int, default=8, help="Number of minimap2 alignment jobs to run concurrently. Total minimap2 CPU threads/processes is --jobs * --minimap2-threads. Default: 8.")
    parser.add_argument("--minimap2-threads", type=int, default=4, help="Threads passed to each minimap2 process with -t. Total minimap2 CPU threads/processes is --jobs * --minimap2-threads. Default: 4.")
    parser.add_argument("--output-subdir", default="hap_bams", help="Subdirectory created under each cluster for BAM output. Default: hap_bams.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate BAMs even when both BAM and BAI already exist.")
    # fmt: on
    return parser.parse_args()


def normalize_preset(
    preset: str,
) -> str:
    """Normalize common short read-type aliases to minimap2 preset names."""

    aliases = {
        "hifi": "map-hifi",
        "ccs": "map-hifi",
        "ont": "map-ont",
        "pb": "map-pb",
    }
    return aliases.get(preset, preset)


def discover_cluster_dirs(
    root: Path,
) -> list[Path]:
    """Find cluster directories with the expected k-mer prefilter outputs."""

    if not root.is_dir():
        raise NotADirectoryError(f"Cluster output root is not a directory: {root}")

    clusters = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "cluster_reads.fasta").is_file() and (path / "hap_fastas").is_dir():
            clusters.append(path)

    return clusters


def discover_tasks(
    cluster_root: Path,
    output_subdir: str,
) -> list[AlignmentTask]:
    """Create one alignment task per cluster/haplotype FASTA."""

    tasks: list[AlignmentTask] = []
    for cluster_dir in discover_cluster_dirs(cluster_root):
        reads_fasta = cluster_dir / "cluster_reads.fasta"
        hap_fasta_dir = cluster_dir / "hap_fastas"
        bam_dir = cluster_dir / output_subdir

        for hap_fasta in sorted(hap_fasta_dir.glob("*.fasta")):
            output_bam = bam_dir / f"{hap_fasta.stem}.bam"
            tasks.append(
                AlignmentTask(
                    cluster_name=cluster_dir.name,
                    reads_fasta=reads_fasta,
                    hap_fasta=hap_fasta,
                    output_bam=output_bam,
                )
            )

    return tasks


def bam_is_complete(
    path: Path,
) -> bool:
    """Return True if a BAM and its index both exist and are non-empty."""

    bai_path = Path(str(path) + ".bai")
    return (
        path.is_file()
        and path.stat().st_size > 0
        and bai_path.is_file()
        and bai_path.stat().st_size > 0
    )


def run_command(
    command: list[str],
) -> None:
    """Run a command and raise a clear error if it fails."""

    subprocess.run(command, check=True)


def run_alignment_pipe(
    minimap2_command: list[str],
    sort_command: list[str],
) -> None:
    """Pipe minimap2 SAM output directly into samtools sort."""

    minimap2_process = subprocess.Popen(minimap2_command, stdout=subprocess.PIPE)
    try:
        sort_process = subprocess.Popen(sort_command, stdin=minimap2_process.stdout)
        if minimap2_process.stdout is not None:
            minimap2_process.stdout.close()

        sort_returncode = sort_process.wait()
        minimap2_returncode = minimap2_process.wait()
    except BaseException:
        minimap2_process.kill()
        if "sort_process" in locals():
            sort_process.kill()
        raise

    if minimap2_returncode != 0:
        raise subprocess.CalledProcessError(minimap2_returncode, minimap2_command)
    if sort_returncode != 0:
        raise subprocess.CalledProcessError(sort_returncode, sort_command)


def align_one_task(
    task: AlignmentTask,
    preset: str,
    minimap2_threads: int,
    overwrite: bool,
) -> AlignmentResult:
    """Run minimap2 for one haplotype FASTA and create sorted/indexed BAM."""

    if not overwrite and bam_is_complete(task.output_bam):
        return AlignmentResult(
            task.cluster_name, task.hap_fasta.stem, task.output_bam, skipped=True
        )

    task.output_bam.parent.mkdir(parents=True, exist_ok=True)
    task.output_bam.unlink(missing_ok=True)
    Path(str(task.output_bam) + ".bai").unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".align_cluster_reads_",
        dir=task.output_bam.parent,
    ) as temp_dir:
        temp_path = Path(temp_dir)
        sorted_bam = temp_path / "alignment.sorted.bam"

        run_alignment_pipe(
            [
                "minimap2",
                "--MD",
                "-Y",
                "-L",
                "-a",
                "-x",
                preset,
                "-t",
                str(minimap2_threads),
                str(task.hap_fasta),
                str(task.reads_fasta),
            ],
            ["samtools", "sort", "-o", str(sorted_bam)],
        )
        run_command(["samtools", "index", str(sorted_bam)])

        sorted_bam.replace(task.output_bam)
        Path(str(sorted_bam) + ".bai").replace(Path(str(task.output_bam) + ".bai"))

    return AlignmentResult(
        task.cluster_name, task.hap_fasta.stem, task.output_bam, skipped=False
    )


def run_tasks(
    tasks: Iterable[AlignmentTask],
    preset: str,
    jobs: int,
    minimap2_threads: int,
    overwrite: bool,
) -> list[AlignmentResult]:
    """Run all alignment tasks with bounded process-level concurrency."""

    task_list = list(tasks)
    if jobs == 1:
        return [
            align_one_task(task, preset, minimap2_threads, overwrite)
            for task in task_list
        ]

    results: list[AlignmentResult] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_task = {
            executor.submit(
                align_one_task, task, preset, minimap2_threads, overwrite
            ): task
            for task in task_list
        }
        for future in concurrent.futures.as_completed(future_to_task):
            results.append(future.result())

    return results


def main() -> None:
    """Entry point."""

    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if args.minimap2_threads < 1:
        raise ValueError("--minimap2-threads must be at least 1")

    cluster_root = Path(args.cluster_dir).resolve()
    preset = normalize_preset(args.preset)
    tasks = discover_tasks(cluster_root, args.output_subdir)

    print(f"Cluster root: {cluster_root}")
    print(f"Alignment tasks: {len(tasks)}")
    print(f"minimap2 preset: {preset}")
    print(f"jobs: {args.jobs}")
    print(f"minimap2 threads per job: {args.minimap2_threads}")
    print(f"maximum concurrent minimap2 threads: {args.jobs * args.minimap2_threads}")

    if not tasks:
        print("No alignment tasks found.")
        return

    results = run_tasks(tasks, preset, args.jobs, args.minimap2_threads, args.overwrite)
    completed = sum(not result.skipped for result in results)
    skipped = sum(result.skipped for result in results)
    print(f"Completed BAMs: {completed}")
    print(f"Skipped existing BAMs: {skipped}")


if __name__ == "__main__":
    main()
