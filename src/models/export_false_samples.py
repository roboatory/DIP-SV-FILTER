# For False sample analysis


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from .inference import get_default_device_name, load_model_from_checkpoint
    from .train import (
        EXPECTED_INPUT_SHAPE,
        load_labels,
        resolve_labels_file_path,
    )
except ImportError:
    from inference import get_default_device_name, load_model_from_checkpoint
    from train import EXPECTED_INPUT_SHAPE, load_labels, resolve_labels_file_path


class SVFalseSampleDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(
        self,
        split_directory: Path,
        labels_file_path: Path,
    ) -> None:
        """Initialize and validate the dataset or accumulator."""

        if not split_directory.exists():
            raise FileNotFoundError(f"Split directory not found: {split_directory}")
        if not split_directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {split_directory}")

        self.labels = load_labels(labels_file_path)
        self.samples = sorted(split_directory.glob("*.npy"))
        if not self.samples:
            raise ValueError(
                f"No .npy files found in split directory: {split_directory}"
            )

        missing_labels = [
            path.name for path in self.samples if path.name not in self.labels
        ]
        if missing_labels:
            preview = ", ".join(missing_labels[:5])
            raise ValueError(
                f"Missing labels for {len(missing_labels)} files in {split_directory}: {preview}"
            )

    def __len__(
        self,
    ) -> int:
        """Return the number of feature windows."""

        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, str]:
        """Load a feature window and its associated metadata."""

        sample_path = self.samples[index]
        features = np.load(sample_path).astype(np.float32, copy=False)
        if features.shape != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"{sample_path} has shape {features.shape}, expected {EXPECTED_INPUT_SHAPE}"
            )
        labels = self.labels[sample_path.name]
        return torch.from_numpy(features), labels.clone(), sample_path.name


def create_false_sample_dataloader(
    split_directory: Path,
    labels_file_path: Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader[tuple[Tensor, Tensor, tuple[str, ...]]]:
    """Create a dataloader for labeled prediction diagnostics."""

    dataset = SVFalseSampleDataset(
        split_directory=split_directory,
        labels_file_path=labels_file_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def format_binary_vector(
    values: Tensor,
) -> str:
    """Format a binary tensor as comma-separated values."""

    return ",".join(str(int(value)) for value in values.tolist())


def resolve_split_labels_file_path(
    split_name: str,
    split_directory: Path,
    labels_file_path: Path | None,
) -> Path:
    """Resolve the labels file used for this split."""

    return resolve_labels_file_path(
        split_directory=split_directory,
        labels_file_path=labels_file_path,
    )


def export_false_samples_for_split(
    model: torch.nn.Module,
    device: torch.device,
    split_name: str,
    split_directory: Path,
    labels_file_path: Path | None,
    output_directory: Path,
    batch_size: int,
    num_workers: int,
    prediction_threshold: float,
) -> Path:
    """Write predictions and labels for misclassified windows."""

    resolved_labels_file_path = resolve_split_labels_file_path(
        split_name=split_name,
        split_directory=split_directory,
        labels_file_path=labels_file_path,
    )
    dataloader = create_false_sample_dataloader(
        split_directory=split_directory,
        labels_file_path=resolved_labels_file_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    output_file_path = output_directory / f"False_Samples_{split_name}.tsv"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    false_sample_count = 0
    sample_count = 0
    with output_file_path.open("w", encoding="utf-8") as handle:
        handle.write("file_name\tprediction\tground_truth_label\n")

        with torch.no_grad():
            for features, labels, file_names in dataloader:
                features = features.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                labels = labels.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                logits = model(features)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= prediction_threshold).to(
                    dtype=torch.int64
                )
                targets = labels.to(dtype=torch.int64)
                mismatched_samples = (predictions != targets).any(dim=1)

                predictions_cpu = predictions.cpu()
                targets_cpu = targets.cpu()

                for index, has_mismatch in enumerate(mismatched_samples.cpu().tolist()):
                    sample_count += 1
                    if not has_mismatch:
                        continue
                    false_sample_count += 1
                    handle.write(
                        f"{file_names[index]}\t"
                        f"{format_binary_vector(predictions_cpu[index])}\t"
                        f"{format_binary_vector(targets_cpu[index])}\n"
                    )

    print(
        f"{split_name}: wrote {false_sample_count} false samples out of {sample_count} "
        f"to {output_file_path}"
    )
    return output_file_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Export misclassified samples for train/validation/test splits."
    )

    parser.add_argument(
        "--checkpoint_file_path",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--train_directory",
        type=Path,
        required=True,
        help="Directory of training .npy files.",
    )
    parser.add_argument(
        "--validation_directory",
        type=Path,
        required=True,
        help="Directory of validation .npy files.",
    )
    parser.add_argument(
        "--test_directory",
        type=Path,
        required=True,
        help="Directory of test .npy files.",
    )
    parser.add_argument(
        "--labels_file_path",
        type=Path,
        default=None,
        help="Optional shared labels file path. If omitted, the script resolves labels per split.",
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        required=True,
        help="Directory to write TSV outputs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device_name(),
        help="Inference device, for example cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--prediction_threshold",
        type=float,
        default=0.5,
        help="Threshold for converting probabilities into binary predictions.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line workflow."""

    arguments = parse_args()
    device = torch.device(arguments.device)
    model = load_model_from_checkpoint(
        checkpoint_file_path=arguments.checkpoint_file_path,
        device=device,
    )

    split_configurations = [
        ("train", arguments.train_directory),
        ("validation", arguments.validation_directory),
        ("test", arguments.test_directory),
    ]
    for split_name, split_directory in split_configurations:
        export_false_samples_for_split(
            model=model,
            device=device,
            split_name=split_name,
            split_directory=split_directory,
            labels_file_path=arguments.labels_file_path,
            output_directory=arguments.output_directory,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            prediction_threshold=arguments.prediction_threshold,
        )


if __name__ == "__main__":
    main()
