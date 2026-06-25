from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from architecture import SVHunterModel


EXPECTED_INPUT_SHAPE = (2000, 9)


def get_default_device_name() -> str:
    """Return the best available torch device name."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SVInferenceDataset(Dataset[tuple[Tensor, str]]):
    def __init__(self, split_directory: Path) -> None:
        if not split_directory.exists():
            raise FileNotFoundError(f"Split directory not found: {split_directory}")
        if not split_directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {split_directory}")

        self.samples = sorted(split_directory.glob("*.npy"))
        if not self.samples:
            raise ValueError(
                f"No .npy files found in split directory: {split_directory}"
            )

        for sample_path in self.samples:
            array = np.load(sample_path, mmap_mode="r")
            if array.shape != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"{sample_path} has shape {array.shape}, expected {EXPECTED_INPUT_SHAPE}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        sample_path = self.samples[index]
        features = np.load(sample_path).astype(np.float32, copy=False)
        return torch.from_numpy(features), sample_path.name


def create_inference_dataloader(
    split_directory: Path,
    batch_size: int,
    worker_count: int,
) -> DataLoader[tuple[Tensor, list[str]]]:
    """Create a deterministic inference dataloader."""

    dataset = SVInferenceDataset(split_directory=split_directory)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )


def format_prediction_vector(values: Tensor, decimal_places: int = 6) -> str:
    """Format probability predictions as a comma-separated vector."""

    return ",".join(f"{value:.{decimal_places}f}" for value in values.tolist())


def format_binary_prediction_vector(values: Tensor) -> str:
    """Format binary predictions as a comma-separated vector."""

    return ",".join(str(int(value)) for value in values.tolist())


def load_model_from_checkpoint(
    checkpoint_file_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load an SVHunter model from a checkpoint file."""

    checkpoint = torch.load(
        checkpoint_file_path,
        map_location=device,
        weights_only=False,
    )
    model = SVHunterModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def write_inference_file(
    output_file_path: Path,
    file_names: list[str],
    probability_predictions: list[Tensor],
    binary_predictions: list[Tensor],
) -> None:
    """Write inference predictions to a TSV file."""

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open("w", encoding="utf-8") as handle:
        handle.write("file_name\tpredicted_probabilities\tpredicted_labels\n")
        for file_name, probability_prediction, binary_prediction in zip(
            file_names,
            probability_predictions,
            binary_predictions,
            strict=True,
        ):
            handle.write(
                f"{file_name}\t"
                f"{format_prediction_vector(probability_prediction)}\t"
                f"{format_binary_prediction_vector(binary_prediction)}\n"
            )


def run_inference(
    checkpoint_file_path: Path,
    split_directory: Path,
    output_file_path: Path,
    batch_size: int = 64,
    worker_count: int = 0,
    device_name: str = "cpu",
    prediction_threshold: float = 0.5,
) -> Path:
    """Run checkpoint inference over a directory of feature matrices."""

    device = torch.device(device_name)
    model = load_model_from_checkpoint(
        checkpoint_file_path=checkpoint_file_path,
        device=device,
    )
    dataloader = create_inference_dataloader(
        split_directory=split_directory,
        batch_size=batch_size,
        worker_count=worker_count,
    )

    file_names: list[str] = []
    probability_predictions: list[Tensor] = []
    binary_predictions: list[Tensor] = []

    with torch.no_grad():
        for features, batch_file_names in dataloader:
            features = features.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            logits = model(features)
            probabilities = torch.sigmoid(logits).cpu()
            binary_outputs = (probabilities >= prediction_threshold).to(
                dtype=torch.int64
            )

            file_names.extend(batch_file_names)
            probability_predictions.extend(probabilities)
            binary_predictions.extend(binary_outputs)

    write_inference_file(
        output_file_path=output_file_path,
        file_names=file_names,
        probability_predictions=probability_predictions,
        binary_predictions=binary_predictions,
    )
    return output_file_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    # fmt: off
    parser = argparse.ArgumentParser(description="Run SVHunter inference.")
    parser.add_argument("--checkpoint-file-path", dest="checkpoint_file_path", type=Path, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--split-directory", dest="split_directory", type=Path, required=True, help="Directory containing .npy feature windows for inference.")
    parser.add_argument("--output-file-path", dest="output_file_path", type=Path, required=True, help="Path to the output TSV file.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--worker-count", dest="worker_count", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--device", type=str, default=get_default_device_name(), help="Inference device, for example cpu, mps, or cuda.")
    parser.add_argument("--prediction-threshold", dest="prediction_threshold", type=float, default=0.5, help="Threshold for converting probabilities into binary predictions.")
    # fmt: on

    return parser.parse_args()


def main() -> None:
    """Run the inference command-line interface."""

    arguments = parse_arguments()
    output_file_path = run_inference(
        checkpoint_file_path=arguments.checkpoint_file_path,
        split_directory=arguments.split_directory,
        output_file_path=arguments.output_file_path,
        batch_size=arguments.batch_size,
        worker_count=arguments.worker_count,
        device_name=arguments.device,
        prediction_threshold=arguments.prediction_threshold,
    )
    print(output_file_path)


if __name__ == "__main__":
    main()
