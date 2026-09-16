from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

try:
    from .architecture import SVHunterModel
except ImportError:
    from architecture import SVHunterModel


EXPECTED_INPUT_SHAPE = (2000, 9)
EXPECTED_LABEL_LENGTH = 10


def get_default_device_name() -> str:
    """Return the best available torch device name."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(
    seed: int,
) -> None:
    """Seed Python, NumPy, and torch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Divide two values and return zero for a zero denominator."""

    return numerator / denominator if denominator else 0.0


def serialize_arguments(
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Convert argparse values into JSON-serializable strings when needed."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(arguments).items()
    }


def prefix_metrics(
    prefix: str,
    metrics: EpochMetrics,
) -> dict[str, float]:
    """Prefix metric keys for grouped logging."""

    return {f"{prefix}/{key}": value for key, value in asdict(metrics).items()}


def parse_label_vector(
    label_text: str,
) -> Tensor:
    """Parse one comma-separated binary label vector."""

    label_parts = [label_part.strip() for label_part in label_text.split(",")]
    if len(label_parts) != EXPECTED_LABEL_LENGTH:
        raise ValueError(
            f"Expected {EXPECTED_LABEL_LENGTH} labels per example, got {len(label_parts)}"
        )

    values: list[float] = []
    for label_part in label_parts:
        if label_part not in {"0", "1"}:
            raise ValueError(f"Labels must be binary 0/1 values, got {label_part!r}")
        values.append(float(label_part))
    return torch.tensor(values, dtype=torch.float32)


def load_labels(
    labels_file_path: Path,
) -> dict[str, Tensor]:
    """Load labels from a labels.txt file."""

    if not labels_file_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file_path}")

    labels: dict[str, Tensor] = {}
    with labels_file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            line_parts = line.split("\t")
            if len(line_parts) != 2:
                raise ValueError(
                    f"Malformed labels.txt line {line_number}: expected <file>\\t<v0,...,v9>"
                )
            file_name, label_text = line_parts
            file_basename = Path(file_name).name
            label_vector = parse_label_vector(label_text)
            if file_basename in labels:
                if not torch.equal(labels[file_basename], label_vector):
                    raise ValueError(f"Conflicting label entry for {file_basename}")
                continue
            labels[file_basename] = label_vector

    if not labels:
        raise ValueError("No labels were loaded from labels.txt")
    return labels


class SVWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        split_directory: Path,
        labels: dict[str, Tensor],
        max_samples: int | None = None,
    ) -> None:
        """Validate feature files and associate them with labels."""

        if not split_directory.exists():
            raise FileNotFoundError(f"Split directory not found: {split_directory}")
        if not split_directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {split_directory}")

        files = sorted(split_directory.glob("*.npy"))
        if not files:
            raise ValueError(
                f"No .npy files found in split directory: {split_directory}"
            )
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("--maximum-samples must be a positive integer")
            files = files[:max_samples]

        missing_labels = [path.name for path in files if path.name not in labels]
        if missing_labels:
            preview = ", ".join(missing_labels[:5])
            raise ValueError(
                f"Missing labels for {len(missing_labels)} files in {split_directory}: {preview}"
            )

        self.samples = [path for path in files if path.name in labels]
        self.labels = labels

        for sample_path in self.samples:
            array = np.load(sample_path, mmap_mode="r")
            if array.shape != EXPECTED_INPUT_SHAPE:
                raise ValueError(
                    f"{sample_path} has shape {array.shape}, expected {EXPECTED_INPUT_SHAPE}"
                )

    def __len__(
        self,
    ) -> int:
        """Return the number of labeled feature windows."""

        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:
        """Load one feature window and a copy of its labels."""

        sample_path = self.samples[index]
        features = np.load(sample_path).astype(np.float32, copy=False)
        labels = self.labels[sample_path.name]
        return torch.from_numpy(features), labels.clone()


@dataclass
class EpochMetrics:
    loss: float
    elementwise_precision: float
    elementwise_recall: float
    elementwise_f1: float
    elementwise_accuracy: float
    exact_match_accuracy: float
    any_sv_precision: float
    any_sv_recall: float
    any_sv_f1: float


class MetricsAccumulator:
    def __init__(
        self,
    ) -> None:
        """Initialize loss and prediction counters."""

        self.loss_sum = 0.0
        self.sample_count = 0
        self.elementwise_true_positives = 0
        self.elementwise_false_positives = 0
        self.elementwise_false_negatives = 0
        self.elementwise_correct = 0
        self.elementwise_total = 0
        self.exact_match_count = 0
        self.any_sv_true_positives = 0
        self.any_sv_false_positives = 0
        self.any_sv_false_negatives = 0

    def update(
        self,
        loss: float,
        logits: Tensor,
        labels: Tensor,
    ) -> None:
        """Accumulate batch loss and prediction counts."""

        batch_size = labels.shape[0]
        self.loss_sum += loss * batch_size
        self.sample_count += batch_size

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).to(dtype=torch.int64)
        targets = labels.to(dtype=torch.int64)

        self.elementwise_true_positives += int(
            ((predictions == 1) & (targets == 1)).sum().item()
        )
        self.elementwise_false_positives += int(
            ((predictions == 1) & (targets == 0)).sum().item()
        )
        self.elementwise_false_negatives += int(
            ((predictions == 0) & (targets == 1)).sum().item()
        )
        self.elementwise_correct += int((predictions == targets).sum().item())
        self.elementwise_total += int(targets.numel())
        self.exact_match_count += int((predictions == targets).all(dim=1).sum().item())

        any_predictions = predictions.any(dim=1).to(dtype=torch.int64)
        any_targets = targets.any(dim=1).to(dtype=torch.int64)
        self.any_sv_true_positives += int(
            ((any_predictions == 1) & (any_targets == 1)).sum().item()
        )
        self.any_sv_false_positives += int(
            ((any_predictions == 1) & (any_targets == 0)).sum().item()
        )
        self.any_sv_false_negatives += int(
            ((any_predictions == 0) & (any_targets == 1)).sum().item()
        )

    def compute(
        self,
    ) -> EpochMetrics:
        """Compute aggregate epoch metrics."""

        elementwise_precision = safe_divide(
            self.elementwise_true_positives,
            self.elementwise_true_positives + self.elementwise_false_positives,
        )
        elementwise_recall = safe_divide(
            self.elementwise_true_positives,
            self.elementwise_true_positives + self.elementwise_false_negatives,
        )
        elementwise_f1 = safe_divide(
            2 * elementwise_precision * elementwise_recall,
            elementwise_precision + elementwise_recall,
        )
        any_sv_precision = safe_divide(
            self.any_sv_true_positives,
            self.any_sv_true_positives + self.any_sv_false_positives,
        )
        any_sv_recall = safe_divide(
            self.any_sv_true_positives,
            self.any_sv_true_positives + self.any_sv_false_negatives,
        )
        any_sv_f1 = safe_divide(
            2 * any_sv_precision * any_sv_recall,
            any_sv_precision + any_sv_recall,
        )

        return EpochMetrics(
            loss=safe_divide(self.loss_sum, self.sample_count),
            elementwise_precision=elementwise_precision,
            elementwise_recall=elementwise_recall,
            elementwise_f1=elementwise_f1,
            elementwise_accuracy=safe_divide(
                self.elementwise_correct, self.elementwise_total
            ),
            exact_match_accuracy=safe_divide(self.exact_match_count, self.sample_count),
            any_sv_precision=any_sv_precision,
            any_sv_recall=any_sv_recall,
            any_sv_f1=any_sv_f1,
        )


def create_dataloader(
    split_directory: Path,
    batch_size: int,
    shuffle: bool,
    worker_count: int,
    max_samples: int | None = None,
    labels_file_path: Path | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create a labeled feature-window dataloader."""

    resolved_labels_file_path = resolve_labels_file_path(
        split_directory=split_directory,
        labels_file_path=labels_file_path,
    )
    labels = load_labels(resolved_labels_file_path)
    dataset = SVWindowDataset(
        split_directory=split_directory,
        labels=labels,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_labels_file_path(
    split_directory: Path,
    labels_file_path: Path | None = None,
) -> Path:
    """Resolve labels.txt from a split directory or its parent."""

    if labels_file_path is not None:
        return labels_file_path

    candidate_paths = [
        split_directory / "labels.txt",
        split_directory.parent / "labels.txt",
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"Could not find labels.txt for split directory {split_directory}"
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> EpochMetrics:
    """Run one training or evaluation epoch."""

    training = optimizer is not None
    model.train(training)
    accumulator = MetricsAccumulator()

    for features, labels in dataloader:
        features = features.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        accumulator.update(loss.item(), logits.detach(), labels.detach())

    return accumulator.compute()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    metrics: EpochMetrics,
    arguments: argparse.Namespace,
) -> None:
    """Save a model checkpoint."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": asdict(metrics),
        "args": serialize_arguments(arguments),
    }
    torch.save(checkpoint, path)


def write_json(
    output_file_path: Path,
    payload: Any,
) -> None:
    """Write a JSON payload to disk."""

    with output_file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def initialize_wandb(
    arguments: argparse.Namespace,
) -> wandb.sdk.wandb_run.Run | None:
    """Initialize a Weights & Biases run unless disabled."""

    if arguments.wandb_mode == "disabled":
        return None

    init_kwargs = {
        "project": arguments.wandb_project,
        "name": arguments.wandb_run_name,
        "mode": arguments.wandb_mode,
        "anonymous": "allow",
        "config": serialize_arguments(arguments),
    }
    try:
        run = wandb.init(**init_kwargs)
    except wandb.errors.UsageError:
        if arguments.wandb_mode != "online":
            raise
        print("wandb online init failed; retrying in offline mode")
        run = wandb.init(**{**init_kwargs, "mode": "offline"})

    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    return run


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train the adapted SVHunter model.")
    parser.add_argument(
        "--train-directory",
        dest="train_directory",
        type=Path,
        required=True,
        help="Directory of training .npy files.",
    )
    parser.add_argument(
        "--validation-directory",
        dest="validation_directory",
        type=Path,
        required=True,
        help="Directory of validation .npy files.",
    )
    parser.add_argument(
        "--test-directory",
        dest="test_directory",
        type=Path,
        required=True,
        help="Directory of test .npy files.",
    )
    parser.add_argument(
        "--output-directory",
        dest="output_directory",
        type=Path,
        required=True,
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--labels-file-path",
        type=Path,
        default=None,
        help="Optional shared labels.txt; otherwise resolve labels per split.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine", "constant"),
        default="cosine",
        help="Learning-rate schedule. Default: cosine.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=64, help="Batch size."
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=2e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-3,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--worker-count",
        dest="worker_count",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--maximum-samples",
        dest="maximum_samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples loaded from each split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device_name(),
        help="Training device, for example cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--wandb-mode",
        dest="wandb_mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode.",
    )
    parser.add_argument(
        "--wandb-project",
        dest="wandb_project",
        type=str,
        default="structural-variant-detection",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        dest="wandb_run_name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )

    return parser.parse_args()


def train(
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Train the model and return run summary metrics."""

    set_seed(arguments.seed)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    wandb_run = initialize_wandb(arguments)

    try:
        train_loader = create_dataloader(
            split_directory=arguments.train_directory,
            batch_size=arguments.batch_size,
            shuffle=True,
            worker_count=arguments.worker_count,
            max_samples=arguments.maximum_samples,
            labels_file_path=arguments.labels_file_path,
        )
        validation_loader = create_dataloader(
            split_directory=arguments.validation_directory,
            batch_size=arguments.batch_size,
            shuffle=False,
            worker_count=arguments.worker_count,
            max_samples=arguments.maximum_samples,
            labels_file_path=arguments.labels_file_path,
        )
        test_loader = create_dataloader(
            split_directory=arguments.test_directory,
            batch_size=arguments.batch_size,
            shuffle=False,
            worker_count=arguments.worker_count,
            max_samples=arguments.maximum_samples,
            labels_file_path=arguments.labels_file_path,
        )

        device = torch.device(arguments.device)
        model = SVHunterModel().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
        )
        scheduler = (
            CosineAnnealingLR(optimizer, T_max=arguments.epochs, eta_min=1e-6)
            if arguments.lr_scheduler == "cosine"
            else None
        )

        best_validation_f1 = float("-inf")
        history: list[dict[str, Any]] = []

        for epoch in range(1, arguments.epochs + 1):
            train_metrics = run_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )
            validation_metrics = run_epoch(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
            )

            if scheduler is not None:
                scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "val": asdict(validation_metrics),
            }
            history.append(epoch_record)
            print(
                f"epoch={epoch} "
                f"train_loss={train_metrics.loss:.4f} "
                f"val_loss={validation_metrics.loss:.4f} "
                f"val_elementwise_f1={validation_metrics.elementwise_f1:.4f}"
            )

            if validation_metrics.elementwise_f1 > best_validation_f1:
                best_validation_f1 = validation_metrics.elementwise_f1
                save_checkpoint(
                    path=arguments.output_directory / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=validation_metrics,
                    arguments=arguments,
                )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        **prefix_metrics("train", train_metrics),
                        **prefix_metrics("val", validation_metrics),
                        "best/validation_elementwise_f1": best_validation_f1,
                    }
                )

        final_validation_metrics = run_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        save_checkpoint(
            path=arguments.output_directory / "final_model.pt",
            model=model,
            optimizer=optimizer,
            epoch=arguments.epochs,
            metrics=final_validation_metrics,
            arguments=arguments,
        )

        best_checkpoint = torch.load(
            arguments.output_directory / "best_model.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = run_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        results = {
            "best_validation_elementwise_f1": best_validation_f1,
            "test_metrics": asdict(test_metrics),
            "history": history,
        }
        write_json(arguments.output_directory / "history.json", history)
        write_json(
            arguments.output_directory / "test_metrics.json", asdict(test_metrics)
        )
        write_json(arguments.output_directory / "run_summary.json", results)

        if wandb_run is not None:
            wandb_run.log(
                {"epoch": arguments.epochs, **prefix_metrics("test", test_metrics)}
            )
            wandb_run.summary["best_validation_elementwise_f1"] = best_validation_f1
            for key, value in prefix_metrics("test", test_metrics).items():
                wandb_run.summary[key] = value

        print(
            "test "
            f"loss={test_metrics.loss:.4f} "
            f"elementwise_f1={test_metrics.elementwise_f1:.4f} "
            f"exact_match_accuracy={test_metrics.exact_match_accuracy:.4f} "
            f"any_sv_f1={test_metrics.any_sv_f1:.4f}"
        )
        return results
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    """Run the training command-line interface."""

    arguments = parse_arguments()
    train(arguments)


if __name__ == "__main__":
    main()
