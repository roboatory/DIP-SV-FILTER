from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
# from torch.optim import Adam
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    from .architecture import SVHunterModel
except ImportError:
    from architecture import SVHunterModel


EXPECTED_INPUT_SHAPE = (2000, 9)
EXPECTED_LABEL_LENGTH = 10


def get_default_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def parse_label_vector(label_text: str) -> Tensor:
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


def load_labels(labels_file_path: Path) -> dict[str, Tensor]:
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
    def __init__(self, split_directory: Path, labels: dict[str, Tensor]) -> None:
        if not split_directory.exists():
            raise FileNotFoundError(f"Split directory not found: {split_directory}")
        if not split_directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {split_directory}")

        files = sorted(split_directory.glob("*.npy"))
        if not files:
            raise ValueError(
                f"No .npy files found in split directory: {split_directory}"
            )

        missing_labels = [path.name for path in files if path.name not in labels]
        if missing_labels:
            preview = ", ".join(missing_labels[:5])
            raise ValueError(
                f"Missing labels for {len(missing_labels)} files in {split_directory}: {preview}"
            )

        self.samples = [path for path in files if path.name in labels]
        self.labels = labels

        ## took a long time to check all files, so skipping for now. Relying on the fact that the data generation script is correct and consistent with labels.txt
        # for sample_path in self.samples:
        #     array = np.load(sample_path, mmap_mode="r")
        #     if array.shape != EXPECTED_INPUT_SHAPE:
        #         raise ValueError(
        #             f"{sample_path} has shape {array.shape}, expected {EXPECTED_INPUT_SHAPE}"
        #         )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
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
    def __init__(self) -> None:
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

    def update(self, loss: float, logits: Tensor, labels: Tensor) -> None:
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

    def compute(self) -> EpochMetrics:
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
    labels_file_path: Path | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    resolved_labels_file_path = resolve_labels_file_path(
        split_directory=split_directory,
        labels_file_path=labels_file_path,
    )
    labels = load_labels(resolved_labels_file_path)
    dataset = SVWindowDataset(split_directory=split_directory, labels=labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_labels_file_path(
    split_directory: Path,
    labels_file_path: Path | None,
) -> Path:
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
    # optimizer: Adam | None = None,
    optimizer: AdamW | None = None,
) -> EpochMetrics:
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
    # optimizer: Adam,
    optimizer: AdamW,
    epoch: int,
    metrics: EpochMetrics,
    args: argparse.Namespace,
) -> None:
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": asdict(metrics),
        "args": serialized_args,
    }
    torch.save(checkpoint, path)


def write_json(output_file_path: Path, payload: Any) -> None:
    with output_file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the adapted SVHunter model.")

    # fmt: off
    parser.add_argument("--train_directory", dest="train_directory", type=Path, required=True, help="Directory of training .npy files.")
    parser.add_argument("--validation_directory", dest="validation_directory", type=Path, required=True, help="Directory of validation .npy files.")
    parser.add_argument("--test_directory", dest="test_directory", type=Path, required=True, help="Directory of test .npy files.")
    parser.add_argument("--output_directory", dest="output_directory", type=Path, required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--labels_file_path", dest="labels_file_path", type=Path, default=None, help="Optional path to a shared labels.txt. If omitted, labels.txt is resolved per split.")

    # parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    # parser.add_argument("--learning_rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Adam learning rate.")

    # parser.add_argument("--weight_decay", type=float, default=0, help="Adam weight decay.")\
    # parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes.")

    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Adam weight decay.")
    parser.add_argument("--num_workers", type=int, default=12, help="DataLoader worker processes.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=get_default_device_name(), help="Training device, for example cpu, mps, or cuda.")
    # fmt: on

    return parser.parse_args()


def train(arguments: argparse.Namespace) -> dict[str, Any]:
    set_seed(arguments.seed)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    train_loader = create_dataloader(
        split_directory=arguments.train_directory,
        labels_file_path=arguments.labels_file_path,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
    )
    validation_loader = create_dataloader(
        split_directory=arguments.validation_directory,
        labels_file_path=arguments.labels_file_path,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
    )
    test_loader = create_dataloader(
        split_directory=arguments.test_directory,
        labels_file_path=arguments.labels_file_path,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
    )

    device = torch.device(arguments.device)
    model = SVHunterModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = Adam(
    #     model.parameters(),
    #     lr=arguments.learning_rate,
    #     weight_decay=arguments.weight_decay,
    # )
    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
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
                args=arguments,
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
        args=arguments,
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
    write_json(arguments.output_directory / "test_metrics.json", asdict(test_metrics))
    write_json(arguments.output_directory / "run_summary.json", results)

    print(
        "test "
        f"loss={test_metrics.loss:.4f} "
        f"elementwise_f1={test_metrics.elementwise_f1:.4f} "
        f"exact_match_accuracy={test_metrics.exact_match_accuracy:.4f} "
        f"any_sv_f1={test_metrics.any_sv_f1:.4f}"
    )
    return results


def main() -> None:
    arguments = parse_args()
    train(arguments)


if __name__ == "__main__":
    main()
