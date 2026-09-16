from pathlib import Path

import torch
from torch import Tensor, nn

from models.architecture import SVHunterModel

EXPECTED_INPUT_SHAPE = (2000, 9)
EXPECTED_LABEL_LENGTH = 10


def get_default_device_name() -> str:
    """Return the best available torch device name."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def format_binary_prediction_vector(
    values: Tensor,
) -> str:
    """Format binary predictions as a comma-separated vector."""

    return ",".join(str(int(value)) for value in values.tolist())


def load_model_from_checkpoint(
    checkpoint_file_path: Path,
    device: torch.device,
) -> nn.Module:
    """Strictly load the checkpoint into the local architecture."""

    if not checkpoint_file_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file_path}")
    checkpoint = torch.load(
        checkpoint_file_path, map_location=device, weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain model_state_dict: {checkpoint_file_path}"
        )

    model = SVHunterModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model
