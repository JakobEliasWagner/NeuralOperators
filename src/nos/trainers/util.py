import json  # noqa: D100
import pathlib
import time

import torch
from continuiti.data import OperatorDataset
from continuiti.operators import Operator


def save_checkpoint(
    operator: Operator,
    val_loss: float,
    train_loss: float,
    epoch: int,
    start: float,
    batch_size: int,
    train_set: OperatorDataset,
    val_set: OperatorDataset,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """Save operator weights and meta-information."""
    checkpoint = {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "epoch": epoch,
        "Time_trained": time.time() - start,
        "batch_size": batch_size,
        "train_size": len(train_set),
        "val_size": len(val_set),
    }

    torch.save(operator, out_dir.joinpath("operator.pt"))
    checkpoint_path = out_dir.joinpath("checkpoint.json")
    with checkpoint_path.open("w") as file_handle:
        json.dump(checkpoint, file_handle)

    return out_dir
