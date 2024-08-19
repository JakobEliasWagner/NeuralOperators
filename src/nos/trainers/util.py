import json
import pathlib
import time

import torch


def save_checkpoint(
    operator, val_loss, train_loss, epoch, start, batch_size, train_set, val_set, out_dir: pathlib.Path = None
) -> pathlib.Path:
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
    with open(checkpoint_path, "w") as file_handle:
        json.dump(checkpoint, file_handle)

    return out_dir
