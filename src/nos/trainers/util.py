import json
import pathlib
import time

from nos.operators import (
    serialize,
)


def save_checkpoint(
    operator, val_loss, train_loss, epoch, start, batch_size, train_set, val_set, out_dir: pathlib.Path = None
) -> pathlib.Path:
    checkpoint = {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "epoch": epoch,
        "Time_trained": (time.time_ns() - start) * 1e-9,
        "batch_size": batch_size,
        "train_size": len(train_set),
        "val_size": len(val_set),
    }

    out_dir = serialize(operator=operator, out_dir=out_dir)
    checkpoint_path = out_dir.joinpath("checkpoint.json")
    with open(checkpoint_path, "w") as file_handle:
        json.dump(checkpoint, file_handle)

    return out_dir
