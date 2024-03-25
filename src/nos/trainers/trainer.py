import json
import pathlib
import time

import pandas as pd
import torch.optim.lr_scheduler as sched
import torch.utils.data
from continuity.data import (
    OperatorDataset,
)
from loguru import (
    logger,
)
from torch.utils.data import (
    DataLoader,
    random_split,
)
from tqdm import (
    tqdm,
)

from nos.operators import (
    NeuralOperator,
)

from .average_metric import (
    AverageMetric,
)
from .util import (
    save_checkpoint,
)


class FastTrainer:
    def __init__(
        self,
        criterion,
        optimizer,
        log_interval: int = 10,
        max_n_saved_models: int = 10,
    ):
        self.test_val_split = 0.9
        self.criterion = criterion
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.max_n_saved_models = max_n_saved_models

    def __call__(
        self,
        operator: NeuralOperator,
        data_set: OperatorDataset,
        max_epochs: int = 100,
        batch_size: int = 2**10,
        scheduler: sched.LRScheduler = None,
        out_dir: pathlib.Path = None,
    ) -> NeuralOperator:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if out_dir is None:
            out_dir = pathlib.Path.cwd().joinpath("out", "single_run")
            out_dir.mkdir(parents=True, exist_ok=True)

        # data
        train_set, val_set = random_split(data_set, [self.test_val_split, 1 - self.test_val_split])
        train_loader = DataLoader(train_set, batch_size=batch_size)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        training_config = {
            "val_indices": val_set.indices,
            "val_size": len(val_set),
            "train_indices": train_set.indices,
            "train_size": len(train_set),
        }
        with open(out_dir.joinpath("training_config.json"), "w") as file_handle:
            json.dump(training_config, file_handle)

        # scheduler
        if scheduler is None:
            scheduler = sched.ConstantLR(self.optimizer, factor=1.0)

        # setup training
        operator.to(device)

        logger.info(f"Starting training for {max_epochs} epochs on {device}.")

        best_val_loss = float("inf")
        last_best_update = 0
        val_losses = []
        train_losses = []
        lrs = []
        times = []

        pbar = tqdm(range(max_epochs))
        train_loss = torch.inf
        val_loss = torch.inf

        start = time.time()
        for epoch in pbar:
            pbar.set_description(
                f"Train Loss: {train_loss: .6f},\t Val Loss: {val_loss: .6f}, Lr: {self.optimizer.param_groups[0]['lr']}"
            )
            train_loss = self.train(train_loader, operator, epoch, device)
            val_loss = self.eval(val_loader, operator, epoch, device)
            scheduler.step()

            # update training parameters
            lrs.append(self.optimizer.param_groups[0]["lr"])
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            times.append(time.time() - start)

            # save best model
            if val_loss < best_val_loss and epoch - last_best_update >= max_epochs // self.max_n_saved_models:
                save_checkpoint(operator, val_loss, train_loss, epoch, start, batch_size, train_set, val_set, out_dir)
                last_best_update = epoch
                best_val_loss = val_loss

        logger.info("Training finished.")
        final_out_dir = save_checkpoint(
            operator, val_loss, train_loss, max_epochs, start, batch_size, train_set, val_set, out_dir
        )
        logger.info(f"Saved final model to {final_out_dir}.")

        training_curves = pd.DataFrame(
            {
                "Epochs": torch.arange(0, max_epochs).tolist(),
                "Val_loss": val_losses,
                "Train_loss": train_losses,
                "Lr": lrs,
                "time": times,
            }
        )
        training_curves.to_csv(out_dir.joinpath("training.csv"))
        logger.info("Saved training curves to file.")
        return operator

    def train(self, loader, model, epoch, device):
        avg_loss = AverageMetric("Train-loss", ":6.3f")

        # switch to train mode
        model.train()

        for x, u, y, v in loader:
            x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)

            # compute output
            output = model(x, u, y)
            loss = self.criterion(output, v)

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update metrics
            avg_loss.update(loss.item())

        return avg_loss.avg

    def eval(self, loader, model, epoch, device):
        avg_loss = AverageMetric("Eval-loss", ":6.3f")

        # switch to train mode
        model.eval()

        for x, u, y, v in loader:
            x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)

            # compute output
            output = model(x, u, y)
            loss = self.criterion(output, v)

            # update metrics
            avg_loss.update(loss.item())

        return avg_loss.avg
