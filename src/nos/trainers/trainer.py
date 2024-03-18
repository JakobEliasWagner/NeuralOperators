import json
import pathlib
import time

import mlflow
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
    serialize,
)

from .average_metric import (
    AverageMetric,
)


class Trainer:
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

        # data
        train_set, val_set = random_split(data_set, [self.test_val_split, 1 - self.test_val_split])
        train_loader = DataLoader(train_set, batch_size=batch_size)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # scheduler
        if scheduler is None:
            scheduler = sched.ConstantLR(self.optimizer, factor=1.0)

        # setup training
        operator.to(device)

        logger.info(f"Starting training for {max_epochs} epochs on {device}.")
        with mlflow.start_run():
            best_val_loss = float("inf")
            last_best_update = 0
            val_losses = []
            train_losses = []
            lrs = []

            pbar = tqdm(range(max_epochs))
            train_loss = torch.inf
            val_loss = torch.inf

            start = time.time_ns()
            for epoch in pbar:
                pbar.set_description(
                    f"Train Loss: {train_loss: .6f},\t Val Loss: {val_loss: .6f}, Lr: {self.optimizer.param_groups[0]['lr']}"
                )
                train_loss = self.train(train_loader, operator, epoch, device)
                val_loss = self.eval(val_loader, operator, epoch, device)
                scheduler.step()
                self.log("lr", self.optimizer.param_groups[0]["lr"], epoch)
                lrs.append(self.optimizer.param_groups[0]["lr"])
                # update model parameters

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # save best model
                if val_loss < best_val_loss and epoch - last_best_update >= max_epochs // self.max_n_saved_models:
                    self._save_checkpoint(
                        operator, val_loss, train_loss, epoch, start, batch_size, train_set, val_set, out_dir
                    )
                    last_best_update = epoch
                    best_val_loss = val_loss

        logger.info("Training finished.")
        final_out_dir = self._save_checkpoint(
            operator, val_loss, train_loss, epoch, start, batch_size, train_set, val_set, out_dir
        )
        logger.info(f"Saved final model to {final_out_dir}.")

        training_curves = pd.DataFrame(
            {
                "Epochs": torch.arange(0, max_epochs).tolist(),
                "Val_loss": val_losses,
                "Train_loss": train_losses,
                "Lr": lrs,
            }
        )
        training_curves.to_csv(out_dir.joinpath("training.csv"))

        logger.info("Saved training curves to file.")
        return operator

    def _save_checkpoint(
        self, operator, val_loss, train_loss, epoch, start, batch_size, train_set, val_set, out_dir
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

    def train(self, loader, model, epoch, device):
        batch_time = AverageMetric("Train-time", ":6.3f")
        data_time = AverageMetric("Train-data-load", ":6.3f")
        data_transfer = AverageMetric("Train-data-transfer", "6.3f")
        avg_loss = AverageMetric("Train-loss", ":6.3f")

        # switch to train mode
        model.train()
        end = time.time()

        for x, u, y, v in loader:
            start = time.time()
            data_time.update(start - end)  # measure data loading time
            x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
            data_transfer.update(time.time() - start)

            # compute output
            output = model(x, u, y)
            loss = self.criterion(output, v)

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update metrics
            avg_loss.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        self.log(**batch_time.to_dict(), epoch=epoch)
        self.log(**data_time.to_dict(), epoch=epoch)
        self.log(**data_transfer.to_dict(), epoch=epoch)
        self.log(**avg_loss.to_dict(), epoch=epoch)
        return avg_loss.avg

    def eval(self, loader, model, epoch, device):
        batch_time = AverageMetric("Eval-time", ":6.3f")
        data_time = AverageMetric("Eval-data-load", ":6.3f")
        data_transfer = AverageMetric("Eval-data-transfer", "6.3f")
        avg_loss = AverageMetric("Eval-loss", ":6.3f")

        # switch to train mode
        model.eval()
        end = time.time()

        for x, u, y, v in loader:
            start = time.time()
            data_time.update(start - end)  # measure data loading time
            x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
            data_transfer.update(time.time() - start)

            # compute output
            output = model(x, u, y)
            loss = self.criterion(output, v)

            # update metrics
            avg_loss.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        self.log(**batch_time.to_dict(), epoch=epoch)
        self.log(**data_time.to_dict(), epoch=epoch)
        self.log(**data_transfer.to_dict(), epoch=epoch)
        self.log(**avg_loss.to_dict(), epoch=epoch)
        return avg_loss.avg

    def log(self, name: str, val: float, epoch: int):
        if epoch % self.log_interval == 0:
            mlflow.log_metric(key=name, value=val, step=epoch)
