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
from continuity.operators import (
    Operator,
)
from torch.utils.data import (
    DataLoader,
    random_split,
)
from tqdm import (
    tqdm,
)

from nos.utils import (
    UniqueId,
)

from .average_metric import (
    AverageMetric,
)
from .util import (
    save_checkpoint,
)


class Trainer:
    def __init__(
        self,
        operator: Operator,
        criterion,
        optimizer,
        lr_scheduler: sched.LRScheduler = None,
        max_epochs: int = 1000,
        batch_size: int = 16,
        max_n_logs: int = 200,
        max_n_saved_models: int = 10,
    ):
        self.operator = operator
        self.criterion = criterion
        self.optimizer = optimizer
        if lr_scheduler is None:
            self.lr_scheduler = sched.ConstantLR(self.optimizer, factor=1.0)
        else:
            self.lr_scheduler = lr_scheduler

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.test_val_split = 0.9

        # logging and model serialization
        uid = UniqueId()
        self.out_dir = pathlib.Path.cwd().joinpath("run", str(uid))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        log_epochs = torch.round(torch.linspace(0, max_epochs, max_n_logs))
        log_epochs = log_epochs.tolist()
        self.log_epochs = [int(epoch) for epoch in log_epochs]

        self.max_n_saved_models = max_n_saved_models

    def __call__(
        self,
        data_set: OperatorDataset,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # data
        train_set, val_set = random_split(data_set, [self.test_val_split, 1 - self.test_val_split])
        train_loader = DataLoader(train_set, batch_size=self.batch_size)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)

        training_config = {
            "val_indices": val_set.indices,
            "val_size": len(val_set),
            "train_indices": train_set.indices,
            "train_size": len(train_set),
        }
        with open(self.out_dir.joinpath("training_config.json"), "w") as file_handle:
            json.dump(training_config, file_handle)

        # setup training
        self.operator.to(device)

        best_val_loss = float("inf")
        last_best_update = 0
        val_losses = []
        train_losses = []
        lrs = []
        times = []

        pbar = tqdm(range(self.max_epochs))
        train_loss = torch.inf
        val_loss = torch.inf

        start = time.time()

        with mlflow.start_run():
            mlflow.pytorch.log_model(self.operator, "operator")
            for epoch in pbar:
                pbar.set_description(
                    f"Train Loss: {train_loss: .6f},\t Val Loss: {val_loss: .6f}, Lr: {self.optimizer.param_groups[0]['lr']}"
                )
                train_loss = self.train(train_loader, self.operator, epoch, device)
                val_loss = self.eval(val_loader, self.operator, epoch, device)
                self.lr_scheduler.step(epoch)

                # update training parameters
                lrs.append(self.optimizer.param_groups[0]["lr"])
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                times.append(time.time() - start)

                # log metrics
                if epoch in self.log_epochs:
                    mlflow.log_metric("Val loss", val_loss, step=epoch)
                    mlflow.log_metric("Train loss", train_loss, step=epoch)
                    mlflow.log_metric("LR", self.optimizer.param_groups[0]["lr"], step=epoch)

                # save best model
                if val_loss < best_val_loss and epoch - last_best_update >= self.max_epochs // self.max_n_saved_models:
                    save_checkpoint(
                        self.operator,
                        val_loss,
                        train_loss,
                        epoch,
                        start,
                        self.batch_size,
                        train_set,
                        val_set,
                        self.out_dir,
                    )
                    last_best_update = epoch
                    best_val_loss = val_loss

        save_checkpoint(
            self.operator,
            val_loss,
            train_loss,
            self.max_epochs,
            start,
            self.batch_size,
            train_set,
            val_set,
            self.out_dir,
        )

        training_curves = pd.DataFrame(
            {
                "Epochs": torch.arange(0, self.max_epochs).tolist(),
                "Val_loss": val_losses,
                "Train_loss": train_losses,
                "Lr": lrs,
                "time": times,
            }
        )
        training_curves.to_csv(self.out_dir.joinpath("training.csv"))

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
