import json  # noqa: D100
import pathlib
import shutil
import time

import mlflow
import pandas as pd
import torch.optim.lr_scheduler as sched
import torch.utils.data
from continuiti.data import OperatorDataset
from continuiti.operators import Operator
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from nos.trainers.util import save_checkpoint
from nos.utils import UniqueId


class Trainer:
    """Simple trainer implementation."""

    def __init__(
        self,
        operator: Operator,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: sched.LRScheduler | None = None,
        max_epochs: int = 1000,
        batch_size: int = 16,
        max_n_logs: int = 200,
        out_dir: pathlib.Path | None = None,
    ) -> None:
        """Initialize.

        Args:
            operator (Operator): Operator that should be trained.
            criterion (torch.nn.Module): Criterion to train.
            optimizer (torch.optim.Optimizer): Optimizer to minimize criterion.
            lr_scheduler (sched.LRScheduler | None, optional): Scheduler to govern lr during training.I
                Defaults to ConstantLR.
            max_epochs (int, optional): Maximum epochs to train. Defaults to 1000.
            batch_size (int, optional): Batch size. Defaults to 16.
            max_n_logs (int, optional): Maxiumum amount of saved models. Defaults to 200.
            out_dir (pathlib.Path | None, optional): Directory to save checkpoints. Defaults to None.

        """
        self.operator = operator
        self.criterion = criterion
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler: sched.LRScheduler
        if lr_scheduler is None:
            self.lr_scheduler = sched.ConstantLR(self.optimizer, factor=1.0)
        else:
            self.lr_scheduler = lr_scheduler

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.test_val_split = 0.9

        # logging and model serialization
        self.out_dir: pathlib.Path
        if out_dir is None:
            uid = UniqueId()
            self.out_dir = pathlib.Path.cwd().joinpath("run", str(uid))
        else:
            self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        log_epochs = torch.round(torch.linspace(0, max_epochs, max_n_logs)).tolist()
        self.log_epochs = [int(epoch) for epoch in log_epochs]

    def __call__(self, data_set: OperatorDataset, run_name: str | None = None) -> Operator:
        """Train operator."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_set.u = data_set.u.to(device)
        data_set.y = data_set.y.to(device)
        data_set.x = data_set.x.to(device)
        data_set.v = data_set.v.to(device)

        for trf in data_set.transform:
            data_set.transform[trf] = data_set.transform[trf].to(device)

        # data
        train_set, val_set = random_split(data_set, [self.test_val_split, 1 - self.test_val_split])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)

        training_config = {
            "val_indices": val_set.indices,
            "val_size": len(val_set),
            "train_indices": train_set.indices,
            "train_size": len(train_set),
        }
        with self.out_dir.joinpath("training_config.json").open("w") as file_handle:
            json.dump(training_config, file_handle)

        # setup training
        self.operator.to(device)
        self.criterion.to(device)

        best_val_loss = float("inf")
        val_losses = []
        train_losses = []
        lrs = []
        times = []

        pbar = tqdm(range(self.max_epochs))
        train_loss = torch.inf
        val_loss = torch.inf

        start = time.time()

        with mlflow.start_run():
            if run_name is not None:
                mlflow.set_tag("mlflow.runName", run_name)
            for epoch in pbar:
                pbar.set_description(
                    f"Train Loss: {train_loss: .6f},\t"
                    f"Val Loss: {val_loss: .6f},\t"
                    f"Lr: {self.optimizer.param_groups[0]['lr']}",
                )
                train_loss = self.train(train_loader, self.operator, device)
                val_loss = self.eval(val_loader, self.operator, device)
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
                if val_loss < best_val_loss:
                    best_dir = self.out_dir.joinpath("best")
                    if best_dir.is_dir():
                        shutil.rmtree(best_dir)
                    best_dir.mkdir(exist_ok=True, parents=True)

                    save_checkpoint(
                        self.operator,
                        val_loss,
                        train_loss,
                        epoch,
                        start,
                        self.batch_size,
                        train_set,
                        val_set,
                        best_dir,
                    )
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
            },
        )
        training_curves.to_csv(self.out_dir.joinpath("training.csv"))
        return self.operator

    def train(self, loader: DataLoader, model: Operator, device: torch.device) -> float:
        """Train operator. Returns mean loss."""
        # switch to train mode
        model.train()
        losses = []
        for x, u, y, v in loader:
            xd, ud, yd, vd = x.to(device), u.to(device), y.to(device), v.to(device)

            # compute output
            output = model(xd, ud, yd)
            loss = self.criterion(output, vd)

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update metrics
            losses.append(loss.item())

        return torch.mean(torch.tensor(losses)).item()

    def eval(self, loader: DataLoader, model: Operator, device: torch.device) -> float:
        """Evaluate operator. Returns mean loss."""
        # switch to train mode
        model.eval()

        losses = []
        for x, u, y, v in loader:
            xd, ud, yd, vd = x.to(device), u.to(device), y.to(device), v.to(device)

            # compute output
            output = model(xd, ud, yd)
            loss = self.criterion(output, vd)

            # update metrics
            losses.append(loss.item())

        return torch.mean(torch.tensor(losses)).item()
