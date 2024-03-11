import time

import mlflow
import torch.optim.lr_scheduler as sched
import torch.utils.data
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from continuity.data import OperatorDataset
from continuity.operators import Operator

from .average_metric import AverageMetric


class Trainer:
    def __init__(self, criterion, optimizer):
        self.test_val_split = 0.9
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = sched.CosineAnnealingWarmRestarts(optimizer, 100)

    def __call__(
        self, operator: Operator, data_set: OperatorDataset, max_epochs: int = 100, batch_size: int = 2**10
    ) -> Operator:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_set, val_set = random_split(data_set, [self.test_val_split, 1 - self.test_val_split])

        logger.info(f"Starting training for {max_epochs} epochs on {device}.")

        train_loader = DataLoader(train_set, batch_size=batch_size)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        operator.to(device)

        with mlflow.start_run():
            pbar = tqdm(range(max_epochs))
            train_loss = torch.inf
            val_loss = torch.inf
            for epoch in pbar:
                pbar.set_description(
                    f"Train Loss: {train_loss: .6f},\t Val Loss: {val_loss: .6f}, Lr: {self.optimizer.param_groups[0]['lr']}"
                )
                train_loss = self.train(train_loader, operator, epoch, device)
                val_loss = self.eval(val_loader, operator, epoch, device)
                self.scheduler.step(epoch)
                self.log("lr", self.optimizer.param_groups[0]["lr"], epoch)

        logger.info("Training finished.")

        return operator

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
            avg_loss.update(loss.item(), loader.batch_size)

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
            avg_loss.update(loss.item(), loader.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        self.log(**batch_time.to_dict(), epoch=epoch)
        self.log(**data_time.to_dict(), epoch=epoch)
        self.log(**data_transfer.to_dict(), epoch=epoch)
        self.log(**avg_loss.to_dict(), epoch=epoch)
        return avg_loss.avg

    def log(self, name: str, val: float, epoch: int):
        mlflow.log_metric(key=name, value=val, step=epoch)
