import time

import mlflow
import torch.utils.data
from torch.utils.data import DataLoader, random_split

from continuity.data import OperatorDataset
from continuity.operators import Operator

from .average_metric import AverageMetric


class Trainer:
    def __init__(self, criterion, optimizer, scheduler):
        self.test_val_split = 0.9
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, operator: Operator, data_set: OperatorDataset, max_epochs: int) -> Operator:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_set, val_set = random_split(data_set, [self.test_val_split, 1 - self.test_val_split])

        train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=4)

        operator.to(device)

        with mlflow.start_run():
            for epoch in range(max_epochs):
                self.train(train_loader, operator, epoch, device)
                self.eval(val_loader, operator, epoch, device)

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

    def log(self, name: str, val: float, epoch: int):
        mlflow.log_metric(key=name, value=val, step=epoch)
