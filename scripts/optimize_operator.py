import pathlib

import optuna
import torch
from torch.utils.data import DataLoader, random_split

from continuity.operators import DeepONet
from nos.data import TLDatasetCompact
from nos.trainer.average_metric import AverageMetric

DATA_DIR = pathlib.Path.cwd().joinpath("data", "transmission_loss_const_gap")
TRAIN_PATH = DATA_DIR.joinpath("dset.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score(train_loss: AverageMetric, val_loss: AverageMetric) -> float:
    alpha = 1.0
    beta = 1.0
    return alpha * val_loss.avg + beta * ((val_loss.avg - train_loss.avg) / val_loss.avg) ** 2


def objective(trial: optuna.trial):
    # datasets
    full_set = TLDatasetCompact(TRAIN_PATH)
    train_set, val_set = random_split(full_set, [0.9, 0.1])
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # operator
    operator_config = {
        "branch_width": 128,
        "branch_depth": trial.suggest_int("branch_depth", 4, 16),
        "trunk_width": 128,
        "trunk_depth": trial.suggest_int("trunk_depth", 4, 16),
        "basis_functions": trial.suggest_int("basis_functions", 4, 16),
    }
    operator = DeepONet(full_set.shapes, **operator_config)
    operator.to(DEVICE)

    # optimizer
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3)

    # criterion
    criterion = torch.nn.MSELoss()

    # training loop
    current_score = torch.inf
    epochs = trial.suggest_int("Epochs", 10, 100)
    for epoch in range(epochs):
        avg_train_loss = AverageMetric("Eval-loss", ":6.3f")
        avg_val_loss = AverageMetric("Eval-loss", ":6.3f")
        # train
        operator.train()
        for x, u, y, v in train_loader:
            x, u, y, v = x.to(DEVICE), u.to(DEVICE), y.to(DEVICE), v.to(DEVICE)
            output = operator(x, u, y)
            loss = criterion(output, v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_train_loss.update(loss.item(), train_loader.batch_size)
        # eval
        operator.eval()
        for x, u, y, v in val_loader:
            x, u, y, v = x.to(DEVICE), u.to(DEVICE), y.to(DEVICE), v.to(DEVICE)
            output = operator(x, u, y)
            loss = criterion(output, v)
            avg_val_loss.update(loss.item(), val_loader.batch_size)

        current_score = score(avg_train_loss, avg_val_loss)

        trial.report(current_score, step=epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return current_score


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner(), storage="sqlite:///study.db"
    )
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
