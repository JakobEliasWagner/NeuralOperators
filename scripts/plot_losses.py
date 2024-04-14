import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Helvetica",
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    }
)

metric_path = pathlib.Path.cwd().joinpath("finished_pi", "DeepDotOperator-narrow-pi", "metrics.csv")
metrics = pd.read_csv(metric_path)

fig, ax = plt.subplots()

train_loss_df = metrics[metrics["key"] == "Train DATA loss"].copy()
train_loss_df["mov_avg"] = train_loss_df["value"].rolling(window=50).mean()
sns.lineplot(train_loss_df, x="step", y="mov_avg", ax=ax, linestyle="--", color="black", label="Training Set")

val_loss_df = metrics[metrics["key"] == "Val DATA loss"].copy()
val_loss_df["mov_avg"] = val_loss_df["value"].rolling(window=50).mean()
sns.lineplot(val_loss_df, x="step", y="mov_avg", ax=ax, linestyle="-", color="black", label="Validation Set")

ax.set_ylabel("Mean Squared Error")
ax.set_xlabel("Epoch")
ax.set_yscale("log")
fig.tight_layout()
plt.savefig(metric_path.parent.joinpath("loss.pdf"))
