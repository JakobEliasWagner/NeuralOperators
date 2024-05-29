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

metric_candidates = [
    ("Data", pathlib.Path.cwd().joinpath("finished_pi", "DeepDotOperator-narrow-data", "metrics.csv")),
    ("PI", pathlib.Path.cwd().joinpath("finished_pi", "DeepDotOperator-narrow-pi-good", "metrics.csv")),
]

metrics = {}
for mc_name, mc in metric_candidates:
    metrics[mc_name] = pd.read_csv(mc)

fig, ax = plt.subplots()
colors = ["black", "blue", "red"]
for i, (name, metric) in enumerate(metrics.items()):
    train_loss_df = metric[metric["key"] == "Train DATA loss"].copy()
    train_loss_df["mov_avg"] = train_loss_df["value"].rolling(window=50).mean()
    sns.lineplot(
        train_loss_df, x="step", y="mov_avg", ax=ax, linestyle="--", color=colors[i], label=f"Training Set ({name})"
    )

    val_loss_df = metric[metric["key"] == "Val DATA loss"].copy()
    val_loss_df["mov_avg"] = val_loss_df["value"].rolling(window=50).mean()
    sns.lineplot(
        val_loss_df, x="step", y="mov_avg", ax=ax, linestyle="-", color=colors[i], label=f"Validation Set ({name})"
    )

ax.set_ylabel("Mean Squared Error")
ax.set_xlabel("Epoch")
ax.set_yscale("log")
fig.tight_layout()
plt.show()

# plt.savefig(metric_path.parent.joinpath("loss.pdf"))
