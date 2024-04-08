import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import (
    tabulate,
)

from nos.data import (
    TLDatasetCompact,
    TLDatasetCompactWave,
)
from nos.operators import (
    DeepDotOperator,
    DeepNeuralOperator,
    DeepONet,
    FourierNeuralOperator,
)


def transform_relevant_parameters(archs: list) -> list:
    out = []
    for arch_size, arch in archs:
        arch_name = arch.__class__.__name__
        n_param = sum(p.numel() for p in arch.parameters() if p.requires_grad)
        n_param_e = f"{np.log10(n_param):.2f}"
        out.append((arch_name, arch_size, n_param, n_param_e))
    return out


archs = []

dataset = TLDatasetCompact(pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth"))
fno_dataset = TLDatasetCompactWave(pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth"))

# deep o net
archs.append(
    (
        "big",
        DeepONet(
            dataset.shapes,
            branch_width=64,
            branch_depth=96,
            trunk_depth=64,
            trunk_width=96,
            dropout_p=0.2,
            stride=8,
            basis_functions=48,
        ),
    )
)
archs.append(
    (
        "medium",
        DeepONet(
            dataset.shapes,
            branch_width=42,
            branch_depth=24,
            trunk_depth=28,
            trunk_width=44,
            dropout_p=0.2,
            stride=4,
            basis_functions=48,
        ),
    )
)
archs.append(
    (
        "small",
        DeepONet(
            dataset.shapes,
            branch_width=32,
            branch_depth=4,
            trunk_depth=5,
            trunk_width=32,
            dropout_p=0.2,
            stride=1,
            basis_functions=4,
        ),
    )
)

# deep neural operator
archs.append(
    (
        "big",
        DeepNeuralOperator(
            dataset.shapes,
            depth=64,
            width=125,
            stride=8,
            dropout_p=0.2,
        ),
    )
)
archs.append(
    (
        "medium",
        DeepNeuralOperator(
            dataset.shapes,
            depth=32,
            width=56,
            stride=4,
            dropout_p=0.2,
        ),
    )
)
archs.append(
    (
        "small",
        DeepNeuralOperator(
            dataset.shapes,
            depth=8,
            width=36,
            stride=2,
            dropout_p=0.2,
        ),
    )
)

# deep dot operator
archs.append(
    (
        "big",
        DeepDotOperator(
            dataset.shapes,
            branch_width=64,
            branch_depth=48,
            trunk_depth=48,
            trunk_width=64,
            dot_depth=48,
            dot_width=112,
            stride=8,
        ),
    )
)
archs.append(
    (
        "medium",
        DeepDotOperator(
            dataset.shapes,
            branch_width=32,
            branch_depth=16,
            trunk_depth=16,
            trunk_width=32,
            dot_depth=32,
            dot_width=46,
            stride=4,
            dropout_p=0.2,
        ),
    )
)
archs.append(
    (
        "small",
        DeepDotOperator(
            dataset.shapes,
            branch_width=28,
            branch_depth=4,
            trunk_depth=4,
            trunk_width=32,
            dot_depth=4,
            dot_width=32,
            stride=2,
            dropout_p=0.2,
        ),
    )
)

"""# transformer operator
archs.append(("big", TransformerOperator(
    dataset.shapes,
    hidden_dim=96,
    encoding_depth=8,
    n_heads=16,
    dropout_p=.5,
    feed_forward_depth=8,
    function_encoder_depth=4,
    function_encoder_layer_depth=8
)))
archs.append(("medium", TransformerOperator(
    dataset.shapes,
    hidden_dim=64,
    encoding_depth=4,
    n_heads=4,
    dropout_p=.5,
    feed_forward_depth=8,
    function_encoder_depth=2,
    function_encoder_layer_depth=8
)))
archs.append(("small", TransformerOperator(
    dataset.shapes,
    hidden_dim=20,
    encoding_depth=2,
    n_heads=2,
    dropout_p=.2,
    feed_forward_depth=6,
    function_encoder_depth=1,
    function_encoder_layer_depth=2
)))"""

# fourier neural operator
archs.append(("big", FourierNeuralOperator(fno_dataset.shapes, width=22, depth=16)))
archs.append(("medium", FourierNeuralOperator(fno_dataset.shapes, width=10, depth=8)))
archs.append(("small", FourierNeuralOperator(fno_dataset.shapes, width=4, depth=5)))

## -------------------------------------------------- ##
col_names = ["Arch", "Size", "#param", "param order"]
archs = transform_relevant_parameters(archs)
print(tabulate(archs, headers=col_names))

df = pd.DataFrame(archs, columns=col_names)
sns.set_style("whitegrid")
fig, ax = plt.subplots()
sns.scatterplot(df, x="Arch", y="#param", hue="Size", style="Arch", ax=ax, legend=False)
ax.set_yscale("log")
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticks([1e4, 1e5, 1e6])
fig.tight_layout()
plt.show()
