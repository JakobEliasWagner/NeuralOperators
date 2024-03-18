from collections import (
    defaultdict,
)
from typing import (
    List,
)

import pandas as pd
import torch
import torch.nn as nn
from continuity.data import (
    OperatorDataset,
)
from torch.utils.data import (
    DataLoader,
)

from nos.operators import (
    NeuralOperator,
)


def eval_operator(operator: NeuralOperator, dataset: OperatorDataset, losses: List[nn.Module]) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    cs = {c.__class__.__name__: c for c in losses}
    ls = defaultdict(list)

    operator.eval()
    with torch.no_grad():
        for x, u, y, v in loader:
            out = operator(x, u, y)

            for name, crit in cs.items():
                ls[name].append(crit(out, v).item())

    values = {
        "x": dataset.x.squeeze().tolist(),
        "u": dataset.u.squeeze().tolist(),
        "y": dataset.y.squeeze().tolist(),
        "v": dataset.v.squeeze().tolist(),
    }
    values.update(ls)

    return pd.DataFrame(values)
