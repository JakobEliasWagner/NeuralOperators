import torch
from continuity.data import (
    OperatorDataset,
)
from continuity.transforms import (
    Transform,
)


class SelfSupervisedDataset(OperatorDataset):
    """Self-supervised dataset."""

    def __init__(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        input_ratio: float = -1,
        n_input: int = 32,
        n_combinations: int = 1,
        x_transform: Transform = None,
        u_transform: Transform = None,
    ):
        self.input_ratio = input_ratio
        self.n_combinations = n_combinations

        if input_ratio > 0:
            n_output = u.size(1) - int(u.size(1) * input_ratio)  # ceil n_input
        else:
            n_output = u.size(1) - n_input
        # assemble
        xx = []
        uu = []
        yy = []
        vv = []

        for xi, ui in zip(x, u):
            for _ in range(n_combinations):
                perm = torch.randperm(ui.size(0))
                in_indices = perm[n_output:]
                out_indices = perm[:n_output]
                # sample according to random permutations
                xx.append(xi[in_indices])
                uu.append(ui[in_indices])
                yy.append(xi[out_indices])
                vv.append(ui[out_indices])

        # assemble dataset
        xx = torch.stack(xx)
        uu = torch.stack(uu)
        yy = torch.stack(yy)
        vv = torch.stack(vv)

        super().__init__(
            xx,
            uu,
            yy,
            vv,
            x_transform=x_transform,
            u_transform=u_transform,
            y_transform=x_transform,
            v_transform=u_transform,
        )
