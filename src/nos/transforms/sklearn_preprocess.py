import sklearn.preprocessing as skp
import torch
from continuity.transforms import (
    Transform,
)


class SKLearnPreprocess(Transform):
    def __init__(self, transform_name: str = "RobustScaler", src: torch.tensor = None, **kwargs):
        super().__init__()

        self.n_dim = src.size(-1)
        self.module = getattr(skp, transform_name)(**kwargs)
        self.module = self.module.fit(src.view(-1, self.n_dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.module.transform(tensor.view(-1, self.n_dim))
        out = torch.tensor(out)
        return out.reshape(tensor.shape).to(torch.get_default_dtype())

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.module.inverse_transform(tensor.view(-1, self.n_dim))
        out = torch.tensor(out)
        return out.reshape(tensor.shape).to(torch.get_default_dtype())
