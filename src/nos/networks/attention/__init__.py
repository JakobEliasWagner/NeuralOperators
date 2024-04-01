from torch.nn.functional import (
    scaled_dot_product_attention,
)

from .scaled_dot_product_attention import (
    nos_scaled_dot_product_attention,
)

__all__ = ["nos_scaled_dot_product_attention", "scaled_dot_product_attention"]
