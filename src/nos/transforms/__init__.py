"""Transformations are bijective projections for data."""

from nos.transforms.min_max_scale import MinMaxScale
from nos.transforms.quantile_scaler import QuantileScaler

__all__ = ["MinMaxScale", "QuantileScaler"]
