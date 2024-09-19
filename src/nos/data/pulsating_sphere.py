import json  # noqa: D100
import pathlib

import adios2 as ad
import torch
from continuiti.data import OperatorDataset


class ConstBoundaryDataset(OperatorDataset):
    """Pulsating sphere dataset with constant valued boundaries."""

    def __init__(
        self,
        dataset_path: pathlib.Path,
        observations: int = -1,
    ) -> None:
        """Initialize.

        Args:
            dataset_path (pathlib.Path): Path to the csv file.
            observations (int, optional): Number of observations. Defaults to -1 (all).

        """
        tensor_path = dataset_path.joinpath("pressure_0.bp")
        with ad.FileReader(str(tensor_path)) as reader:
            geom = torch.from_numpy(reader.read("geometry"))[:, :-1]

            pr = []
            pi = []
            for step in range(reader.num_steps()):
                pr.append(reader.read("p_real", step_selection=[step, 1]))
                pi.append(reader.read("p_imag", step_selection=[step, 1]))
            pressure_real = torch.stack([torch.from_numpy(p) for p in pr])
            pressure_imag = torch.stack([torch.from_numpy(p) for p in pi])
            pressure = torch.stack([pressure_real.squeeze(), pressure_imag.squeeze()])
            pressure = pressure.to(torch.get_default_dtype())

        n_observations = pressure.size(1)

        json_path = dataset_path.joinpath("properties.json")
        with json_path.open("r") as file:
            properties = json.load(file)
        top_samples = torch.tensor(properties["top_samples"])
        right_samples = torch.tensor(properties["right_samples"])
        frequency_samples = torch.tensor(properties["frequency_samples"]).unsqueeze(1)

        x = torch.cat([top_samples, right_samples, frequency_samples], dim=1).unsqueeze(-1)
        x_min, _ = torch.min(x, dim=0, keepdim=True)
        x_max, _ = torch.max(x, dim=0, keepdim=True)

        u = x

        y = geom.transpose(0, 1).unsqueeze(0).expand(n_observations, -1, -1).to(torch.get_default_dtype())
        v = pressure.transpose(0, 1)

        perm = torch.randperm(n_observations)
        idx = perm[:observations]

        super().__init__(x[idx], u[idx], y[idx], v[idx])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item at idx from dataset."""
        x_min = torch.tensor([[0.0], [-1.0], [0.0], [-1.0], [400.0]])
        x_scale = torch.tensor([[1.0], [1.0], [1.0], [1.0], [100.0]])
        x = ((self.x[idx] - x_min) / x_scale) * 2.0 - 1.0
        v_max, _ = torch.max(torch.abs(self.v[idx]), dim=1, keepdim=True)

        return x, x, self.y[idx] * 2.0 - 1.0, self.v[idx] / v_max


class InverseConstBoundaryDataset(OperatorDataset):
    """Inverse pulsating sphere dataset with constant boundary conditions."""

    def __init__(
        self,
        dataset_path: pathlib.Path,
        observations: int = -1,
        points: int = 10,
        sensors: int = -1,
        sensor_idx: torch.Tensor | None = None,
    ) -> None:
        """Initialize.

        Args:
            dataset_path (pathlib.Path): path to the dataset bp.
            observations (int, optional): Number of observations in the dataset. Defaults to -1 (all).
            points (int, optional): Number of points on which the boundary condition is sampled. Defaults to 10.
            sensors (int, optional): Number of sensors during training. Defaults to -1 (all).
            sensor_idx (torch.Tensor | None, optional): Indices of the sensors used during training. Defaults to None.

        """
        tensor_path = dataset_path.joinpath("pressure_0.bp")
        with ad.FileReader(str(tensor_path)) as reader:
            geom = torch.from_numpy(reader.read("geometry"))[:, :-1]

            pr = []
            pi = []
            for step in range(reader.num_steps()):
                pr.append(reader.read("p_real", step_selection=[step, 1]))
                pi.append(reader.read("p_imag", step_selection=[step, 1]))
            pressure_real = torch.stack([torch.from_numpy(p) for p in pr])
            pressure_imag = torch.stack([torch.from_numpy(p) for p in pi])
            pressure = torch.stack([pressure_real.squeeze(), pressure_imag.squeeze()])
            pressure = pressure.to(torch.get_default_dtype())

        n_observations = pressure.size(1)

        json_path = dataset_path.joinpath("properties.json")
        with json_path.open("r") as file:
            properties = json.load(file)
        top_samples = torch.tensor(properties["top_samples"])
        right_samples = torch.tensor(properties["right_samples"])
        frequency_samples = torch.tensor(properties["frequency_samples"]).unsqueeze(1)

        x = geom.transpose(0, 1).unsqueeze(0).expand(n_observations, -1, -1).to(torch.get_default_dtype())
        u = pressure.transpose(0, 1)

        if sensors > 0 or sensor_idx is not None:
            if sensor_idx is not None:
                self.sensor_idx = sensor_idx
            else:
                self.sensor_idx = torch.randperm(x.size(-1))
                self.sensor_idx = self.sensor_idx[:sensors]
            x = x[:, :, self.sensor_idx]
            u = u[:, :, self.sensor_idx]
        else:
            self.sensor_idx = torch.tensor([])

        y = torch.linspace(0, 1, points).reshape(1, 1, -1)
        y = y.expand(n_observations, -1, -1)

        v = torch.cat([top_samples, right_samples, frequency_samples], dim=1).unsqueeze(-1)
        self.v_min = torch.tensor([[0.0], [-1.0], [0.0], [-1.0], [400.0]])
        self.v_scale = torch.tensor([[1.0], [1.0], [1.0], [1.0], [100.0]])
        v = v.expand(-1, -1, y.size(-1))

        perm = torch.randperm(n_observations)
        idx = perm[:observations]

        super().__init__(x[idx], u[idx], y[idx], v[idx])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item at idx from dataset."""
        x = self.x[idx] * 2.0 - 1.0
        u_max, _ = torch.max(torch.abs(self.u[idx]), dim=1, keepdim=True)
        u = self.u[idx] / u_max

        y = self.y[idx] * 2.0 - 1.0
        v = ((self.v[idx] - self.v_min) / self.v_scale) * 2.0 - 1.0

        return x, u, y, v
