import pathlib

import numpy as np
import torch
from bokeh.io import (
    curdoc,
)
from bokeh.layouts import (
    column,
    row,
)
from bokeh.models import (
    ColumnDataSource,
    Slider,
)
from bokeh.plotting import (
    figure,
)
from loguru import (
    logger,
)

from nos.data import (
    PulsatingSphere,
)
from nos.operators import (
    deserialize,
)


class OperatorApp:
    def __init__(self, data_path: pathlib.Path, model_path: pathlib.Path):
        logger.info("Start initializing the application.")
        # data
        self.dataset = PulsatingSphere(data_path)
        self.model = deserialize(model_path)
        self.model.eval()

        # bokeh
        self.source = ColumnDataSource(
            data=dict(x=np.random.rand(50, 50), y=np.random.rand(50, 50), z=np.random.rand(50, 50))
        )

        self.fig = figure(
            height=800,
            width=1400,
            title="Operator Transmission loss",
            tools="crosshair,pan,reset,save,wheel_zoom",
            x_range=[0, 1],
            y_range=[0, 1],
        )

        self.tl_prediction = self.fig.contour(
            "x", "y", "z", source=self.source, levels=np.linspace(-1, 1, 31), line_color="black"
        )

        # widgets
        widget_steps = 50

        # Y1
        min_y1_real = float(torch.min(self.dataset.u[:, :, 0]))
        max_y1_real = float(torch.max(self.dataset.u[:, :, 0]))
        delta = max_y1_real - min_y1_real
        self.y1_real = Slider(
            title="RE(Y1)",
            value=min_y1_real + delta / 2,
            start=min_y1_real,
            end=max_y1_real,
            step=delta / widget_steps,
        )
        min_y1_imag = float(torch.min(self.dataset.u[:, :, 0]))
        max_y1_imag = float(torch.max(self.dataset.u[:, :, 0]))
        delta = max_y1_imag - min_y1_imag
        self.y1_imag = Slider(
            title="IM(Y1)",
            value=min_y1_imag + delta / 2,
            start=min_y1_imag,
            end=max_y1_imag,
            step=delta / widget_steps,
        )
        # Y2
        min_y2_real = float(torch.min(self.dataset.u[:, :, 0]))
        max_y2_real = float(torch.max(self.dataset.u[:, :, 0]))
        delta = max_y2_real - min_y2_real
        self.y2_real = Slider(
            title="RE(Y2)",
            value=min_y2_real + delta / 2,
            start=min_y2_real,
            end=max_y2_real,
            step=delta / widget_steps,
        )
        min_y2_imag = float(torch.min(self.dataset.u[:, :, 0]))
        max_y2_imag = float(torch.max(self.dataset.u[:, :, 0]))
        delta = max_y2_imag - min_y2_imag
        self.y2_imag = Slider(
            title="IM(Y2)",
            value=min_y2_imag + delta / 2,
            start=min_y2_imag,
            end=max_y2_imag,
            step=delta / widget_steps,
        )

        # set the correct callback
        for w in [self.y1_real, self.y1_imag, self.y2_real, self.y2_imag]:
            w.on_change("value", self.update_data)

        # layout
        self.inputs = column(self.y1_real, self.y1_imag, self.y2_real, self.y2_imag)

        logger.info("Finished initializing the application.")
        self.run()

    def run(self):
        logger.info("Start running the application.")
        curdoc().add_root(row(self.inputs, self.fig, width=1600))
        curdoc().title = "Operator Transmission Loss"

    def update_data(self, attr, old, new):
        # Get the current slider values
        y1_real = float(self.y1_real.value)
        y1_imag = float(self.y1_imag.value)
        y2_real = float(self.y2_real.value)
        y2_imag = float(self.y2_imag.value)
        f = 500.0

        u_plot = torch.tensor([y1_real, y1_imag, y2_real, y2_imag, f]).reshape(1, 1, -1)
        x = u = self.dataset.transform["u"](u_plot)

        res = 50
        x_plot = torch.linspace(0, 1, res)
        y_plot = torch.linspace(0, 1, res)
        z_plot = torch.zeros(res**2).reshape(1, -1, 1)
        xx, yy = torch.meshgrid([x_plot, y_plot])
        xx, yy = xx.reshape(1, -1, 1), yy.reshape(1, -1, 1)
        y = torch.cat([xx, yy, z_plot], dim=2)
        y_trf = self.dataset.transform["y"](y)

        v = self.model(x, u, y_trf)
        v_plot = self.dataset.transform["v"].undo(v)

        self.source.data = dict(
            x=xx.squeeze().detach().numpy(),
            y=yy.squeeze().detach().numpy(),
            z=v_plot.reshape(yy.shape).squeeze().detach().numpy(),
        )


operator_path = pathlib.Path.cwd().joinpath("out_models", "DeepDotOperator_2024_04_08_13_49_02")
dataset_path = pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_500")

app = OperatorApp(data_path=dataset_path, model_path=operator_path)
