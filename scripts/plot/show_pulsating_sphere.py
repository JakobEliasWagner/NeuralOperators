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
    Slider,
)
from bokeh.palettes import (
    RdBu,
)
from bokeh.plotting import (
    figure,
)
from bokeh.plotting.contour import (
    contour_data,
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
        x, y = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
        z = np.sin(x) + np.cos(2 * y)

        self.fig = figure(
            height=800,
            width=1400,
            title="Operator Transmission loss",
            tools="crosshair,pan,reset,save,wheel_zoom",
            x_range=[0, 1],
            y_range=[0, 1],
        )

        self.contour_plt = self.fig.contour(x, y, z, levels=np.linspace(-2, 2, 128), fill_color=RdBu)

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
        min_y1_imag = float(torch.min(self.dataset.u[:, :, 1]))
        max_y1_imag = float(torch.max(self.dataset.u[:, :, 1]))
        delta = max_y1_imag - min_y1_imag
        self.y1_imag = Slider(
            title="IM(Y1)",
            value=min_y1_imag + delta / 2,
            start=min_y1_imag,
            end=max_y1_imag,
            step=delta / widget_steps,
        )
        # Y2
        min_y2_real = float(torch.min(self.dataset.u[:, :, 2]))
        max_y2_real = float(torch.max(self.dataset.u[:, :, 2]))
        delta = max_y2_real - min_y2_real
        self.y2_real = Slider(
            title="RE(Y2)",
            value=min_y2_real + delta / 2,
            start=min_y2_real,
            end=max_y2_real,
            step=delta / widget_steps,
        )
        min_y2_imag = float(torch.min(self.dataset.u[:, :, 3]))
        max_y2_imag = float(torch.max(self.dataset.u[:, :, 3]))
        delta = max_y2_imag - min_y2_imag
        self.y2_imag = Slider(
            title="IM(Y2)",
            value=min_y2_imag + delta / 2,
            start=min_y2_imag,
            end=max_y2_imag,
            step=delta / widget_steps,
        )
        min_frequency = float(torch.min(self.dataset.u[:, :, 4]))
        max_frequency = float(torch.max(self.dataset.u[:, :, 4]))
        delta = max_frequency - min_frequency
        self.frequency = Slider(
            title="Frequency",
            value=min_frequency + delta / 2,
            start=min_frequency,
            end=max_frequency,
            step=delta / widget_steps,
        )

        self.resolution = Slider(
            title="Resolution",
            value=50,
            start=10,
            end=200,
            step=1,
        )

        # set the correct callback
        for w in [self.y1_real, self.y1_imag, self.y2_real, self.y2_imag, self.frequency, self.resolution]:
            w.on_change("value", self.update_data)

        # layout
        self.inputs = column(self.y1_real, self.y1_imag, self.y2_real, self.y2_imag, self.frequency, self.resolution)

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
        f = float(self.frequency.value)

        u_plot = torch.tensor([y1_real, y1_imag, y2_real, y2_imag, f]).reshape(1, 1, -1)
        x = u = self.dataset.transform["u"](u_plot)

        res = int(self.resolution.value)
        x_plot, y_plot = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res))
        z_plot = torch.zeros(res**2).reshape(1, -1, 1)
        y = torch.cat([x_plot.flatten().reshape(1, -1, 1), y_plot.flatten().reshape(1, -1, 1), z_plot], dim=2)
        y_trf = self.dataset.transform["y"](y)

        v = self.model(x, u, y_trf)
        v_plot = self.dataset.transform["v"].undo(v)
        v_plot = v_plot.reshape(1, -1, 2)
        v_plot = v_plot[:, :, 0]
        v_plot = v_plot.reshape(x_plot.shape)
        v_abs_max = torch.max(torch.abs(v_plot)).item()

        x_data = x_plot.detach().numpy()
        y_data = y_plot.detach().numpy()
        v_data = v_plot.detach().numpy()

        data = contour_data(
            x=x_data, y=y_data, z=v_data, levels=np.linspace(-v_abs_max, v_abs_max, 128), want_line=False
        )
        self.contour_plt.set_data(data)


operator_path = pathlib.Path.cwd().joinpath("finished_pi", "DeepDotOperator_2024_04_09_21_52_17_narrow")
dataset_path = pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow")

app = OperatorApp(data_path=dataset_path, model_path=operator_path)
