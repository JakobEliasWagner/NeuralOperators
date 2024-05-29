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
    TLDatasetCompact,
)
from nos.operators import (
    deserialize,
)


class OperatorApp:
    def __init__(self, data_path: pathlib.Path, model_path: pathlib.Path):
        logger.info("Start initializing the application.")
        # data
        self.dataset = TLDatasetCompact(data_path)
        self.model = deserialize(model_path)
        self.model.eval()

        # bokeh
        self.source = ColumnDataSource(data=dict(x=np.linspace(-100, 10, 10), y=np.linspace(2000, 20000, 10)))
        self.closest_source = ColumnDataSource(data=dict(xs=np.random.rand(5, 10), ys=np.random.rand(5, 10)))

        self.fig = figure(
            height=800,
            width=1400,
            title="Operator Transmission loss",
            tools="crosshair,pan,reset,save,wheel_zoom",
            x_range=[-100, 10],
            y_range=[2000, 20000],
        )

        self.tl_prediction = self.fig.line(
            "x", "y", source=self.source, line_width=3, line_alpha=1, line_color="black"
        )
        self.tl_closest = self.fig.multi_line("xs", "ys", source=self.closest_source, line_width=15, line_alpha=0.05)

        # widgets
        widget_steps = 50

        # Radius
        min_radius = float(torch.min(self.dataset.u[:, :, 0])) * 1e3
        max_radius = float(torch.max(self.dataset.u[:, :, 0])) * 1e3
        delta = max_radius - min_radius
        self.radius = Slider(
            title="Radius", value=min_radius + delta / 2, start=min_radius, end=max_radius, step=delta / widget_steps
        )
        # Inner radius
        min_radius = float(torch.min(self.dataset.u[:, :, 1])) * 1e3
        max_radius = float(torch.max(self.dataset.u[:, :, 1])) * 1e3
        delta = max_radius - min_radius
        self.inner_radius = Slider(
            title="Inner Radius",
            value=min_radius + delta / 2,
            start=min_radius,
            end=max_radius,
            step=delta / widget_steps,
        )
        # Gap width
        min_gw = float(torch.min(self.dataset.u[:, :, 2])) * 1e3
        max_gw = float(torch.max(self.dataset.u[:, :, 2])) * 1e3
        delta = max_gw - min_gw
        self.gap_width = Slider(
            title="Gap Width", value=min_gw + delta / 2, start=min_gw, end=max_gw, step=delta / widget_steps
        )

        # set correct callback
        for w in [self.radius, self.inner_radius, self.gap_width]:
            w.on_change("value", self.update_data)

        # toggle if closes samples are plotted
        self.n_closest = Slider(title="N Closest from training", start=0, end=50, step=1, value=0)
        self.n_closest.on_change("value", self.update_data)

        # layout
        self.inputs = column(self.radius, self.inner_radius, self.gap_width, self.n_closest)

        self.update_data(None, None, None)
        logger.info("Finished initializing the application.")
        self.run()

    def run(self):
        logger.info("Start running the application.")
        curdoc().add_root(row(self.inputs, self.fig, width=1600))
        curdoc().title = "Operator Transmission Loss"

    def update_data(self, attr, old, new):
        # Get the current slider values
        r = self.radius.value
        ri = self.inner_radius.value
        gw = self.gap_width.value

        if ri >= r:
            self.inner_radius.value = 0.99 * self.radius.value
            ri = self.inner_radius.value

        if gw >= ri:
            self.gap_width.value = 0.99 * self.inner_radius.value
            gw = self.gap_width.value

        r *= 1e-3
        ri *= 1e-3
        gw *= 1e-3

        u_plot = torch.tensor([r, ri, gw]).reshape(1, 1, 3)
        x = u = self.dataset.transform["u"](u_plot)

        y_plot = torch.linspace(2000, 20000, 257)
        y = self.dataset.transform["y"](y_plot.reshape(1, -1, 1))
        v = self.model(x, u, y)
        v_plot = self.dataset.transform["v"].undo(v)

        self.source.data = dict(x=v_plot.squeeze().detach().numpy(), y=y_plot.squeeze().detach().numpy())

        n_closest = int(self.n_closest.value)
        if n_closest > 0:
            self.tl_closest.visible = True
            self.update_closest(r, ri, gw, n_closest)
        else:
            self.tl_closest.visible = False

    def update_closest(self, r, ri, gw, n):
        # find n closest
        ui = torch.tensor([r, ri, gw]).reshape(1, 1, 3)
        diff = (self.dataset.u - ui) ** 2
        diff = torch.sum(diff, dim=-1).squeeze()

        _, indices = torch.topk(input=-diff, k=n)

        vs = self.dataset.v[indices]
        ys = self.dataset.y[indices]
        ys = ys.squeeze().detach().tolist()
        vs = vs.squeeze().detach().tolist()

        self.closest_source.data = dict(xs=vs, ys=ys)


operator_path = pathlib.Path.cwd().joinpath(
    "finished_models",
    "medium",
    "deep_neural_operator",
    "DeepNeuralOperator_medium_0",
    "best_mean",
    "DeepNeuralOperator_2024_05_06_04_47_58",
)
dataset_path = pathlib.Path.cwd().joinpath("data", "train", "transmission_loss_smooth")

app = OperatorApp(data_path=dataset_path, model_path=operator_path)
