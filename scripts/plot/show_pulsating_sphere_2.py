import pathlib
import time

import dash
import numpy as np
import plotly.express as px
import torch
from dash import (
    Input,
    Output,
    dcc,
    html,
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

# Initialize the Dash app
app = dash.Dash(__name__)
train_dataset = PulsatingSphere(pathlib.Path.cwd().joinpath("data", "train", "pulsating_sphere_narrow"))
operator = deserialize(
    pathlib.Path.cwd().joinpath(
        "finished_pressure",
        "parf",
        "ddo",
        "2024_05_10_01_01_30-c79b7adc-4191-4900-a7d1-2bf5cad69e09",
        "best_mean",
        "DeepDotOperator_2024_05_10_01_24_38",
    )
)


def create_images(y1_real: float, y1_imag: float, y2_real: float, y2_imag: float, f: float, res: int):
    u_plot = torch.tensor([y1_real, y1_imag, y2_real, y2_imag, f]).reshape(1, 1, -1)
    x = u = train_dataset.transform["u"](u_plot)

    x_plot, y_plot = torch.meshgrid(torch.linspace(0, 1, res + 1)[1:], torch.linspace(0, 1, res + 1)[1:])
    y = torch.cat([x_plot.flatten().reshape(1, -1, 1), y_plot.flatten().reshape(1, -1, 1)], dim=2)
    y_trf = train_dataset.transform["y"](y)

    start = time.time()
    v = operator(x, u, y_trf)
    end = time.time()
    delta = (end - start) * 1e3
    logger.info(f"Forward pass time: {delta: .2f}ms, Scalar forward: {delta / v.nelement(): .4f}ms")

    # v_plot = train_dataset.transform["v"].undo(v)
    v_plot = v
    v_plot = v_plot.reshape(1, -1, 2)
    v_plot = v_plot.reshape(res, res, 2)

    v_data = v_plot.detach().numpy()

    real_image = np.block(
        [
            [np.flip(np.flip(v_data[:, :, 0], axis=1), axis=0), np.flip(v_data[:, :, 0], axis=0)],
            [np.flip(v_data[:, :, 0], axis=1), v_data[:, :, 0]],
        ]
    ).T

    imag_image = np.block(
        [
            [np.flip(np.flip(v_data[:, :, 1], axis=1), axis=0), np.flip(v_data[:, :, 1], axis=0)],
            [np.flip(v_data[:, :, 1], axis=1), v_data[:, :, 1]],
        ]
    ).T

    print(real_image.shape)

    return real_image, imag_image


app.layout = html.Div(
    [
        dcc.Slider(id="Re(Y_1)", min=0, max=1, value=0.5, tooltip={"placement": "bottom", "always_visible": False}),
        dcc.Slider(id="Im(Y_1)", min=-1, max=0, value=-0.5, tooltip={"placement": "bottom", "always_visible": False}),
        dcc.Slider(id="Re(Y_2)", min=0, max=1, value=0.5, tooltip={"placement": "bottom", "always_visible": False}),
        dcc.Slider(id="Im(Y_2)", min=-1, max=0, value=-0.5, tooltip={"placement": "bottom", "always_visible": False}),
        dcc.Slider(
            id="frequency", min=400, max=500, value=450, tooltip={"placement": "bottom", "always_visible": False}
        ),
        dcc.Slider(
            id="resolution", min=10, max=200, value=50, tooltip={"placement": "bottom", "always_visible": False}
        ),
        html.Div(
            id="image-container",
            children=[
                html.Div(
                    [
                        dcc.Graph(id="image-real"),
                    ],
                    style={"display": "inline-block", "width": "49%"},
                ),  # Graph container for image-1
                html.Div(
                    [
                        dcc.Graph(id="image-imag"),
                    ],
                    style={"display": "inline-block", "width": "49%"},
                ),  # Graph container for image-2
            ],
            style={"width": "100%", "display": "flex"},
        ),
    ]
)


@app.callback(
    [Output("image-real", "figure"), Output("image-imag", "figure")],
    [
        Input("Re(Y_1)", "value"),
        Input("Im(Y_1)", "value"),
        Input("Re(Y_2)", "value"),
        Input("Im(Y_2)", "value"),
        Input("frequency", "value"),
        Input("resolution", "value"),
    ],
)
def update_images(s1, s2, s3, s4, s5, res):
    # Update this function based on the sliders to apply transformations
    real, imag = create_images(s1, s2, s3, s4, s5, int(res))

    abs_real_max = np.max(np.abs(real))
    fig1 = px.imshow(
        real,
        zmin=-abs_real_max,
        zmax=abs_real_max,
        color_continuous_scale="RdBu",
        x=np.linspace(-1, 1, 2 * int(res)),
        y=np.linspace(-1, 1, 2 * int(res)),
        origin="lower",
    )
    fig1.add_shape(type="circle", xref="x", yref="y", x0=-0.2, y0=-0.2, x1=0.2, y1=0.2, fillcolor="Black")
    abs_imag_max = np.max(np.abs(imag))
    fig2 = px.imshow(
        imag,
        zmin=-abs_imag_max,
        zmax=abs_imag_max,
        color_continuous_scale="RdBu",
        x=np.linspace(-1, 1, 2 * int(res)),
        y=np.linspace(-1, 1, 2 * int(res)),
        origin="lower",
    )
    fig2.add_shape(type="circle", xref="x", yref="y", x0=-0.2, y0=-0.2, x1=0.2, y1=0.2, fillcolor="Black")

    fig1.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    fig2.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))

    return fig1, fig2


if __name__ == "__main__":
    app.run_server(debug=True)
