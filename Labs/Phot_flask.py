#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JWST Labs - Photometry Lab (multi-user safe)

Fix: all dataset data is pre-loaded into a read-only cache at startup.
Callbacks are pure functions that never write to self, so concurrent
users cannot overwrite each other's state.
"""

import sys
import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from astropy.modeling.models import Sersic2D
from astropy.convolution import Gaussian2DKernel, convolve_fft
import astropy.io.fits as pyfits
import astropy.stats as stats


class JADES_photo_lab:
    def __init__(self, server, requests_pathname_prefix, routes_pathname_prefix, app_name=None):
        self.app = dash.Dash(
            app_name or __name__,
            server=server,
            requests_pathname_prefix=requests_pathname_prefix,
            routes_pathname_prefix=routes_pathname_prefix,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        self.app.title = "JWST Photometry Lab"

        self.image_lim = 20
        self.initial_dataset = 'Galaxy1'

        self.data_files = {
            'Galaxy1': {'file': 'hlsp_jades_jwst_nirspec_goods-s-mediumjwst-00188208_clear-prism_v1.0_x1d.fits', 'target': 'generic'},
            'Galaxy2': {'file': 'hlsp_jades_jwst_nirspec_goods-s-mediumjwst-00003204_clear-prism_v1.0_x1d.fits', 'target': 'generic'},
            'GN-z11':  {'file': 'hlsp_jades_jwst_nirspec_goods-n-mediumhst-00003991_clear-prism_v1.0_x1d.fits', 'target': 'generic'},
            'GSz14':   {'file': 'hlsp_jades_jwst_nirspec_goods-s-deepjwst-00183348_clear-prism_v1.0_x1d.fits',  'target': 'GSz14'},
            'Nrb':     {'file': 'hlsp_jades_jwst_nirspec_goods-s-mediumjwst-00200212_clear-prism_v1.0_x1d.fits', 'target': 'generic'},
            'QC':      {'file': 'hlsp_jades_jwst_nirspec_goods-n-mediumhst-00023924_clear-prism_v1.0_x1d.fits',  'target': 'generic'},
            'PSB':     {'file': 'hlsp_jades_jwst_nirspec_goods-n-mediumhst-00024824_clear-prism_v1.0_x1d.fits',  'target': 'generic'},
            'Galaxy3': {'file': 'hlsp_jades_jwst_nirspec_goods-n-mediumjwst-10004051_clear-prism_v1.0_x1d.fits', 'target': 'low_snr'},
        }

        # Pre-load ALL datasets once at startup into a read-only cache.
        # FITS files never change, so this is safe to share across all users/workers.
        self._cache = {}
        for key in self.data_files:
            try:
                self._cache[key] = self._load_dataset(key)
                print(f"Phot_flask: loaded dataset '{key}'")
            except Exception as e:
                print(f"Phot_flask: failed to load dataset '{key}': {e}")

        self.setup_layout()
        self.setup_callbacks()

    # ------------------------------------------------------------------
    # Data loading (called once per dataset at startup, result is cached)
    # ------------------------------------------------------------------
    def _load_dataset(self, key):
        """Load a FITS file and return an immutable dict. Never call from a callback."""
        config = self.data_files[key]
        pth = sys.path[0] if sys.path[0] else '.'
        filepath = os.path.join(pth, 'Data/phot', config['file'])
        lim = self.image_lim
        with pyfits.open(filepath) as hdu:
            image = hdu['F444W'].data
            image = image[84 - lim: 84 + lim + 1, 84 - lim: 84 + lim + 1]
            header = hdu['F444W'].header
        image_error = stats.sigma_clipped_stats(image, sigma=3.0, maxiters=10)[2] * np.ones_like(image)
        psf_pixel = 0.145 / (header['CDELT1'] * 3600) / 2.355
        psf_kernel = Gaussian2DKernel(psf_pixel)
        return {
            'image':       image,
            'image_error': image_error,
            'shape':       image.shape,
            'PSF_kernel':  psf_kernel,
            'target':      config['target'],
        }

    # ------------------------------------------------------------------
    # Pure computation helpers — no self state is read or written
    # ------------------------------------------------------------------
    def _generate_model(self, data, amplitude, radius, index, theta, x, y, ellipticity):
        lim = self.image_lim
        shape = data['shape']
        xg, yg = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        model = Sersic2D(
            amplitude=amplitude, r_eff=radius, n=index,
            x_0=x + lim, y_0=y + lim,
            ellip=ellipticity, theta=np.deg2rad(theta),
        )
        model_image = convolve_fft(model(xg, yg), data['PSF_kernel'])
        residual = (data['image'] - model_image) / data['image_error']
        return model_image, residual

    @staticmethod
    def _calculate_score(data, model_image):
        chi2 = np.nansum(((data['image'] - model_image) / data['image_error']) ** 2)
        dof = np.sum(~np.isnan(data['image'])) - 7
        return chi2 / dof if dof > 0 else np.nan

    @staticmethod
    def _create_plot(data, model_image, residual, score):
        vmin = np.percentile(data['image'], 1)
        vmax = np.percentile(data['image'], 99.5)

        def asinh_stretch(d):
            a = 0.1
            return np.arcsinh((d - vmin) / (vmax - vmin) / a) / np.arcsinh(1.0 / a)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Observed Image", "Your Model", "Difference"),
            horizontal_spacing=0.1,
        )
        fig.add_trace(go.Heatmap(z=asinh_stretch(data['image']), colorscale='Viridis',
                                  showscale=True, colorbar=dict(x=0.30, len=0.8, title="Brightness")),
                      row=1, col=1)
        fig.add_trace(go.Heatmap(z=asinh_stretch(model_image), colorscale='Viridis',
                                  showscale=True, colorbar=dict(x=0.63, len=0.8, title="Brightness")),
                      row=1, col=2)
        fig.add_trace(go.Heatmap(z=residual, colorscale='RdBu_r', showscale=True,
                                  zmid=0, zmin=-5, zmax=5,
                                  colorbar=dict(x=1.0, len=0.8, title="Residual")),
                      row=1, col=3)
        fig.update_layout(
            title=f"JWST Photometry Fit — score = {score:.2f} (lower is better)",
            template="plotly_white", height=600, showlegend=False,
        )
        for col in range(1, 4):
            fig.update_xaxes(scaleanchor=f"y{'' if col == 1 else col}", scaleratio=1,
                             constrain="domain", row=1, col=col)
            fig.update_yaxes(constrain="domain", row=1, col=col)
        return fig

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def setup_layout(self):
        self.app.layout = dbc.Container([
            # Per-session state — each browser tab gets its own Store
            dcc.Store(id='active-dataset', data=self.initial_dataset),

            dbc.Row([dbc.Col(html.H1("JADES Photometry Lab", className="text-center mb-4"), width=12)]),
            dbc.Row([
                dbc.Col(
                    html.A("📖 Learn the science behind this lab",
                           href="../learn-photometry", target="_blank",
                           style={'color': '#0066cc', 'fontWeight': 'bold', 'textDecoration': 'none'}),
                    width=12, className="text-center mb-3"
                )
            ]),
            dbc.Row([dbc.Col(
                dbc.ButtonGroup([dbc.Button(k, id=f"btn-{k}", color="info", size="sm")
                                 for k in self.data_files]),
                width=12, className="mb-3",
            )]),
            dbc.Row([dbc.Col(dcc.Graph(id="main-plot", style={'height': '600px'}), width=12)],
                    style={'margin-bottom': '60px'}),

            dbc.Row([
                dbc.Col([
                    _labeled_slider("Peak size",       "tooltip-amp",   "Making the galaxy brighter/fainter",
                                    "amp-slider",      0.1, 10, 0.01, 0.5, {i: str(i) for i in range(11)}),
                    _labeled_slider("Size (pix)",      "tooltip-size",  "Size of the galaxy in pixels",
                                    "size-slider",     0, 10, 0.1, 5, {i: str(i) for i in range(11)}),
                    _labeled_slider("Sersic index",    "tooltip-sersic","n=1 disk, n=4 elliptical",
                                    "Sersic-slider",   0.0, 5, 0.1, 2, {0:'0',1:'1',2:'2',3:'3',4:'4'}),
                    _labeled_slider("Angle",           "tooltip-angle", "Orientation on sky",
                                    "angle-slider",    0, 180, 1, 90,
                                    {0:'horizontal',45:'45',90:'vertical',135:'135',180:'horizontal'}),
                ], width=6),
                dbc.Col([
                    _labeled_slider("X",               "tooltip-x",     "X-offset of galaxy centre (px)",
                                    "x-slider",        -5, 5, 0.1, 0, {i: str(i) for i in range(-5, 6)}),
                    _labeled_slider("Y",               "tooltip-y",     "Y-offset of galaxy centre (px)",
                                    "y-slider",        -5, 5, 0.1, 0, {i: str(i) for i in range(-5, 6)}),
                    _labeled_slider("Ellipticity",     "tooltip-ellip", "0=circular, 1=elongated",
                                    "ellipticity-slider", 0, 1, 0.01, 0.5, {0:'Circle',0.5:'0.5',1:'Line'}),
                ], width=6),
            ], className="mt-3"),
        ], fluid=True)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def setup_callbacks(self):
        @self.app.callback(
            Output("main-plot",      "figure"),
            Output("active-dataset", "data"),
            [Input("amp-slider",         "value"),
             Input("size-slider",        "value"),
             Input("Sersic-slider",      "value"),
             Input("angle-slider",       "value"),
             Input("x-slider",           "value"),
             Input("y-slider",           "value"),
             Input("ellipticity-slider", "value")]
            + [Input(f"btn-{k}", "n_clicks") for k in self.data_files],
            State("active-dataset", "data"),
            prevent_initial_call=False,
        )
        def update_app(amplitude, radius, index, theta, x, y, ellipticity,
                       *btn_and_state):
            # Unpack button clicks and the State value (last element)
            btn_clicks      = btn_and_state[:-1]
            active_dataset  = btn_and_state[-1] or self.initial_dataset

            # Determine whether a dataset button was clicked
            ctx = callback_context
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                for i, key in enumerate(self.data_files):
                    if trigger_id == f"btn-{key}" and btn_clicks[i]:
                        active_dataset = key
                        break

            # All computation is local — no writes to self
            data        = self._cache.get(active_dataset, self._cache[self.initial_dataset])
            model_image, residual = self._generate_model(data, amplitude, radius, index,
                                                         theta, x, y, ellipticity)
            score = self._calculate_score(data, model_image)
            fig   = self._create_plot(data, model_image, residual, score)
            return fig, active_dataset


# ------------------------------------------------------------------
# Small helper to reduce layout boilerplate
# ------------------------------------------------------------------
def _labeled_slider(label, tooltip_id, tooltip_text, slider_id, mn, mx, step, value, marks):
    return html.Div([
        html.Label([
            label + " ",
            html.Span("ℹ", id=tooltip_id,
                      style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"}),
        ], className="fw-bold mb-2"),
        dbc.Tooltip(tooltip_text, target=tooltip_id, placement="right"),
        dcc.Slider(id=slider_id, min=mn, max=mx, step=step, value=value,
                   marks=marks, tooltip={"placement": "bottom", "always_visible": True}),
    ], className="mb-4")