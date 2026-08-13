#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JWST Labs - Redshift Lab (multi-user safe)

Fix: all dataset data is pre-loaded into a read-only cache at startup.
Callbacks are pure functions that never write to self, so concurrent
users cannot overwrite each other's state.
"""

import sys
import os
import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

import astropy.io.fits as pyfits


class Redshift_lab:
    def __init__(self, server, requests_pathname_prefix, routes_pathname_prefix, app_name=None):
        self.app = dash.Dash(
            app_name or __name__,
            server=server,
            requests_pathname_prefix=requests_pathname_prefix,
            routes_pathname_prefix=routes_pathname_prefix,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        self.app.title = "JWST Redshift Lab"

        self.initial_dataset = 'SF943'

        self.data_files = {
            'SF943':    {'file': '10058975_prism_clear_v5.0_1D.fits',  'ztrue': 9.436,   'target': 'generic'},
            'QC_galaxy':{'file': '199773_prism_clear_v5.0_1D.fits',    'ztrue': 2.820,   'target': 'generic'},
            'SF_galaxy':{'file': '001882_prism_clear_v5.0_1D.fits',    'ztrue': 5.4431,  'target': 'generic'},
            'GSz14':    {'file': '183348_prism_clear_v5.0_1D.fits',    'ztrue': 14.18,   'target': 'GSz14'},
            'COS30':    {'file': '007437_prism_clear_v3.1_1D.fits',    'ztrue': 6.856,   'target': 'generic'},
            'SF2':      {'file': '001927_prism_clear_v5.0_1D.fits',    'ztrue': 3.6591,  'target': 'generic'},
            'PSB':      {'file': '023286_prism_clear_v5.1_1D.fits',    'ztrue': 1.781,   'target': 'generic'},
            'zhig':     {'file': '066585_prism_clear_v5.1_1D.fits',    'ztrue': 7.1404,  'target': 'low_snr'},
            'zhig2':    {'file': '003991_prism_clear_v5.1_1D.fits',    'ztrue': 10.603,  'target': 'gnz11'},
            'zhig3':    {'file': '001936_prism_clear_v5.1_1D.fits',    'ztrue': 7.08989, 'target': 'gnz11'},
        }

        # Pre-load all datasets once at startup into a read-only cache
        self._cache = {}
        for key in self.data_files:
            try:
                self._cache[key] = self._load_dataset(key)
                print(f"Redshift_lab: loaded dataset '{key}'")
            except Exception as e:
                print(f"Redshift_lab: failed to load dataset '{key}': {e}")

        self.setup_layout()
        self.setup_callbacks()

    # ------------------------------------------------------------------
    # Data loading — called once per dataset at startup
    # ------------------------------------------------------------------
    def _load_dataset(self, key):
        config = self.data_files[key]
        pth = sys.path[0] if sys.path[0] else '.'
        filepath = os.path.join(pth, 'Data', config['file'])
        with pyfits.open(filepath) as hdu:
            data_wave  = hdu['WAVELENGTH'].data * 1e6
            data_flux  = hdu['DATA'].data * 1e-7
            data_error = hdu['ERR'].data * 1e-7
        if key == 'COS30':
            data_wave  = np.append(data_wave,  np.linspace(5.32, 5.5, 32))
            data_flux  = np.append(data_flux,  np.zeros(32))
            data_error = np.append(data_error, np.ones(32) * 0.001e-18)
        return {
            'data_wave':  data_wave,
            'data_flux':  data_flux,
            'data_error': data_error,
            'ztrue':      config['ztrue'],
            'target':     config['target'],
        }

    # ------------------------------------------------------------------
    # Pure helpers — no self state read or written
    # ------------------------------------------------------------------
    @staticmethod
    def _get_emission_lines(z):
        emlines = {
            r'C⁺⁺':    (1907.,     'red'),
            r'Mg⁺':    (2797.,     'blue'),
            r'[O⁺]':   (3728.,     'green'),
            r'[Ne⁺⁺]': (3869.860,  'purple'),
            'Hδ':       (4102.860,  'orange'),
            'Hγ':       (4341.647,  'pink'),
            'Hβ':       (4862.647,  'brown'),
            r'[O⁺⁺]a': (4960.0,   'red'),
            r'[O⁺⁺]b': (5008.0,   'red'),
            r'[O⁰]':   (6302.0,   'green'),
            'Na':       (5891.583,  'yellow'),
            'Hα':       (6564.522,  'red'),
            r'[S⁺]':   (6725.,     'blue'),
            r'[S⁺⁺]':  (9070.0,   'purple'),
            'HeI':      (10832.1,   'orange'),
            'Paγ':      (10940.978, 'pink'),
            r'[Fe⁺]':  (12570.200, 'brown'),
            'Paβ':      (12821.432, 'green'),
            'Paα':      (18755.804, 'red'),
        }
        visible = {}
        for name, (rest_wave, color) in emlines.items():
            obs_wave = rest_wave * (1 + z) / 1e4
            if 0.5 * 1.001 < obs_wave < 5.5 * 0.999:
                visible[name] = (obs_wave, color)
        return visible

    @staticmethod
    def _create_plot(data, z):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['data_wave'], y=data['data_flux'] / 1e-18,
            mode='lines', line=dict(color='black', shape='hv'),
            name='Spectrum', showlegend=False,
        ))
        for line_name, (obs_wave, color) in Redshift_lab._get_emission_lines(z).items():
            fig.add_vline(x=obs_wave, line=dict(color=color, dash='dash', width=2), opacity=0.7)
            fig.add_annotation(x=obs_wave, y=0.95, text=line_name, textangle=90,
                               showarrow=False, yref='paper', bgcolor='white',
                               bordercolor=color, font=dict(size=16))
        fig.update_layout(
            xaxis_title="Wavelength (μm) — blue ← → red",
            yaxis_title="Brightness (×10⁻¹⁸)",
            xaxis=dict(range=[0.5, 5.3]),
            template="plotly_white", height=500,
        )
        target = data['target']
        if target == 'GSz14':
            fig.update_yaxes(range=[-0.00025, 0.01])
        elif target == 'gnz11':
            fig.update_yaxes(range=[-0.01, 0.04])
        elif target == 'low_snr':
            fig.update_yaxes(range=[-0.01, 0.025])
        return fig

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def setup_layout(self):
        self.app.layout = dbc.Container([
            # Per-session state lives in the browser, not on the server
            dcc.Store(id='active-dataset', data=self.initial_dataset),
            dcc.Store(id='show-score-state', data=False),

            dbc.Row([dbc.Col(html.H1("JADES Redshift Lab", className="text-center mb-4"), width=12)]),
            dbc.Row([
                dbc.Col(
                    html.A("📖 Learn the science behind this lab",
                           href="../learn-redshift", target="_blank",
                           style={'color': '#b22222', 'fontWeight': 'bold', 'textDecoration': 'none'}),
                    width=12, className="text-center mb-3"
                )
            ]),
            dbc.Row([dbc.Col(
                dbc.ButtonGroup([dbc.Button(k, id=f"btn-{k}", color="info", size="sm")
                                 for k in self.data_files]),
                width=12, className="mb-3",
            )]),
            dbc.Row([dbc.Col(dcc.Graph(id="main-plot", style={'height': '460px'}), width=10)]),
            dbc.Row([
                dbc.Col([
                    html.Label("Redshift", className="fw-bold"),
                    dcc.Slider(id="redshift-slider", min=1, max=15, step=0.001, value=1,
                               marks={i: str(i) for i in range(1, 16)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], width=10),
                dbc.Col(dbc.Button("Show Score", id="show-score-btn", color="success",
                                   className="mt-4"), width=2),
            ], className="mt-3"),
            dbc.Row([dbc.Col(html.Div(id="score-display", className="mt-3"), width=12)]),
        ], fluid=True)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def setup_callbacks(self):
        @self.app.callback(
            Output("main-plot",        "figure"),
            Output("score-display",    "children"),
            Output("show-score-state", "data"),
            Output("active-dataset",   "data"),
            Input("redshift-slider",  "value"),
            Input("show-score-btn",   "n_clicks"),
            *[Input(f"btn-{k}", "n_clicks") for k in self.data_files],
            State("active-dataset",   "data"),
            State("show-score-state", "data"),
            prevent_initial_call=False,
        )
        def update_app(z, score_btn_clicks, *btn_and_states):
            # Unpack: N dataset button clicks, then 2 State values
            n = len(self.data_files)
            btn_clicks     = btn_and_states[:n]
            active_dataset = btn_and_states[n] or self.initial_dataset
            show_score     = btn_and_states[n + 1]

            ctx = callback_context
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

            # Determine active dataset from button clicks
            if trigger_id:
                for i, key in enumerate(self.data_files):
                    if trigger_id == f"btn-{key}" and btn_clicks[i]:
                        active_dataset = key
                        show_score = False   # reset score display on dataset change
                        break

            data = self._cache.get(active_dataset, self._cache[self.initial_dataset])
            fig  = self._create_plot(data, z)

            # Score display
            if trigger_id == "show-score-btn":
                velocity_offset = (z - data['ztrue']) / (1 + data['ztrue']) * 3e5
                color = ("success"  if abs(velocity_offset) < 1000
                         else "warning" if abs(velocity_offset) < 3000
                         else "danger")
                score_content = dbc.Alert([
                    html.P("Aim to get the score as close to 0 km/s as possible.", className="mb-2"),
                    html.P("Score between −1000 and 1000 is amazing!", className="mb-2"),
                    html.H4(f"Score: {velocity_offset:.2f} km/s", className="alert-heading"),
                ], color=color)
                show_score = True
            elif trigger_id in ("redshift-slider", None) or trigger_id.startswith("btn-"):
                score_content = ""
                show_score = False
            else:
                score_content = ""

            return fig, score_content, show_score, active_dataset