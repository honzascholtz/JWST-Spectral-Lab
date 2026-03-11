from flask import Flask, render_template_string
import sys
import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from astropy.cosmology import Planck18 as cosmo
import astropy.io.fits as pyfits


# ============================================================================
# APP 2: REDSHIFT LAB
# ============================================================================
class Redshift_lab:
    def __init__(self, server, requests_pathname_prefix, routes_pathname_prefix, app_name=None):
        
        """Initialize the Redshift Dash application with Flask server"""
        self.app = dash.Dash(
            app_name or __name__,          # use unique name if provided
            server=server,
            requests_pathname_prefix=requests_pathname_prefix,
            routes_pathname_prefix=routes_pathname_prefix,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        self.app.title = "JWST Redshift Lab"
        
        self.data_wave = None
        self.data_flux = None
        self.data_error = None
        self.ztrue = 9.436
        self.z = 1
        self.target = 'generic'
        
        self.data_files = {
            'SF943': {'file': '10058975_prism_clear_v5.0_1D.fits', 'ztrue': 9.436, 'target': 'generic'},
            'QC_galaxy': {'file': '199773_prism_clear_v5.0_1D.fits', 'ztrue': 2.820, 'target': 'generic'},
            'SF_galaxy': {'file': '001882_prism_clear_v5.0_1D.fits', 'ztrue': 5.4431, 'target': 'generic'},
            'GSz14': {'file': '183348_prism_clear_v5.0_1D.fits', 'ztrue': 14.18, 'target': 'GSz14'},
            'COS30': {'file': '007437_prism_clear_v3.1_1D.fits', 'ztrue': 6.856, 'target': 'generic'},
            'SF2': {'file': '001927_prism_clear_v5.0_1D.fits', 'ztrue': 3.6591, 'target': 'generic'},
            'PSB': {'file': '023286_prism_clear_v5.1_1D.fits', 'ztrue': 1.781, 'target': 'generic'},
            'zhig': {'file': '066585_prism_clear_v5.1_1D.fits', 'ztrue': 7.1404, 'target': 'low_snr'},
            'zhig2': {'file': '003991_prism_clear_v5.1_1D.fits', 'ztrue': 10.603, 'target': 'gnz11'},
            'zhig3': {'file': ' 001936_prism_clear_v5.1_1D.fits', 'ztrue': 7.08989, 'target': 'gnz11'},
           
        }
        
        self.load_data('SF943')
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self, dataset_key):
        config = self.data_files[dataset_key]
        self.ztrue = config['ztrue']
        self.target = config['target']
        
        try:
            pth = sys.path[0] if sys.path[0] else '.'
            filepath = os.path.join(pth, 'Data', config['file'])
            with pyfits.open(filepath) as hdu:
                self.data_wave = hdu['WAVELENGTH'].data * 1e6
                self.data_flux = hdu['DATA'].data * 1e-7
                self.data_error = hdu['ERR'].data * 1e-7
            
            if dataset_key == 'COS30':
                self.data_wave = np.append(self.data_wave, np.linspace(5.32, 5.5, 32))
                self.data_flux = np.append(self.data_flux, np.zeros(32))
                self.data_error = np.append(self.data_error, np.ones(32) * 0.001e-18)
        except Exception as e:
            print(f"Error loading {config['file']}: {e}")
    
    def get_emission_lines(self):
        emlines = {
            r'C⁺⁺': (1907., 'red'), r'Mg⁺': (2797., 'blue'), r'[O⁺]': (3728., 'green'),
            r'[Ne⁺⁺]': (3869.860, 'purple'), 'Hδ': (4102.860, 'orange'), 'Hγ': (4341.647, 'pink'),
            'Hβ': (4862.647, 'brown'), r'': (4960.0, 'red'), r'[O⁺⁺]': (5008.0, 'red'),
            r'[O⁰]': (6302.0, 'green'), 'Na': (5891.583, 'yellow'), 'Hα': (6564.522, 'red'),
            r'[S⁺]': (6725, 'blue'), r'[S⁺⁺]': (9070.0, 'purple'), 'HeI': (10832.1, 'orange'),
            'Paγ': (10940.978, 'pink'), r'[Fe⁺]': (12570.200, 'brown'), 'Paβ': (12821.432, 'green'),
            'Paα': (18755.80357, 'red')
        }
        
        visible_lines = {}
        for line_name, (rest_wave, color) in emlines.items():
            obs_wave = rest_wave * (1 + self.z) / 1.e4
            if 0.5 * 1.001 < obs_wave < 5.5 * 0.999:
                visible_lines[line_name] = (obs_wave, color)
        return visible_lines
    
    def setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([dbc.Col([html.H1("JADES Redshift Lab", className="text-center mb-4")], width=12)]),
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([dbc.Button(key, id=f"btn-{key}", color="info", size="sm") 
                                    for key in self.data_files.keys()], className="mb-3")
                ], width=12)
            ]),
            dbc.Row([dbc.Col([dcc.Graph(id="main-plot", style={'height': '460px'})], width=10)]),
            dbc.Row([
                dbc.Col([
                    html.Label("Redshift", className="fw-bold"),
                    dcc.Slider(id="redshift-slider", min=1, max=15, step=0.001, value=1,
                              marks={i: str(i) for i in range(1, 16)},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], width=10),
                dbc.Col([dbc.Button("Show Score", id="show-score-btn", color="success", className="mt-4")], width=2)
            ], className="mt-3"),
            dbc.Row([dbc.Col([html.Div(id="score-display", className="mt-3")], width=12)]),
            dcc.Store(id="show-score-state", data=False)
        ], fluid=True)
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output("main-plot", "figure"), Output("score-display", "children"), Output("show-score-state", "data")],
            [Input("redshift-slider", "value"), Input("show-score-btn", "n_clicks")] + 
            [Input(f"btn-{key}", "n_clicks") for key in self.data_files.keys()],
            [State("show-score-state", "data")],
            prevent_initial_call=False
        )
        def update_app(*args):
            ctx = callback_context
            if not ctx.triggered:
                self.z = args[0]
                return self.create_plot(), "", False
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == "show-score-btn":
                self.z = args[0]
                score = (self.z - self.ztrue) / (1 + self.ztrue) * 3e5
                score_text = dbc.Alert([
                    html.P("Aim to get the score as close to 0 km/s as possible.", className="mb-2"),
                    html.P("Score between -1000 and 1000 is amazing!", className="mb-2"),
                    html.H4(f"Score: {score:.2f} km/s", className="alert-heading")
                ], color="success" if abs(score) < 1000 else "warning" if abs(score) < 3000 else "danger")
                return self.create_plot(), score_text, True
            elif trigger_id == "redshift-slider":
                self.z = args[0]
                return self.create_plot(), "", False
            else:
                for i, key in enumerate(self.data_files.keys()):
                    if trigger_id == f"btn-{key}" and args[2 + i]:
                        self.load_data(key)
                        self.z = args[0]
                        return self.create_plot(), "", False
            return dash.no_update, dash.no_update, dash.no_update
    
    def create_plot(self):
        if self.data_wave is None:
            return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data_wave, y=self.data_flux / 1e-18, mode='lines',
                                line=dict(color='black', shape='hv'), name='Spectrum', showlegend=False))
        
        emission_lines = self.get_emission_lines()
        for line_name, (obs_wave, color) in emission_lines.items():
            fig.add_vline(x=obs_wave, line=dict(color=color, dash='dash', width=2), opacity=0.7)
            fig.add_annotation(x=obs_wave, y=0.95, text=line_name, textangle=90, showarrow=False,
                             yref='paper', bgcolor='white', bordercolor=color, font=dict(size=16))
        
        fig.update_layout(xaxis_title="Wavelength (μm) - blue ← → red", yaxis_title="Brightness (×10⁻¹⁸)",
                         xaxis=dict(range=[0.5, 5.3]), template="plotly_white", height=500)
        
        if self.target == 'GSz14':
            fig.update_yaxes(range=[-0.00025, 0.01])
        elif self.target == 'gnz11':
            fig.update_yaxes(range=[-0.01, 0.04])
        elif self.target == 'low_snr':
            fig.update_yaxes(range=[-0.01, 0.025])
        
        return fig