#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JWST Labs - Flask Integration with Multiple Dash Apps
Three separate Dash apps integrated with Flask

@author: jansen (converted to Flask)
"""

from flask import Flask, render_template_string
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
from astropy.cosmology import Planck18 as cosmo
import astropy.io.fits as pyfits
import astropy.stats as stats


# ============================================================================
# APP 1: PHOTOMETRY LAB
# ============================================================================
class JADES_photo_lab:
    def __init__(self, server, requests_pathname_prefix, routes_pathname_prefix):
        """Initialize the Photometry Dash application with Flask server"""
        self.app = dash.Dash(
            __name__, 
            server=server,
            requests_pathname_prefix=requests_pathname_prefix,
            routes_pathname_prefix=routes_pathname_prefix,
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self.app.title = "JWST Photometry Lab"
        
        # Initialize data variables
        self.amplitude = 1
        self.radius = 10
        self.index = 4
        self.x = 0
        self.y = 0
        self.ellipticity = 0.5
        self.theta = -1
        self.target = 'generic'
        self.score = 0
        self.image_lim = 20
        
        # Data file configurations
        self.data_files = {
            'Galaxy1': {'file': 'hlsp_jades_jwst_nirspec_goods-s-mediumjwst-00188208_clear-prism_v1.0_x1d.fits', 'z': 9.436, 'target': 'generic'},
            'Galaxy2': {'file': 'hlsp_jades_jwst_nirspec_goods-s-mediumjwst-00003204_clear-prism_v1.0_x1d.fits', 'z': 2.820, 'target': 'generic'},
            'SF_galaxy': {'file': '001882_prism_clear_v5.0_1D.fits', 'z': 5.4431, 'target': 'generic'},
            'GSz14': {'file': '183348_prism_clear_v5.0_1D.fits', 'z': 14.18, 'target': 'GSz14'},
            'COS30': {'file': '007437_prism_clear_v3.1_1D.fits', 'z': 6.856, 'target': 'generic'},
            'SF2': {'file': '001927_prism_clear_v5.0_1D.fits', 'z': 3.6591, 'target': 'generic'},
            'PSB': {'file': '023286_prism_clear_v5.1_1D.fits', 'z': 1.781, 'target': 'generic'},
            'zhig': {'file': '066585_prism_clear_v5.1_1D.fits', 'z': 7.1404, 'target': 'low_snr'},
        }
        
        self.load_data('Galaxy1')
        self.generate_model()
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self, dataset_key):
        config = self.data_files[dataset_key]
        self.z = config['z']
        self.target = config['target']
        
        pth = sys.path[0] if sys.path[0] else '.'
        filepath = os.path.join(pth, 'Data/phot', config['file'])
        with pyfits.open(filepath) as hdu:
            self.image = hdu['F444W'].data
            self.image = self.image[84-self.image_lim:84+self.image_lim+1, 84-self.image_lim:84+self.image_lim+1]
            self.image_header = hdu['F444W'].header
            self.image_error = stats.sigma_clipped_stats(self.image, sigma=3.0, maxiters=10)[2] * np.ones_like(self.image)
            self.shape = self.image.shape
            self.psf_pixel = 0.145/(self.image_header['CDELT1']*3600) / 2.355
            self.PSF_kernel = Gaussian2DKernel(self.psf_pixel)
    
    def generate_model(self):
        x, y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        self.model = Sersic2D(
            amplitude=self.amplitude, r_eff=self.radius, n=self.index,
            x_0=self.x+self.image_lim, y_0=self.y+self.image_lim, 
            ellip=self.ellipticity, theta=np.deg2rad(self.theta)
        )
        self.model_image = self.model(x, y)
        self.model_image = convolve_fft(self.model_image, self.PSF_kernel)
        self.residual = (self.image - self.model_image)/self.image_error
    
    def calculate_score(self):
        chi_squared = np.nansum(((self.image - self.model_image) / self.image_error) ** 2)
        dof = np.sum(~np.isnan(self.image)) - 7
        self.score = chi_squared / dof if dof > 0 else np.nan
        return self.score
    
    def setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([dbc.Col([html.H1("JADES Photometry Lab", className="text-center mb-4")], width=12)]),
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(key, id=f"btn-{key}", color="info", size="sm") 
                        for key in self.data_files.keys()
                    ], className="mb-3")
                ], width=12)
            ]),
            dbc.Row([dbc.Col([dcc.Graph(id="main-plot", style={'height': '600px'})], width=12)], style={'margin-bottom': '60px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label([
                            "Peak size ",
                            html.Span("ℹ", id="tooltip-amp", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("Making the galaxy brighter/fainter", target="tooltip-amp", placement="right"),
                        dcc.Slider(id="amp-slider", min=0.1, max=10, step=0.01, value=0.5,
                                marks={i: str(i) for i in range(11)},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label([
                            "Size (pix) ",
                            html.Span("ℹ", id="tooltip-size", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("What is the size of the galaxy in pixels?", target="tooltip-size", placement="right"),
                        dcc.Slider(id="size-slider", min=0, max=10, step=0.1, value=5,
                                marks={i: str(i) for i in range(11)},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label([
                            "Sersic index ",
                            html.Span("ℹ", id="tooltip-sersic", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("Shape parameter: n=1 (exponential disk), n=4 (de Vaucouleurs/elliptical)", target="tooltip-sersic", placement="right"),
                        dcc.Slider(id="Sersic-slider", min=0.0, max=5, step=0.1, value=2,
                                marks={0: '0', 1: '1', 2: '2', 3: '3', 4: '4'},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label([
                            "Angle ",
                            html.Span("ℹ", id="tooltip-angle", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("How is the galaxy angled on the sky?", target="tooltip-angle", placement="right"),
                        dcc.Slider(id="angle-slider", min=0.0, max=180, step=1, value=90,
                                marks={0: 'horizontal', 45: '45', 90: 'vertical', 135: '135', 180: 'horizontal'},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4")
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.Label([
                            "X ",
                            html.Span("ℹ", id="tooltip-x", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("X-position offset of galaxy center in pixels", target="tooltip-x", placement="right"),
                        dcc.Slider(id="x-slider", min=-5, max=5, step=0.1, value=0,
                                marks={i: str(i) for i in range(-5, 6)},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label([
                            "Y ",
                            html.Span("ℹ", id="tooltip-y", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("Y-position offset of galaxy center in pixels", target="tooltip-y", placement="right"),
                        dcc.Slider(id="y-slider", min=-5, max=5, step=0.1, value=0,
                                marks={i: str(i) for i in range(-5, 6)},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label([
                            "Ellipticity ",
                            html.Span("ℹ", id="tooltip-ellip", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("Galaxy ellipticity: 0=circular, 1=highly elongated", target="tooltip-ellip", placement="right"),
                        dcc.Slider(id="ellipticity-slider", min=0, max=1, step=0.01, value=0.5,
                                marks={0: 'Circle', 0.5: '0.5', 1: 'Line'},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4")
                ], width=6)
            ], className="mt-3")
        ], fluid=True)
    
    def setup_callbacks(self):
        @self.app.callback(
            Output("main-plot", "figure"),
            [Input("amp-slider", "value"), Input("size-slider", "value"), Input("Sersic-slider", "value"),
             Input("angle-slider", "value"), Input("x-slider", "value"), Input("y-slider", "value"),
             Input("ellipticity-slider", "value")] +
            [Input(f"btn-{key}", "n_clicks") for key in self.data_files.keys()],
            prevent_initial_call=False
        )
        def update_app(*args):
            ctx = callback_context
            self.amplitude, self.radius, self.index = args[0], args[1], args[2]
            self.theta, self.x, self.y, self.ellipticity = args[3], args[4], args[5], args[6]
            
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                for i, key in enumerate(self.data_files.keys()):
                    if trigger_id == f"btn-{key}" and args[7 + i]:
                        self.load_data(key)
                        break
            
            self.generate_model()
            return self.create_main_plot()
    
    def create_main_plot(self):
        vmin, vmax = np.percentile(self.image, 1), np.percentile(self.image, 99.5)
        def asinh_stretch(data, vmin, vmax, a):
            return np.arcsinh((data - vmin) / (vmax - vmin) / a) / np.arcsinh(1 / a)
        
        image_scaled = asinh_stretch(self.image, vmin, vmax, 0.1)
        model_scaled = asinh_stretch(self.model_image, vmin, vmax, 0.1)
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Observed Image", "Your Model", "Difference"), horizontal_spacing=0.1)
        fig.add_trace(go.Heatmap(z=image_scaled, colorscale='Viridis', showscale=True, 
                                  colorbar=dict(x=0.3, len=0.8, title="Brightness")), row=1, col=1)
        fig.add_trace(go.Heatmap(z=model_scaled, colorscale='Viridis', showscale=True,
                                  colorbar=dict(x=0.63, len=0.8, title="Brightness")), row=1, col=2)
        fig.add_trace(go.Heatmap(z=self.residual, colorscale='RdBu_r', showscale=True, zmid=0, zmin=-5, zmax=5,
                                  colorbar=dict(x=1, len=0.8, title="Residual")), row=1, col=3)
        
        fig.update_layout(title=f"JWST Photometry Fit, score= {self.calculate_score():.2f}, lower is better",
                         template="plotly_white", height=600, showlegend=False)
        fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1, constrain="domain")
        fig.update_yaxes(constrain="domain", row=1, col=1)
        fig.update_xaxes(scaleanchor="y2", scaleratio=1, row=1, col=2, constrain="domain")
        fig.update_yaxes(constrain="domain", row=1, col=2)
        fig.update_xaxes(scaleanchor="y3", scaleratio=1, row=1, col=3, constrain="domain")
        fig.update_yaxes(constrain="domain", row=1, col=3)
        return fig