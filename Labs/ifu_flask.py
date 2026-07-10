#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive FITS Data Visualization using Dash
"""

from astropy.io import fits as pyfits
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import os
import sys
nan = float('nan')
pi = np.pi
e = np.e
c = 3.*10**8


class IFU_lab:
    '''Class to visualize FITS data using Dash'''
    def __init__(self, server, requests_pathname_prefix, routes_pathname_prefix, app_name=None):
        
        """Initialize the IFU Dash application with Flask server"""
        self.app = dash.Dash(
            app_name or __name__,          # use unique name if provided
            server=server,
            requests_pathname_prefix=requests_pathname_prefix,
            routes_pathname_prefix=routes_pathname_prefix,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        pth = sys.path[0] if sys.path[0] else '.'
        filepath = os.path.join(pth, 'Data/IFU/COS30_R2700_Halpha_OIII_fits_maps.fits')

        self.z = 6.85072093
        self.vlim2 = [-150,150]
        self.indc = [1,0]
        self.map_hdu_name = ['OIII', 'Narrow_vel']
        
        # Load FITS data lazily via memmap and keep the file open for the
        # lifetime of the app. The flux/error/yeval cubes are ~300MB each
        # (~1.4GB total); most callbacks only touch a single spaxel or a
        # 2-3 plane slice, so we let the OS page in only what's accessed
        # instead of copying the whole cube into the process at startup.
        # This also lets multiple Apache/mod_wsgi worker processes share
        # the same pages via the OS page cache rather than each holding
        # its own full copy.
        self.hdulist = pyfits.open(filepath, memmap=True)
        hdulist = self.hdulist

        self.map = []
        for ind, hdu_name in zip(self.indc, self.map_hdu_name):
            self.map.append(np.array(hdulist[hdu_name].data[ind, :, :]))

        self.yeval = hdulist['yeval'].data
        self.flux = hdulist['flux'].data
        self.error = hdulist['error'].data
        names = [hdu.name for hdu in hdulist]

        if 'YEVAL_NAR' in names:
            self.yeval_nar = hdulist['yeval_nar'].data
        else:
            self.yeval_nar = None

        if 'YEVAL_BRO' in names:
            self.yeval_bro = hdulist['yeval_bro'].data
        else:
            self.yeval_bro = None

        self.header = hdulist['PRIMARY'].header.copy()
        nwave = np.shape(self.yeval)[0]
        self.obs_wave = (self.header['CRVAL3'] +
                       (np.arange(nwave) - (self.header['CRPIX3'] - 1.0)) *
                       self.header['CDELT3'])
        
        # Store data shape first
        self.nwave = len(self.obs_wave)
        self.ny, self.nx = self.flux.shape[1], self.flux.shape[2]
        
        # Initialize with safe default values based on actual data size
        self.slice_val_ind = min(2, self.nwave - 1)
        self.slice_val = self.obs_wave[self.slice_val_ind]
        self.current_i = self.nx // 2
        self.current_j = self.ny // 2
        
        self.showme()

    
    def create_map_figure(self, map_name, title):
        """Create a heatmap figure for a given map"""
       
        if map_name ==1:  # Velocity map
            colorscale = 'RdBu_r'
            zmin, zmax = self.vlim2[0], self.vlim2[1]
        else:
            colorscale = 'Viridis'
            zmin, zmax = None, None
        
        fig = go.Figure(data=go.Heatmap(
            z=self.map[map_name].tolist(),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(thickness=20, len=0.7)
        ))

        
        fig.update_layout(
            title=title,
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_slice_figure(self, slice_idx):
        """Create slice map figure"""
        slice_data = np.sum(
            self.flux[max(0, slice_idx-1):min(self.nwave, slice_idx+2), :, :],
            axis=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=slice_data,
            colorscale='Viridis',
            colorbar=dict(thickness=20, len=0.7)
        ))
        
        fig.update_layout(
            title=f'Wavelength Slice: {self.obs_wave[slice_idx]:.4f}',
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_spectrum_figure(self, i, j, xlims, slice_idx):
        """Create spectrum plot for given spaxel"""
        fluxm = self.flux[:, j, i].tolist()
        errorm = self.error[:, j, i].tolist()
        yevalm = self.yeval[:, j, i].tolist()

        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(self.obs_wave, fluxm, drawstyle='steps-mid', color='blue', label='Observed')
        #plt.show()
        fig = go.Figure()
        
        # Flux
        fig.add_trace(go.Scatter(
            x=self.obs_wave.tolist(),
            y=fluxm,
            mode='lines',
            name='Observed',
            line=dict(color='blue', shape='hv')
        ))
        
        # Model
        fig.add_trace(go.Scatter(
            x=self.obs_wave.tolist(),
            y=yevalm,
            mode='lines',
            name='Model',
            line=dict(color='red', dash='dash')
        ))
        
        # Error
        fig.add_trace(go.Scatter(
            x=self.obs_wave.tolist(),
            y=errorm,
            mode='lines',
            name='Error',
            line=dict(color='black', dash='dot')
        ))
        
        # Narrow component if exists
        if self.yeval_nar is not None:
            fig.add_trace(go.Scatter(
                x=self.obs_wave,
                y=self.yeval_nar[:, j, i].tolist(),
                mode='lines',
                name='Narrow',
                line=dict(color='green', dash='dash')
            ))
        
        # Broad component if exists
        if self.yeval_bro is not None:
            fig.add_trace(go.Scatter(
                x=self.obs_wave,
                y=self.yeval_bro[:, j, i].tolist(),
                mode='lines',
                name='Broad',
                line=dict(color='purple', dash='dash')
            ))
        
        # Add vertical line for slice position
        slice_wave = self.obs_wave[slice_idx]
        fig.add_vline(
            x=slice_wave,
            line_dash="dash",
            line_color="black",
            line_width=2
        )
        
        fig.update_layout(
            title=f'Spectrum at (x={i}, y={j})',
            xaxis_title="Wavelength",
            yaxis_title="Flux",
            xaxis_range=xlims,
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def showme(self, xlims=None, vmax=1e-15, ylims=None):
        """
        Create interactive Dash application
        
        Parameters
        ----------
        xlims : tuple, optional
            X-axis wavelength limits
        vmax : float, optional
            Maximum value for color scale
        ylims : tuple, optional
            Y-axis flux limits
        """
        self.xlims = xlims if xlims else [self.obs_wave[0], self.obs_wave[-1]]
        self.ylims = ylims
                
        # Create initial figures
        map0_fig = self.create_map_figure(0, self.map_hdu_name[0])
        map1_fig = self.create_map_figure(1, self.map_hdu_name[1])
        slice_fig = self.create_slice_figure(self.slice_val_ind)
        spectrum_fig = self.create_spectrum_figure(
            self.current_i, self.current_j, self.xlims, self.slice_val_ind
        )
        
        # Layout
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Interactive FITS Cube Visualization", 
                               className="text-center mb-4"), width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='map0', figure=map0_fig, 
                             style={'height': '400px'})
                ], width=4),
                dbc.Col([
                    dcc.Graph(id='map1', figure=map1_fig,
                             style={'height': '400px'})
                ], width=4),
                dbc.Col([
                    dcc.Graph(id='slice-map', figure=slice_fig,
                             style={'height': '400px'})
                ], width=4),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Wavelength Slice:", className="fw-bold"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("−", id='decrease-slice', color="primary", 
                                      className="me-2", size="sm")
                        ], width="auto"),
                        dbc.Col([
                            dcc.Slider(
                                id='wavelength-slider',
                                min=0,
                                max=self.nwave - 1,
                                step=1,
                                value=self.slice_val_ind,
                                marks={i: f'{self.obs_wave[i]:.3f}' 
                                       for i in range(0, self.nwave, max(1, self.nwave//10))},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=True),
                        dbc.Col([
                            dbc.Button("+", id='increase-slice', color="primary", 
                                      className="ms-2", size="sm")
                        ], width="auto")
                    ], align="center")
                ], width=12, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("X-axis Range (Wavelength):", className="fw-bold"),
                    dcc.RangeSlider(
                        id='xlim-slider',
                        min=0,
                        max=self.nwave - 1,
                        value=[0, self.nwave - 1],
                        marks={i: f'{self.obs_wave[i]:.3f}' 
                               for i in range(0, self.nwave, max(1, self.nwave//10))},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=12, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='spectrum', figure=spectrum_fig,
                             style={'height': '500px'})
                ], width=12)
            ]),
            
            # Hidden div to store current position
            html.Div(id='current-position', 
                    style={'display': 'none'},
                    children=f'{self.current_i},{self.current_j}')
            
        ], fluid=True)
        
        # Callback for button clicks to update slider
        @self.app.callback(
            Output('wavelength-slider', 'value'),
            [Input('decrease-slice', 'n_clicks'),
             Input('increase-slice', 'n_clicks')],
            [State('wavelength-slider', 'value')]
        )
        def update_slider_from_buttons(n_decrease, n_increase, current_value):
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return current_value
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'decrease-slice' and n_decrease:
                new_value = max(0, current_value - 1)
            elif button_id == 'increase-slice' and n_increase:
                new_value = min(self.nwave - 1, current_value + 1)
            else:
                new_value = current_value
            
            return new_value
        
        # Callback for map clicks and spectrum update
        @self.app.callback(
            [Output('spectrum', 'figure'),
             Output('current-position', 'children')],
            [Input('map0', 'clickData'),
             Input('map1', 'clickData'),
             Input('slice-map', 'clickData'),
             Input('wavelength-slider', 'value'),
             Input('xlim-slider', 'value')],
            [State('current-position', 'children')]
        )
        def update_spectrum(click0, click1, click2, slice_idx, xlim_range, position):
            ctx = dash.callback_context
            
            # Get current position
            i, j = map(int, position.split(','))
            
            # Check which input triggered the callback
            if ctx.triggered:
                prop_id = ctx.triggered[0]['prop_id']
                
                # Handle map clicks
                if 'clickData' in prop_id and ctx.triggered[0]['value'] is not None:
                    click_data = ctx.triggered[0]['value']
                    if click_data and 'points' in click_data:
                        point = click_data['points'][0]
                        i = int(round(point['x']))
                        j = int(round(point['y']))
                        # Ensure within bounds
                        i = max(0, min(self.nx - 1, i))
                        j = max(0, min(self.ny - 1, j))
            
            # Get xlims from slider
            xlims = [self.obs_wave[xlim_range[0]], self.obs_wave[xlim_range[1]]]
            
            # Create updated spectrum figure
            fig = self.create_spectrum_figure(i, j, xlims, slice_idx)
            
            return fig, f'{i},{j}'
        
        # Callback for slice map update
        @self.app.callback(
            Output('slice-map', 'figure'),
            Input('wavelength-slider', 'value')
        )
        def update_slice(slice_idx):
            return self.create_slice_figure(slice_idx)