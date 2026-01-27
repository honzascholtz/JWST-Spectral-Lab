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

# Try to import bagpipes
try:
    import bagpipes as pipes
    pipes.config.max_redshift = 17
    BAGPIPES_AVAILABLE = True
except ImportError:
    BAGPIPES_AVAILABLE = False
    print("Warning: bagpipes not available. Using mock model generation.")

nan = float('nan')
pi = np.pi
e = np.e
c = 3.*10**8

# ============================================================================
# APP 3: STELLAR POPULATION LAB
# ============================================================================
class Stellar_pop_lab:
    def __init__(self, server, requests_pathname_prefix, routes_pathname_prefix):
        self.app = dash.Dash(__name__, server=server, requests_pathname_prefix=requests_pathname_prefix, routes_pathname_prefix=routes_pathname_prefix,

                            external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "JADES Stellar Population Lab"
        
        self.data_wave = None
        self.data_flux = None
        self.data_error = None
        self.model = None
        self.model_spectrum = None
        
        self.z = 9.431
        self.Mass = 9.0
        self.age = 0.3
        self.tau = 0.3
        self.Z = 1.0
        self.U = -3
        self.Av = 0.5
        self.target = 'generic'
        
        self.data_files = {
            'SF943': {'file': '10058975_prism_clear_v5.0_1D.fits', 'z': 9.436, 'target': 'generic'},
            'QC_galaxy': {'file': '199773_prism_clear_v5.0_1D.fits', 'z': 2.820, 'target': 'generic'},
            'SF_galaxy': {'file': '001882_prism_clear_v5.0_1D.fits', 'z': 5.4431, 'target': 'generic'},
            'GSz14': {'file': '183348_prism_clear_v5.0_1D.fits', 'z': 14.18, 'target': 'GSz14'},
            'COS30': {'file': '007437_prism_clear_v3.1_1D.fits', 'z': 6.856, 'target': 'generic'},
            'SF2': {'file': '001927_prism_clear_v5.0_1D.fits', 'z': 3.6591, 'target': 'generic'},
            'PSB': {'file': '023286_prism_clear_v5.1_1D.fits', 'z': 1.781, 'target': 'generic'},
            'zhig': {'file': '066585_prism_clear_v5.1_1D.fits', 'z': 7.1404, 'target': 'low_snr'},
        }
        
        self.load_data('SF943')
        self.pregenerate_model()
        self.generate_model()
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self, dataset_key):
        config = self.data_files[dataset_key]
        self.z = config['z']
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
    
    def create_mock_model_spectrum(self):
        if self.data_wave is None:
            return
        age_factor = max(0.1, self.age / 2.0)
        mass_factor = 10 ** (self.Mass - 9.0)
        metal_factor = self.Z
        dust_factor = np.exp(-self.Av / 3.0)
        continuum = age_factor * mass_factor * metal_factor * dust_factor * 0.005 * (self.data_wave / 2.0) ** (-1.5)
        emission = np.zeros_like(self.data_wave)
        if self.U > -4:
            ionization_factor = 10 ** (self.U + 3)
            emission = ionization_factor * (np.exp(-((self.data_wave - 1.2) / 0.05) ** 2) * 0.01 +
                                           np.exp(-((self.data_wave - 2.1) / 0.03) ** 2) * 0.005 +
                                           np.exp(-((self.data_wave - 3.7) / 0.04) ** 2) * 0.008)
        model_flux = continuum + emission
        self.model_spectrum = np.column_stack([self.data_wave * 1e4, model_flux])
    
    def pregenerate_model(self):
        global BAGPIPES_AVAILABLE
        if BAGPIPES_AVAILABLE:
            try:
                delayed = {"age": self.age, "tau": self.tau, "massformed": self.Mass, "metallicity": self.Z}
                dust = {"type": "Calzetti", "Av": self.Av}
                model_components = {"redshift": self.z, "delayed": delayed, "dust": dust}
                if self.U > -4:
                    model_components["nebular"] = {"logU": self.U}
                
                try:
                    pth = sys.path[0] if sys.path[0] else '.'
                    with pyfits.open(os.path.join(pth, "Data", "jwst_nirspec_prism_disp.fits")) as hdul:
                        model_components["R_curve"] = np.c_[1e4 * hdul[1].data["WAVELENGTH"], hdul[1].data["R"]]
                except:
                    print("Warning: Could not load resolution curve")
                
                self.model = pipes.model_galaxy(model_components, 
                    spec_wavs=self.data_wave * 1e4 if self.data_wave is not None else np.linspace(5000, 53000, 1000))
            except Exception as e:
                print(f"Error creating bagpipes model: {e}")
                BAGPIPES_AVAILABLE = False
                self.create_mock_model_spectrum()
        else:
            self.create_mock_model_spectrum()
    
    def generate_model(self):
        global BAGPIPES_AVAILABLE
        if BAGPIPES_AVAILABLE and self.model is not None:
            try:
                delayed = {"age": self.age, "tau": self.tau, "massformed": self.Mass, "metallicity": self.Z}
                dust = {"type": "Calzetti", "Av": self.Av}
                model_components = {"redshift": self.z, "delayed": delayed, "dust": dust}
                if self.U > -5:
                    model_components["nebular"] = {"logU": self.U}
                
                try:
                    pth = sys.path[0] if sys.path[0] else '.'
                    with pyfits.open(os.path.join(pth, "Data", "jwst_nirspec_prism_disp.fits")) as hdul:
                        model_components["R_curve"] = np.c_[1e4 * hdul[1].data["WAVELENGTH"], hdul[1].data["R"]]
                except:
                    pass
                
                self.model.update(model_components)
                self.model_spectrum = self.model.spectrum
            except Exception as e:
                print(f"Error updating bagpipes model: {e}")
                self.create_mock_model_spectrum()
        else:
            self.create_mock_model_spectrum()
    
    def calculate_score(self):
        if self.model_spectrum is not None and self.data_flux is not None:
            try:
                model_flux = self.model_spectrum[:, 1]
                if len(model_flux) == len(self.data_flux) and np.nansum(model_flux) > 0:
                    return np.nansum((self.data_flux - model_flux)**2 / self.data_error**2) / (len(self.data_flux) - 6)
            except:
                pass
        return 0.0
    
    def get_emission_lines(self):
        emlines = {
            r'C⁺⁺': (1907., 'red'), r'Mg⁺': (2797., 'blue'), r'[O⁺]': (3728., 'green'),
            r'[Ne⁺⁺]': (3869.860, 'purple'), 'Hδ': (4102.860, 'orange'), 'Hγ': (4341.647, 'pink'),
            'Hβ': (4862.647, 'brown'), r'[O⁺⁺]': (4960.0, 'red'), r'[O⁰]': (6302.0, 'green'),
            'Na': (5891.583, 'yellow'), 'Hα': (6564.522, 'red'), r'[S⁺]': (6725, 'blue'),
            r'[S⁺⁺]': (9070.0, 'purple'), 'HeI': (10832.1, 'orange'), 'Paγ': (10940.978, 'pink'),
            r'[Fe⁺]': (12570.200, 'brown'), 'Paβ': (12821.432, 'green')
        }
        visible_lines = {}
        for line_name, (rest_wave, color) in emlines.items():
            obs_wave = rest_wave * (1 + self.z) / 1.e4
            if 0.5 * 1.001 < obs_wave < 5.3 * 0.999:
                visible_lines[line_name] = (obs_wave, color)
        return visible_lines
    
    def calculate_sfh(self):
        sfh = self.model.sfh
        age_universe = (sfh.age_of_universe - sfh.ages) * 10**-9
        sfr = sfh.sfh
        return age_universe, sfr
    
    def get_instantaneous_sfr(self):
        """Get the current star formation rate of the galaxy"""
        age_universe, sfr = self.calculate_sfh()
        # Get the most recent SFR (at the observation time)
        return sfr[0]
            
    
    def get_main_sequence(self, z):
        """
        Calculate the star-forming main sequence based on redshift
        Using Speagle et al. (2014) relation
        """
        # Calculate cosmic time at given redshift
        t = cosmo.age(z).value  # in Gyr
        
        # Mass range for plotting
        log_mass = np.linspace(7, 12, 100)
        
        # Speagle et al. (2014) main sequence relation
        # log(SFR) = (0.84 - 0.026*t) * log(M*) - (6.51 - 0.11*t)
        log_sfr_ms = (0.84 - 0.026*t) * (log_mass - 9) + (0.84 - 0.026*t) * 9 - (6.51 - 0.11*t)
        
        # Scatter in main sequence (±0.3 dex)
        log_sfr_upper = log_sfr_ms + 0.3
        log_sfr_lower = log_sfr_ms - 0.3
        
        return log_mass, log_sfr_ms, log_sfr_upper, log_sfr_lower
    
    def setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([dbc.Col([html.H1("JADES Stellar Population Lab", className="text-center mb-4")], width=12)]),
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([dbc.Button(key, id=f"btn-{key}", color="info", size="sm") 
                                    for key in self.data_files.keys()], className="mb-3")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(id="main-plot", style={'height': '600px'})], width=9),
                dbc.Col([dcc.Graph(id="sfh-plot", style={'height': '600px'})], width=3)
            ], style={'margin-bottom': '20px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label([
                            "Mass [log(M☉)] ",
                            html.Span("ℹ", id="tooltip-mass", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("Total stellar mass of the galaxy in solar masses (logarithmic scale)", target="tooltip-mass", placement="right"),
                        dcc.Slider(id="mass-slider", min=7, max=12, step=0.1, value=8,
                                marks={i: str(i) for i in range(7, 13)},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label([
                            "Radiation Strength [log U] ",
                            html.Span("ℹ", id="tooltip-logU", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("Ionization parameter: ratio of ionizing photon density to gas density", target="tooltip-logU", placement="right"),
                        dcc.Slider(id="logU-slider", min=-4.01, max=-1, step=0.1, value=-2.5,
                                marks={-4: '-4', -3: '-3', -2: '-2', -1: '-1'},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label([
                            "Heavy Elements [Z/Z☉] ",
                            html.Span("ℹ", id="tooltip-metal", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("Metallicity: abundance of elements heavier than helium relative to the Sun", target="tooltip-metal", placement="right"),
                        dcc.Slider(id="metal-slider", min=0.01, max=1.4, step=0.05, value=0.5,
                                marks={0: '0', 0.5: '0.5', 1: '1', 1.4: '1.4'},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4")
                ], width=6),
                
                dbc.Col([
                    html.Div([
                        html.Label([
                            "Age of Stars [log Gyr] ",
                            html.Span("ℹ", id="tooltip-age", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("Time since the onset of star formation in billions of years", target="tooltip-age", placement="right"),
                        dcc.Slider(id="age-slider", min=-2, max=1, step=0.1, value=-1,
                                marks={-2: '0.01', -1: '0.1', 0: '1', 1: '10'},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label([
                            "Decline of Stars [log Gyr] ",
                            html.Span("ℹ", id="tooltip-tau", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("e-folding timescale: how quickly star formation declines (tau parameter)", target="tooltip-tau", placement="right"),
                        dcc.Slider(id="tau-slider", min=-2, max=1, step=0.1, value=-1.4,
                                marks={-2: '0.01', -1: '0.1', 0: '1', 1: '10'},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label([
                            "Dust [Av mag] ",
                            html.Span("ℹ", id="tooltip-dust", style={"color": "#17a2b8", "cursor": "pointer", "fontSize": "16px"})
                        ], className="fw-bold mb-2"),
                        dbc.Tooltip("Visual extinction: amount of light absorbed by interstellar dust", target="tooltip-dust", placement="right"),
                        dcc.Slider(id="dust-slider", min=0, max=3, step=0.1, value=0.5,
                                marks={0: '0', 1: '1', 2: '2', 3: '3'},
                                tooltip={"placement": "bottom", "always_visible": True})
                    ], className="mb-4")
                ], width=6)
            ], className="mt-3"),
            
            # New row for SFR vs Mass plot
            dbc.Row([
                dbc.Col([
                    html.Hr(className="my-4"),
                    dcc.Graph(id="sfr-mass-plot", style={'height': '500px'})
                ], width=12)
            ], className="mt-3")
        ], fluid=True)
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output("main-plot", "figure"), Output("sfh-plot", "figure"), Output("sfr-mass-plot", "figure")],
            [Input("mass-slider", "value"), Input("logU-slider", "value"), Input("metal-slider", "value"),
             Input("age-slider", "value"), Input("tau-slider", "value"), Input("dust-slider", "value")] +
            [Input(f"btn-{key}", "n_clicks") for key in self.data_files.keys()],
            prevent_initial_call=False
        )
        def update_app(*args):
            ctx = callback_context
            self.Mass, self.U, self.Z = args[0], args[1], args[2]
            self.age, self.tau, self.Av = 10 ** args[3], 10 ** args[4], args[5]
            
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                for i, key in enumerate(self.data_files.keys()):
                    if trigger_id == f"btn-{key}" and args[6 + i]:
                        self.load_data(key)
                        self.pregenerate_model()
                        break
            
            self.generate_model()
            return self.create_main_plot(), self.create_sfh_plot(), self.create_sfr_mass_plot()
    
    def create_main_plot(self):
        if self.data_wave is None:
            return go.Figure()
        
        fig = go.Figure()
        model_valid = (self.model_spectrum is not None and len(self.model_spectrum) > 0 and 
                      np.nansum(self.model_spectrum[:, 1]) > 0)
        
        if not model_valid:
            fig.add_annotation(text="Your stars are older than the Universe!<br>Reduce Age parameter!",
                             x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                             font=dict(size=16, color="red"), bgcolor="yellow", bordercolor="red")
        else:
            if self.model_spectrum is not None:
                fig.add_trace(go.Scatter(x=self.model_spectrum[:, 0] / 1e4, y=self.model_spectrum[:, 1] / 1e-18,
                                        mode='lines', line=dict(color='firebrick', shape='hv'), 
                                        name='Model', showlegend=True))
        
        fig.add_trace(go.Scatter(x=self.data_wave, y=self.data_flux / 1e-18, mode='lines',
                                line=dict(color='black', shape='hv'), name='Observations', showlegend=True))
        
        if model_valid:
            emission_lines = self.get_emission_lines()
            for line_name, (obs_wave, color) in emission_lines.items():
                fig.add_vline(x=obs_wave, line=dict(color=color, dash='dash', width=1.5), opacity=0.5)
                fig.add_annotation(x=obs_wave, y=0.95, text=line_name, textangle=90, showarrow=False,
                                 yref='paper', bgcolor='white', bordercolor=color, 
                                 font=dict(size=16, color=color), borderwidth=1)
        
        fig.update_layout(title=f"JWST Spectrum | Score: {self.calculate_score():.2f}",
                         xaxis_title="Wavelength [μm]", yaxis_title="Flux [10⁻¹⁸ erg/s/cm²/Å]",
                         xaxis=dict(range=[0.5, 5.5]), template="plotly_white", height=600,
                         legend=dict(x=0.02, y=0.98), title_font_size=16)
        
        if self.target == 'GSz14':
            fig.update_yaxes(range=[-0.00025, 0.01])
        elif self.target == 'gnz11':
            fig.update_yaxes(range=[-0.01, 0.04])
        elif self.target == 'low_snr':
            fig.update_yaxes(range=[-0.01, 0.025])
        
        return fig
    
    def create_sfh_plot(self):
        fig = go.Figure()
        try:
            age_universe, sfr = self.calculate_sfh()
            fig.add_trace(go.Scatter(x=age_universe, y=sfr, mode='lines', line=dict(color='steelblue', width=2),
                                    fill='tozeroy', fillcolor='rgba(70, 130, 180, 0.3)', 
                                    name='SFR', showlegend=False))
            
            age_now = cosmo.age(self.z).value
            fig.add_vline(x=0, line=dict(color='red', dash='dash', width=2),
                         annotation_text="Big Bang", annotation_position="top left")
            fig.add_vline(x=age_now, line=dict(color='red', dash='dash', width=2),
                         annotation_text="Galaxy Now", annotation_position="top right")
            
            fig.update_layout(title="Star Formation History", xaxis_title="Age of Universe [Gyr]",
                            yaxis_title="SFR [M☉/yr]", template="plotly_white", height=600,
                            title_font_size=14, margin=dict(l=60, r=20, t=60, b=60))
            fig.update_yaxes(rangemode='tozero')
            fig.update_xaxes(range=[cosmo.age(self.z).value, 0])
        except Exception as e:
            print(f"Error creating SFH plot: {e}")
            fig.add_annotation(text="Error calculating SFH", x=0.5, y=0.5, xref="paper", yref="paper",
                             showarrow=False, font=dict(size=14, color="red"))
        return fig
    
    def create_sfr_mass_plot(self):
        """Create the SFR vs Mass plot with main sequence"""
        fig = go.Figure()
        
        try:
            # Get main sequence relation
            log_mass, log_sfr_ms, log_sfr_upper, log_sfr_lower = self.get_main_sequence(self.z)
            
            # Plot main sequence as a shaded region
            fig.add_trace(go.Scatter(
                x=log_mass, y=10**log_sfr_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=log_mass, y=10**log_sfr_lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(200, 200, 200, 0.3)',
                name='Main Sequence ±0.3 dex',
                showlegend=True
            ))
            
            # Plot main sequence line
            fig.add_trace(go.Scatter(
                x=log_mass, y=10**log_sfr_ms,
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                name=f'Main Sequence (z={self.z:.2f})',
                showlegend=True
            ))
            
            # Calculate and plot current galaxy position
            current_sfr = self.get_instantaneous_sfr()
            log_sfr = np.log10(current_sfr) if current_sfr > 0 else -5
            # Determine if galaxy is on, above, or below main sequence
            log_mass_current = self.Mass
            log_sfr_ms_current = (0.84 - 0.026*cosmo.age(self.z).value) * (log_mass_current - 9) + \
                                 (0.84 - 0.026*cosmo.age(self.z).value) * 9 - \
                                 (6.51 - 0.11*cosmo.age(self.z).value)
            
            offset = log_sfr - log_sfr_ms_current
            
            if offset > 0.3:
                galaxy_type = "Starburst"
                marker_color = "red"
            elif offset < -0.3:
                galaxy_type = "Quiescent"
                marker_color = "blue"
            else:
                galaxy_type = "Main Sequence"
                marker_color = "green"
            
            # Plot the current galaxy
            fig.add_trace(go.Scatter(
                x=[self.Mass],
                y=[current_sfr],
                mode='markers',
                marker=dict(
                    size=15,
                    color=marker_color,
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                name=f'Your Galaxy ({galaxy_type})',
                showlegend=True,
                text=f'Mass: {self.Mass:.2f}<br>SFR: {current_sfr:.2f} M☉/yr<br>Offset: {offset:.2f} dex',
                hoverinfo='text'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Star Formation Rate vs Stellar Mass (z={self.z:.2f})",
                xaxis_title="log(M* / M☉)",
                yaxis_title="SFR [M☉/yr]",
                xaxis=dict(range=[7, 12]),
                yaxis_type="log",
                yaxis=dict(range=[-2, 3]),
                template="plotly_white",
                height=500,
                title_font_size=16,
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
                hovermode='closest'
            )
            
        except Exception as e:
            print(f"Error creating SFR-Mass plot: {e}")
            fig.add_annotation(
                text=f"Error creating plot: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14, color="red")
            )
        
        return fig