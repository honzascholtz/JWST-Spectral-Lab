from flask import Flask
import sys
import os
import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from astropy.cosmology import Planck18 as cosmo
import astropy.io.fits as pyfits
import traceback

try:
    import bagpipes as pipes
    pipes.config.max_redshift = 17
    BAGPIPES_AVAILABLE = True
except ImportError:
    BAGPIPES_AVAILABLE = False

# ============================================================================
# Module-level read-only data cache (safe to share — never mutated after load)
# ============================================================================
_DATA_CACHE = {}

DATA_FILES = {
    'SF943':     {'file': '10058975_prism_clear_v5.0_1D.fits', 'z': 9.436,  'target': 'generic'},
    'QC_galaxy': {'file': '199773_prism_clear_v5.0_1D.fits',   'z': 2.820,  'target': 'generic'},
    'SF_galaxy': {'file': '001882_prism_clear_v5.0_1D.fits',   'z': 5.4431, 'target': 'generic'},
    'GSz14':     {'file': '183348_prism_clear_v5.0_1D.fits',   'z': 14.18,  'target': 'GSz14'},
    'COS30':     {'file': '007437_prism_clear_v3.1_1D.fits',   'z': 6.856,  'target': 'generic'},
    'SF2':       {'file': '001927_prism_clear_v5.0_1D.fits',   'z': 3.6591, 'target': 'generic'},
    'PSB':       {'file': '023286_prism_clear_v5.1_1D.fits',   'z': 1.781,  'target': 'generic'},
    'zhig':      {'file': '066585_prism_clear_v5.1_1D.fits',   'z': 7.1404, 'target': 'low_snr'},
}

def _load_data(dataset_key):
    """Load spectrum data; results are cached since files never change."""
    if dataset_key in _DATA_CACHE:
        return _DATA_CACHE[dataset_key]

    config = DATA_FILES[dataset_key]
    pth = sys.path[0] if sys.path[0] else '.'
    filepath = os.path.join(pth, 'Data', config['file'])
    with pyfits.open(filepath) as hdu:
        wave  = hdu['WAVELENGTH'].data * 1e6
        flux  = hdu['DATA'].data * 1e-7
        error = hdu['ERR'].data * 1e-7

    result = dict(wave=wave, flux=flux, error=error,
                  z=config['z'], target=config['target'])
    _DATA_CACHE[dataset_key] = result
    return result


def _load_r_curve():
    """Load the NIRSpec resolution curve (cached)."""
    if 'r_curve' in _DATA_CACHE:
        return _DATA_CACHE['r_curve']
    pth = sys.path[0] if sys.path[0] else '.'
    with pyfits.open(os.path.join(pth, 'Data', 'jwst_nirspec_prism_disp.fits')) as hdul:
        r_curve = np.c_[1e4 * hdul[1].data['WAVELENGTH'], hdul[1].data['R']]
    _DATA_CACHE['r_curve'] = r_curve
    return r_curve


# ============================================================================
# Pure computation helpers (no class state)
# ============================================================================


def _get_emission_lines(z):
    emlines = {
        r'C⁺⁺': (1907., 'red'),       r'Mg⁺': (2797., 'blue'),
        r'[O⁺]': (3728., 'green'),     r'[Ne⁺⁺]': (3869.860, 'purple'),
        'Hδ': (4102.860, 'orange'),    'Hγ': (4341.647, 'pink'),
        'Hβ': (4862.647, 'brown'),     r'[O⁺⁺]': (4960.0, 'red'),
        r'[O⁰]': (6302.0, 'green'),   'Na': (5891.583, 'yellow'),
        'Hα': (6564.522, 'red'),       r'[S⁺]': (6725, 'blue'),
        r'[S⁺⁺]': (9070.0, 'purple'), 'HeI': (10832.1, 'orange'),
        'Paγ': (10940.978, 'pink'),    r'[Fe⁺]': (12570.200, 'brown'),
        'Paβ': (12821.432, 'green'),
    }
    return {
        name: (rest * (1 + z) / 1e4, color)
        for name, (rest, color) in emlines.items()
        if 0.5 * 1.001 < rest * (1 + z) / 1e4 < 5.3 * 0.999
    }


def _calculate_score(data_flux, data_error, model_spectrum):
    if model_spectrum is None:
        return 0.0
    model_flux = model_spectrum[:, 1]
    if len(model_flux) != len(data_flux) or np.nansum(model_flux) == 0:
        return 0.0
    return np.nansum((data_flux - model_flux)**2 / data_error**2) / (len(data_flux) - 6)


def _get_main_sequence(z):
    t = cosmo.age(z).value
    log_mass = np.linspace(7, 12, 100)
    log_sfr_ms    = (0.84 - 0.026*t) * (log_mass - 9) + (0.84 - 0.026*t) * 9 - (6.51 - 0.11*t)
    return log_mass, log_sfr_ms, t


# ============================================================================
# Plot builders (pure functions — take data, return figures)
# ============================================================================

def _build_main_plot(data, model_spectrum, z, target):
    fig = go.Figure()
    model_valid = (model_spectrum is not None and
                   len(model_spectrum) > 0 and
                   np.nansum(model_spectrum[:, 1]) > 0)

    if not model_valid:
        fig.add_annotation(
            text="Your stars are older than the Universe!<br>Reduce Age parameter!",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(size=16, color="red"), bgcolor="yellow", bordercolor="red")
    else:
        fig.add_trace(go.Scatter(
            x=model_spectrum[:, 0] / 1e4, y=model_spectrum[:, 1] / 1e-18,
            mode='lines', line=dict(color='firebrick', shape='hv'),
            name='Model', showlegend=True))

    fig.add_trace(go.Scatter(
        x=data['wave'], y=data['flux'] / 1e-18,
        mode='lines', line=dict(color='black', shape='hv'),
        name='Observations', showlegend=True))

    if model_valid:
        for line_name, (obs_wave, color) in _get_emission_lines(z).items():
            fig.add_vline(x=obs_wave, line=dict(color=color, dash='dash', width=1.5), opacity=0.5)
            fig.add_annotation(x=obs_wave, y=0.95, text=line_name, textangle=90,
                               showarrow=False, yref='paper', bgcolor='white',
                               bordercolor=color, font=dict(size=16, color=color), borderwidth=1)

    score = _calculate_score(data['flux'], data['error'], model_spectrum)
    fig.update_layout(
        title=f"JWST Spectrum | Score: {score:.2f}",
        xaxis_title="Wavelength [μm]", yaxis_title="Flux [10⁻¹⁸ erg/s/cm²/Å]",
        xaxis=dict(range=[0.5, 5.5]), template="plotly_white", height=600,
        legend=dict(x=0.02, y=0.98), title_font_size=16)

    if target == 'GSz14':
        fig.update_yaxes(range=[-0.00025, 0.01])
    elif target == 'gnz11':
        fig.update_yaxes(range=[-0.01, 0.04])
    elif target == 'low_snr':
        fig.update_yaxes(range=[-0.01, 0.025])
    return fig


def _build_sfh_plot(sfh_obj, z):
    fig = go.Figure()
    try:
        age_universe = (sfh_obj.age_of_universe - sfh_obj.ages) * 1e-9
        sfr = sfh_obj.sfh
        fig.add_trace(go.Scatter(
            x=age_universe, y=sfr, mode='lines',
            line=dict(color='steelblue', width=2),
            fill='tozeroy', fillcolor='rgba(70,130,180,0.3)',
            name='SFR', showlegend=False))
        age_now = cosmo.age(z).value
        fig.add_vline(x=0,       line=dict(color='red', dash='dash', width=2),
                      annotation_text="Big Bang",   annotation_position="top left")
        fig.add_vline(x=age_now, line=dict(color='red', dash='dash', width=2),
                      annotation_text="Galaxy Now", annotation_position="top right")
        fig.update_layout(
            title="Star Formation History",
            xaxis_title="Age of Universe [Gyr]", yaxis_title="SFR [M☉/yr]",
            template="plotly_white", height=600, title_font_size=14,
            margin=dict(l=60, r=20, t=60, b=60))
        fig.update_yaxes(rangemode='tozero')
        fig.update_xaxes(range=[age_now, 0])
    except Exception:
        traceback.print_exc()
        fig.add_annotation(text="Error calculating SFH", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=14, color="red"))
    return fig


def _build_sfr_mass_plot(sfh_obj, Mass, z):
    fig = go.Figure()
    try:
        log_mass, log_sfr_ms, t = _get_main_sequence(z)
        log_sfr_upper = log_sfr_ms + 0.3
        log_sfr_lower = log_sfr_ms - 0.3

        fig.add_trace(go.Scatter(x=log_mass, y=10**log_sfr_upper,
                                 mode='lines', line=dict(width=0),
                                 showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=log_mass, y=10**log_sfr_lower,
                                 mode='lines', line=dict(width=0),
                                 fill='tonexty', fillcolor='rgba(200,200,200,0.3)',
                                 name='Main Sequence ±0.3 dex', showlegend=True))
        fig.add_trace(go.Scatter(x=log_mass, y=10**log_sfr_ms,
                                 mode='lines', line=dict(color='gray', width=2, dash='dash'),
                                 name=f'Main Sequence (z={z:.2f})', showlegend=True))

        current_sfr = sfh_obj.sfh[0]
        log_sfr      = np.log10(current_sfr) if current_sfr > 0 else -5
        log_sfr_ms_c = (0.84 - 0.026*t) * (Mass - 9) + (0.84 - 0.026*t) * 9 - (6.51 - 0.11*t)
        offset       = log_sfr - log_sfr_ms_c

        galaxy_type, marker_color = (
            ("Starburst",     "blue")  if offset >  0.3 else
            ("Quiescent",     "red")   if offset < -0.3 else
            ("Main Sequence", "green")
        )
        fig.add_trace(go.Scatter(
            x=[Mass], y=[current_sfr], mode='markers',
            marker=dict(size=15, color=marker_color, symbol='star',
                        line=dict(color='black', width=2)),
            name=f'Your Galaxy ({galaxy_type})', showlegend=True,
            text=f'Mass: {Mass:.2f}<br>SFR: {current_sfr:.2f} M☉/yr<br>Offset: {offset:.2f} dex',
            hoverinfo='text'))

        fig.update_layout(
            title=f"Star Formation Rate vs Stellar Mass (z={z:.2f})",
            xaxis_title="log(M* / M☉)", yaxis_title="SFR [M☉/yr]",
            xaxis=dict(range=[7, 12]),
            yaxis_type="log", yaxis=dict(range=[-2, 3]),
            template="plotly_white", height=500, title_font_size=16,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='closest')
    except Exception:
        traceback.print_exc()
        fig.add_annotation(text="Error creating plot", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=14, color="red"))
    return fig


# ============================================================================
# APP CLASS — only holds immutable config + Dash wiring
# ============================================================================
class Stellar_pop_lab:
    def __init__(self, server, requests_pathname_prefix,
                 routes_pathname_prefix, app_name=None):
        self.app = dash.Dash(
            app_name or __name__,
            server=server,
            requests_pathname_prefix=requests_pathname_prefix,
            routes_pathname_prefix=routes_pathname_prefix,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )
        self.app.title = "JADES Stellar Population Lab"

        # Pre-load all datasets into the module-level cache at startup
        # so the first user doesn't pay the I/O cost
        for key in DATA_FILES:
            try:
                _load_data(key)
            except Exception as e:
                print(f"Warning: could not pre-cache {key}: {e}")
        try:
            _load_r_curve()
        except Exception as e:
            print(f"Warning: could not pre-cache R-curve: {e}")

        self._setup_layout()
        self._setup_callbacks()

        self.r_curve = _load_r_curve()

        pthtemp = os.path.join(sys.path[0] if sys.path[0] else '.', 'Data', '007437_prism_clear_v3.1_1D.fits')
        with pyfits.open(pthtemp) as hdul:
            self.obs_wave = hdul['WAVELENGTH'].data * 1e6

        mc = self._build_model_components(5.5, 8, 1000, 1.0, 0.02, -2.0, 0.1)
        self.model = pipes.model_galaxy(mc, spec_wavs=self.obs_wave * 1e4)


    def _build_model_components(self, z, Mass, age, tau, Z, U, Av):
        delayed = {'age': age, 'tau': tau, 'massformed': Mass, 'metallicity': Z}
        dust    = {'type': 'Calzetti', 'Av': Av}
        mc = {'redshift': z, 'delayed': delayed, 'dust': dust}
        if U > -4:
            mc['nebular'] = {'logU': U}
        mc['R_curve'] = self.r_curve
        return mc


    def _generate_spectrum(self, dataset_key, Mass, age, tau, Z, U, Av):
        """Return model_spectrum (N×2 array) or None. Entirely local — no shared state."""
        data = _load_data(dataset_key)
        
        mc = self._build_model_components(data['z'], Mass, age, tau, Z, U, Av)
        self.model.update(mc)
        return self.model.spectrum, self.model.sfh   # both local

    # ------------------------------------------------------------------
    def _setup_layout(self):
        self.app.layout = dbc.Container([
            dcc.Store(id="active-dataset", data='SF943'),
            dcc.Interval(id='init-trigger', interval=500, max_intervals=1),

            dbc.Row([dbc.Col([html.H1("JADES Stellar Population Lab",
                                      className="text-center mb-4")], width=12)]),
            dbc.Row([
                dbc.Col(
                    html.A("📖 Learn the science behind this lab",
                           href="../learn-stellar-pop", target="_blank",
                           style={'color': '#1b825e', 'fontWeight': 'bold', 'textDecoration': 'none'}),
                    width=12, className="text-center mb-3"
                )
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button(key, id=f"btn-{key}", color="info", size="sm")
                        for key in DATA_FILES
                    ], className="mb-3")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(id="main-plot", style={'height': '600px'})], width=9),
                dbc.Col([dcc.Graph(id="sfh-plot",  style={'height': '600px'})], width=3),
            ], style={'margin-bottom': '20px'}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label(["Mass [log(M☉)] ",
                            html.Span("ℹ", id="tooltip-mass",
                                     style={"color":"#17a2b8","cursor":"pointer","fontSize":"16px"})],
                            className="fw-bold mb-2"),
                        dbc.Tooltip("Total stellar mass of the galaxy in solar masses (log scale)",
                                    target="tooltip-mass", placement="right"),
                        dcc.Slider(id="mass-slider", min=7, max=12, step=0.1, value=8,
                                   marks={i: str(i) for i in range(7, 13)},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], className="mb-4"),
                    html.Div([
                        html.Label(["Radiation Strength [log U] ",
                            html.Span("ℹ", id="tooltip-logU",
                                     style={"color":"#17a2b8","cursor":"pointer","fontSize":"16px"})],
                            className="fw-bold mb-2"),
                        dbc.Tooltip("Ionization parameter: ratio of ionising photon density to gas density",
                                    target="tooltip-logU", placement="right"),
                        dcc.Slider(id="logU-slider", min=-4.01, max=-1, step=0.1, value=-2.5,
                                   marks={-4:'-4',-3:'-3',-2:'-2',-1:'-1'},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], className="mb-4"),
                    html.Div([
                        html.Label(["Heavy Elements [Z/Z☉] ",
                            html.Span("ℹ", id="tooltip-metal",
                                     style={"color":"#17a2b8","cursor":"pointer","fontSize":"16px"})],
                            className="fw-bold mb-2"),
                        dbc.Tooltip("Metallicity: abundance of elements heavier than helium relative to the Sun",
                                    target="tooltip-metal", placement="right"),
                        dcc.Slider(id="metal-slider", min=0.01, max=1.4, step=0.05, value=0.5,
                                   marks={0:'0', 0.5:'0.5', 1:'1', 1.4:'1.4'},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], className="mb-4"),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Label(["Age of Stars [log Gyr] ",
                            html.Span("ℹ", id="tooltip-age",
                                     style={"color":"#17a2b8","cursor":"pointer","fontSize":"16px"})],
                            className="fw-bold mb-2"),
                        dbc.Tooltip("Time since the onset of star formation in billions of years",
                                    target="tooltip-age", placement="right"),
                        dcc.Slider(id="age-slider", min=-2, max=1, step=0.1, value=-1,
                                   marks={-2:'0.01',-1:'0.1',0:'1',1:'10'},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], className="mb-4"),
                    html.Div([
                        html.Label(["Decline of Stars [log Gyr] ",
                            html.Span("ℹ", id="tooltip-tau",
                                     style={"color":"#17a2b8","cursor":"pointer","fontSize":"16px"})],
                            className="fw-bold mb-2"),
                        dbc.Tooltip("e-folding timescale: how quickly star formation declines",
                                    target="tooltip-tau", placement="right"),
                        dcc.Slider(id="tau-slider", min=-2, max=1, step=0.1, value=-1.4,
                                   marks={-2:'0.01',-1:'0.1',0:'1',1:'10'},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], className="mb-4"),
                    html.Div([
                        html.Label(["Dust [Av mag] ",
                            html.Span("ℹ", id="tooltip-dust",
                                     style={"color":"#17a2b8","cursor":"pointer","fontSize":"16px"})],
                            className="fw-bold mb-2"),
                        dbc.Tooltip("Visual extinction: amount of light absorbed by interstellar dust",
                                    target="tooltip-dust", placement="right"),
                        dcc.Slider(id="dust-slider", min=0, max=3, step=0.1, value=0.5,
                                   marks={0:'0',1:'1',2:'2',3:'3'},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], className="mb-4"),
                ], width=6),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    html.Hr(className="my-4"),
                    dcc.Graph(id="sfr-mass-plot", style={'height': '500px'}),
                ], width=12)
            ], className="mt-3"),
        ], fluid=True)

    # ------------------------------------------------------------------
    def _setup_callbacks(self):
        @self.app.callback(
            [Output("main-plot",     "figure"),
             Output("sfh-plot",      "figure"),
             Output("sfr-mass-plot", "figure"),
             Output("active-dataset","data")],
            [Input("mass-slider",  "value"),
             Input("logU-slider",  "value"),
             Input("metal-slider", "value"),
             Input("age-slider",   "value"),
             Input("tau-slider",   "value"),
             Input("dust-slider",  "value"),
             Input("init-trigger", "n_intervals")] +
            [Input(f"btn-{key}", "n_clicks") for key in DATA_FILES],
            State("active-dataset", "data"),
            prevent_initial_call=True,
        )
        def update_app(*args):
            n_datasets = len(DATA_FILES)
            Mass_log, U, Z, age_log, tau_log, Av = args[:6]
            btn_args        = args[7:7 + n_datasets]
            current_dataset = args[7 + n_datasets] or 'SF943'

            # Resolve which dataset is active
            new_dataset = current_dataset
            ctx = callback_context
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                for i, key in enumerate(DATA_FILES):
                    if trigger_id == f"btn-{key}" and btn_args[i]:
                        new_dataset = key
                        break

            # Convert log-scale sliders
            Mass = Mass_log
            age  = 10 ** age_log
            tau  = 10 ** tau_log

            # Load data (from cache) — read-only, safe to share
            data = _load_data(new_dataset)

            # Generate model entirely in local scope — never stored on self
            model_spectrum = None
            sfh_obj        = None
            if BAGPIPES_AVAILABLE:
                try:
                    model_spectrum, sfh_obj = self._generate_spectrum(
                        new_dataset, Mass, age, tau, Z, U, Av)
                except Exception as e:
                    print(f"Model generation failed: {e}")
                    traceback.print_exc()

            return (
                _build_main_plot(data, model_spectrum, data['z'], data['target']),
                _build_sfh_plot(sfh_obj, data['z']) if sfh_obj else go.Figure(),
                _build_sfr_mass_plot(sfh_obj, Mass, data['z']) if sfh_obj else go.Figure(),
                new_dataset,
            )