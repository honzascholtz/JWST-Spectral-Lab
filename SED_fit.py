#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun April 27 21:23:34 2025

@author: jansen
"""

#importing modules
from astropy.io import fits as pyfits
from sympy import frac
from zmq import Frame
import Graph_setup as gst
import numpy as np

nan= float('nan')

pi= np.pi
e= np.e
c= 3.*10**8

fsz = gst.graph_format()

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.cm
from brokenaxes import brokenaxes
from matplotlib.widgets import Slider, Button, RadioButtons, RangeSlider, TextBox


import matplotlib.pyplot as plt
import numpy as np
import bagpipes as pipes

wave = np.arange(0.05 * 1e4, 1* 5e4, 10.0)

import sys
sys.setrecursionlimit(1500)

wave_grid = np.logspace(2,4,1000)

class Viz_outreach:
    '''Class to visualize SED'''
    def __init__(self, ):
        """
        Load the fit and the data
        Returns
        -------
        """

        self.fig = plt.figure(figsize=(15.6, 8))
        self.fig.canvas.manager.set_window_title('vicube')
        gs = self.fig.add_gridspec(
            3, 3, height_ratios=(2.2,0.7,1.3), width_ratios=(1,1,1))
        
        self.ax0 = self.fig.add_subplot(gs[0,:])
        self.axres = self.fig.add_subplot(gs[1,:], sharex=self.ax0)

        plt.subplots_adjust(hspace=0)

        with pyfits.open('/Users/jansen/JADES/MSA_data/goods-s-deephst/prism_clear/10058975_prism_clear_v5.0_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7
            
        self.ax0.plot(self.data_wave, self.data_flux, color='black', drawstyle='steps-mid')
        self.z = 7.43
        self.Mass = 9.
        self.age = 0.3
        self.tau = 0.3
        self.Z = 1,
        self.U = -3
        self.Av = 0.5

        self.wave = np.arange(0.05 * 1e4, 1* 5e4, 10.0)


        self.pregenerate_model()
        

        self.ax0.set_ylabel("Flux")
        self.ax0.set_xlabel("Wavelength [microns]")

        y0 = 0.25
        dy = 0.04
        x0 = 0.1

        # Redshift slider
        self.slider_z_ax = plt.axes([x0, y0, 0.8, 0.03])
        self.z_slider = Slider(self.slider_z_ax, 'redshift', valmin=1,
                                   valmax=15,
                                    valinit=self.z)
        self.z_slider.on_changed(self.slide_update)

        # Mass Slider
        self.slider_mass_ax = plt.axes([x0, y0-dy, 0.8, 0.03])
        self.mass_slider = Slider(self.slider_mass_ax, 'Mass', valmin=7,
                                   valmax=12,
                                    valinit=self.Mass)
        self.mass_slider.on_changed(self.slide_update)

        # Log U slider
        self.slider_logU_ax = plt.axes([x0, y0-2*dy, 0.8, 0.03])
        self.logU_slider = Slider(self.slider_logU_ax, 'logU', valmin=-4.01,
                                   valmax=-1,
                                    valinit=-2)
        self.logU_slider.on_changed(self.slide_update)

        # Metal Slider
        self.slider_metal_ax = plt.axes([x0, y0-3*dy, 0.8, 0.03])
        self.metal_slider = Slider(self.slider_metal_ax, 'metal', valmin=0.01,
                                   valmax=1.4,
                                    valinit=0.5)
        self.metal_slider.on_changed(self.slide_update)
    
        # Age Slider
        self.slider_age_ax = plt.axes([x0, y0-4*dy, 0.8, 0.03])
        self.age_slider = Slider(self.slider_age_ax, 'age', valmin=-2,
                                 valmax=1,
                                 valinit=-1)
        self.age_slider.on_changed(self.slide_update)

        # Age Slider
        self.slider_tau_ax = plt.axes([x0, y0-5*dy, 0.8, 0.03])
        self.tau_slider = Slider(self.slider_tau_ax, 'Tau', valmin=-2,
                                 valmax=1,
                                 valinit=-1)
        self.tau_slider.on_changed(self.slide_update)

        # Dust Slider
        self.slider_dust_ax = plt.axes([x0, y0-6*dy, 0.8, 0.03])
        self.dust_slider = Slider(self.slider_dust_ax, 'Dust', valmin=0.,
                                 valmax=3,
                                 valinit=0.2)
        self.dust_slider.on_changed(self.slide_update)
        # Text box for redshift input
        axbox = plt.axes([0.95, y0, 0.03, 0.03])
        text_box = TextBox(axbox, '', initial='')
        text_box.on_submit(self.submit_redshift)

        # Button to load SF943 data
        axbutton1 = plt.axes([0.1,0.9, 0.05,0.05])
        self.SF943 = Button(axbutton1, 'SF943', color='lightblue', hovercolor='lightgreen')
        self.SF943.on_clicked(self.SF943_load)

        # Buttons to load other QC galaxies
        axbutton2 = plt.axes([0.2,0.9, 0.05,0.05])
        self.QC_galaxy = Button(axbutton2, 'QC_galaxy', color='lightblue', hovercolor='lightgreen')
        self.QC_galaxy.on_clicked(self.QC_galaxy_load)

        # Buttons to load other SF galaxies
        axbutton3 = plt.axes([0.3,0.9, 0.05,0.05])
        self.SF_galaxy = Button(axbutton3, 'SF_galaxy', color='lightblue', hovercolor='lightgreen')
        self.SF_galaxy.on_clicked(self.SF_galaxy_load)

        # Buttons to load other GSz14
        axbutton4 = plt.axes([0.4,0.9, 0.05,0.05])
        self.GSz14 = Button(axbutton4, 'GSz14', color='lightblue', hovercolor='lightgreen')
        self.GSz14.on_clicked(self.GSz14_load)

        self.generate_model()
        self.plot_general()
        plt.show()

    def submit_redshift(self, text): 
        ydata = float(text)
        self.z_slider.set_val(ydata)
        self.z = ydata
    
    def SF943_load(self, event):
        with pyfits.open('/Users/jansen/JADES/MSA_data/goods-s-deephst/prism_clear/10058975_prism_clear_v5.0_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.plot_general()
        
    def QC_galaxy_load(self, event):
        with pyfits.open('/Users/jansen/JADES/MSA_data/goods-s-ultradeep/prism_clear/199773_prism_clear_v5.0_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.plot_general()

    def SF_galaxy_load(self, event):
        with pyfits.open('/Users/jansen/JADES/MSA_data/goods-n-mediumjwst/prism_clear/000110_prism_clear_v5.0_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.plot_general()
    def GSz14_load(self, event):
        with pyfits.open('/Users/jansen/JADES/MSA_data/goods-s-deepjwst/prism_clear/183348_prism_clear_v5.0_1D.fits') as hdu:
            self.data_wave = hdu['WAVELENGTH'].data*1e6
            self.data_flux = hdu['DATA'].data*1e-7
            self.data_error = hdu['ERR'].data*1e-7

            self.plot_general()

    def slide_update(self,val):
        self.z = self.z_slider.val
        self.Mass = self.mass_slider.val
        self.U = self.logU_slider.val
        self.Z = self.metal_slider.val
        self.age = 10**self.age_slider.val
        self.tau = 10**self.tau_slider.val
        self.Av = self.dust_slider.val
        self.generate_model()
        self.plot_general()
    
    def pregenerate_model(self, ):
        """
        Pre-generate the model components to speed up the generation of the model
        """
        exponential = {}  # Tau model star formation history component
        exponential["age"] = self.age  # Gyr
        exponential["tau"] = self.tau  # Gyr
        exponential["massformed"] = self.Mass  # log_10(M*/M_solar)
        exponential["metallicity"] = self.Z  # Z/Z_oldsolar

        dust = {}  # Dust component
        dust["type"] = "Calzetti"  # Define the shape of the attenuation curve
        dust["Av"] = self.Av  # magnitudes

        model_components = {}  # The model components dictionary
        if self.U>-4:
            nebular = {}  # Nebular emission component
            nebular["logU"] = self.U  # log_10(ionization parameter)
            model_components["nebular"] = nebular
        
        
        model_components["redshift"] = self.z  # Observed redshift
        model_components["exponential"] = exponential
        model_components["dust"] = dust
        with pyfits.open("/Users/jansen/JADES/LSFs/jwst_nirspec_prism_disp.fits") as hdul:
            model_components["R_curve"] = np.c_[1e4*hdul[1].data["WAVELENGTH"], hdul[1].data["R"]]

        self.model = pipes.model_galaxy(
            model_components, spec_wavs=self.data_wave*1e4)

    def generate_model(self, ):
        
        exponential = {}  # Tau model star formation history component
        exponential["age"] = self.age  # Gyr
        exponential["tau"] = self.tau  # Gyr
        exponential["massformed"] = self.Mass  # log_10(M*/M_solar)
        exponential["metallicity"] = self.Z  # Z/Z_oldsolar

        dust = {}  # Dust component
        dust["type"] = "Calzetti"  # Define the shape of the attenuation curve
        dust["Av"] = self.Av  # magnitudes

        model_components = {}  # The model components dictionary
        if self.U>-5:
            nebular = {}  # Nebular emission component
            nebular["logU"] = self.U  # log_10(ionization parameter)
            model_components["nebular"] = nebular
        
        
        model_components["redshift"] = self.z  # Observed redshift
        model_components["exponential"] = exponential
        model_components["dust"] = dust
        model_components["R_curve"] = np.c_[self.wave,700 * np.ones_like(self.wave)] 


        self.model.update(model_components)

       
    def plot_general(self, event=None):
        self.ax0.clear()       
        self.axres.clear()

        if np.nansum(self.model.spectrum[:, 1]/1e-18) ==0:
            self.ax0.plot(self.data_wave, self.data_flux/1e-18, color='black', drawstyle='steps-mid')

            self.axres.plot(self.data_wave, (self.data_flux-self.model.spectrum[:, 1])/self.data_error, color='black', drawstyle='steps-mid')
            self.axres.axhline(0, color='red', lw=1.5, ls='--')

            self.ax0.set_ylabel("Flux [10$^{-18}$ erg/s/cm$^2$/Å]")
            self.axres.set_xlabel("Wavelength [microns]")
            self.ax0.set_xlim(0.5, 5.3)

            self.ax0.text(0.05, 0.9,'Your stars are older than the Universe, reduce Age parameter!',\
                        transform = self.ax0.transAxes)

            

        else:
            self.ax0.plot(self.model.spectrum[:, 0]/1e4, self.model.spectrum[:, 1]/1e-18, color='darkblue', drawstyle='steps-mid')
            self.ax0.plot(self.data_wave, self.data_flux/1e-18, color='black', drawstyle='steps-mid')

            self.axres.plot(self.data_wave, (self.data_flux-self.model.spectrum[:, 1])/self.data_error, color='black', drawstyle='steps-mid')
            self.axres.axhline(0, color='red', lw=1.5, ls='--')

            self.ax0.set_ylabel("Flux [10$^{-18}$ erg/s/cm$^2$/Å]")
            self.axres.set_xlabel("Wavelength [microns]")
            self.ax0.set_xlim(0.5, 5.3)

            self.score = np.nansum((self.data_flux- self.model.spectrum[:, 1])**2/self.data_error**2)/(len(self.data_flux)-6)

            self.ax0.text(0.05, 0.85,'(Smaller is better)\nscore: {:.2f}'.format(self.score),\
                        transform = self.ax0.transAxes, bbox=dict(boxstyle='Round,pad=0.01', facecolor='white',
                            alpha=1.0, edgecolor='none'))

            self.labels_eml()
        self.fig.canvas.draw()
  

    def labels_eml(self,):
        emlines = {
            'CIII]'   : ( 1907.,  0.00,  0.95),
            'MgII'    : ( 2797.,  0.00,  0.95),
            '[OII]'   : ( 3728.,  -0.02,  0.95),
            '[NeIII]' : ( 3869.860, -0.00, 0.95),
            'H$\\delta$'         : ( 4102.860, -0.00, 0.95),
            'H$\\gamma$'          : ( 4341.647,  0.025, 0.95),
            'H$\\beta$'           : ( 4862.647, -0.03, 0.9),
            '[OII]'  : ( np.array((4960.0,5008.0)),  0.03,  0.95),
            '[OI]'    : ( np.array((6302.0,6365.0)),  -0.03,  0.95),
            'NaI'       : ( 5891.583,  0.00,  0.95),
            'H$\\alpha$'          : (  6564.522, -0.00, 0.95),
            '[SII]'   : ( 6725,  0.03,  0.9),
            '[SIII]'  : ( np.array((9070.0,9535.0)),  0.0,  0.95),
            'HeI}'     : ( 10832.1, -0.03, 0.95),
            'Pa$\\gamma$'         : ( 10940.978,  0.03, 0.95),
            '[FeII]'  : ( 12570.200, -0.015, 0.50),
            'Pa$\\beta$'          : ( 12821.432,  0.015, 0.95),
        }
        n_lines = len(emlines)+1
        cmap = matplotlib.colormaps['nipy_spectral']
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=n_lines)
        transMix = matplotlib.transforms.blended_transform_factory(
            self.ax0.transData, self.ax0.transAxes)

        for i,(line,lineprop) in enumerate(emlines.items()):
            waves, offset, y_position = lineprop
            waves = np.atleast_1d(waves)
            waves= np.array(waves, dtype=float)
            waves *= (1+self.z)/1.e4
            wave = np.mean(waves)
            #if not 1.001*self.data_wave<wave<self.data_wave[-1]*0.999: continue
            if not self.ax0.get_xlim()[0]*1.001<wave<self.ax0.get_xlim()[1]*0.999: continue
        
            color = cmap(norm(float(i)))
            where_line = np.argmin(np.abs(self.wave-wave))
            where_line = slice(where_line-5, where_line+6, 1)
            #data_at_line = (
            #    np.min(spec1d[where_line]) if pre_dash
            #    else np.max(spec1d[where_line])
            #)
            va = 'center'
            if y_position<0.05: va='bottom'
            if y_position>0.90: va='top'
            
            line = line
            for w in waves:
                    self.ax0.axvline(w, color=color, lw=1.5, alpha=0.5, ls='--', zorder=0)
                            
            self.ax0.text(
                wave+offset, y_position, line,
                color=color, transform=transMix, va=va, ha='center',
                fontsize=12,
                rotation='vertical',
                bbox=dict(boxstyle='Round,pad=0.01', facecolor='white',
                            alpha=1.0, edgecolor='none'),
                zorder=99,
                )



    

inst = Viz_outreach()
        
