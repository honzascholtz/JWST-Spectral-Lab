#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JADES Labs - Flask Integration with Multiple Dash Apps
Four separate Dash apps integrated with Flask

@author: jansen (converted to Flask)
"""

from flask import Flask, render_template
import Labs.Phot_flask as phot
import Labs.redshift_flask as redshift
import Labs.stellar_flask as stellar
import Labs.ifu_flask as ifu
import os

import multiprocessing
multiprocessing.set_start_method('fork')

# Create Flask server
server = Flask(__name__)

# Deal with a non-standard route if there is one
try:
    prefix=os.environ['APPLICATION_ROOT']
except KeyError:
    prefix=""


# ============================================================================
# FLASK ROUTES
# ============================================================================
@server.route('/')
def index():
    """Home page with links to all four labs"""
    return render_template('index.html')

@server.route('/api/health')
def health():
    """API health check endpoint"""
    return {'status': 'healthy', 'message': 'Flask + 4 Dash apps running'}

# ============================================================================
# INITIALIZE ALL FOUR APPS
# Each app gets a unique app_name so Dash doesn't collide on asset routes.
# Each lab's __init__ must accept app_name and pass it as the first argument
# to dash.Dash(...) instead of __name__.
# ============================================================================
photometry_app = phot.JADES_photo_lab(
    server, 
    requests_pathname_prefix=prefix+'/photometry/',## External browser URL
    routes_pathname_prefix='/photometry/' ## Internal Flask route
)

redshift_app = redshift.Redshift_lab(
    server, 
    requests_pathname_prefix=prefix+'/redshift/',
    routes_pathname_prefix='/redshift/'
)

stellar_app = stellar.Stellar_pop_lab(
    server, 
    requests_pathname_prefix=prefix+'/stellar-pop/',
    routes_pathname_prefix='/stellar-pop/'
)

ifu_app = ifu.IFU_lab(
    server, 
    requests_pathname_prefix=prefix+'/ifu/',
    routes_pathname_prefix='/ifu/'
)

if __name__ == '__main__':
    server.run(debug=True)
