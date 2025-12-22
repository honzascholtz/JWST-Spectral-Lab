#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JADES Labs - Flask Integration with Multiple Dash Apps
Four separate Dash apps integrated with Flask

@author: jansen (converted to Flask)
"""

from flask import Flask,  render_template
import Labs.Phot_flask as phot
import Labs.redshift_flask as redshift
import Labs.stellar_flask as stellar
import Labs.ifu_flask as ifu


# Create Flask server
server = Flask(__name__)

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
# ============================================================================
photometry_app = phot.JADES_photo_lab(server, url_base_pathname='/photometry/')
redshift_app = redshift.Redshift_lab(server, url_base_pathname='/redshift/')
stellar_app = stellar.Stellar_pop_lab(server, url_base_pathname='/stellar-pop/')
ifu_app = ifu.IFU_lab(server, url_base_pathname='/ifu/')

if __name__ == '__main__':
    server.run(debug=True)