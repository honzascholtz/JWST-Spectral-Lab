import sys
import os
# Set the next line if you want to serve static images via Apache rather than the flask app.
os.environ['STATICPATH']='https://jades.herts.ac.uk'
os.environ['APPLICATION_ROOT']='/JADES-lab'
os.environ['MPLCONFIGDIR']='/tmp/jades-lab-matplotlib'
basedir='/beegfs/car/jades'
os.chdir(basedir)
sys.path.insert(0, basedir+'/JADES-lab/env/lib64/python3.9/site-packages')
sys.path.insert(0, basedir+'/JADES-lab')
from JADES_lab import server as application
