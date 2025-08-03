# This file checks that Bagpipes has properely setup grids and if not, it will copy over new grids.
import os
import sys
from threading import local
import astropy.io.fits as pyfits
import numpy as np


def check_setup():
    """
    Check if the Bagpipes setup is correct.
    If not, copy the grids from the default location to the user's directory.
    """
    import bagpipes as pipes

    path_bagpipes = os.path.dirname(pipes.__file__)

    with pyfits.open(os.path.join(path_bagpipes, 'models/grids', 'bc03_miles_nebular_line_grids.fits')) as hdulist:

        logUs = []
        for hdu in hdulist:
            if hdu.name != 'PRIMARY':
                name = hdu.name
                logUs.append(float(name.split('_')[-1]))
    logUs = np.array(logUs)
    if np.where(logUs > -2.0)[0].size > 0:
        print('Bagpipes setup is correct.')
    
    if np.where(logUs > -2.0)[0].size == 0:
        print('Bagpipes setup is incorrect. Copying grids from default location...')
        print("Would you like me to copy the grids from the default location? (yes/no)")
        answer = input()
        
        if answer== 'yes' or answer == 'Yes':
            local_pth = sys.path[0]+'Grids'
            
            # Backing up the old grids just in case
            os.system(f'mkdir -p {path_bagpipes+"/models/grids/old"}')

            input_path = os.path.join(path_bagpipes, 'models/grids', 'bc03_miles_nebular_line_grids.fits')
            os.system(f'cp {input_path} {input_path.replace("models/grids", "models/grids/old")}')

            input_path = os.path.join(path_bagpipes, 'models/grids', 'bc03_miles_nebular_cont_grids.fits')
            os.system(f'cp {input_path} {input_path.replace("models/grids", "models/grids/old")}')

            print(f'Old grids backed up to {path_bagpipes}/models/grids/old')

            # Now copying over the new grids
            input_path = os.path.join(local_pth, 'Grids/bc03_miles_nebular_line_grids.fits')
            os.system(f'cp {input_path} {path_bagpipes+"/models/grids/bc03_miles_nebular_line_grids.fits"}')

            input_path = os.path.join(local_pth, 'Grids/bc03_miles_nebular_cont_grids.fits')
            os.system(f'cp {input_path} {path_bagpipes+"/models/grids/bc03_miles_nebular_cont_grids.fits"}')

            print('Grids copied successfully. You can now continue.')
        else:
            print('Ok things will not work properly. Please copy the grids manually.')

check_setup()