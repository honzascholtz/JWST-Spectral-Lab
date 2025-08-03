
# SED Fitter

SED Fitter is an outreach activity designed to show students and the general public how SED fitting works and what can we learn from a spectrum. The entire thing is a wrapper around Bagpipes to create a best fitting model to various JWST prism spectra. 

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")


# How to run 

In order to run the code we need to install Bagpipes from Adam Carnall. You can find more info about Bagpipes <a href="https://bagpipes.readthedocs.io/en/latest/" target=_blank>here</a>.

```
pip install bagpipes

python SED_fitter.py

```

# Grid models updates

If you are a scientist, I would highly recommend to create a separate Conda/other enviroment to install this. This is mostly due to the need to ensure that the nebular grids are updates for high logU and low metallicities. The code check if these are present in your grid nebular models and copies over the ones in the package. Although it saves them in bagpipes/models/grids/old/ it is still better to create a new enviroment so it doesnt ruin your custom nebular models. 
