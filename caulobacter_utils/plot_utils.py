import datashader
import bebi103
import bebi103.image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bokeh.io
import bokeh.plotting
import holoviews as hv
import skimage
import glob
hv.extension('bokeh')
import colorcet
from base64 import b16encode


def get_n_categories_cmap(n):
    # Allow us to have variable length categorical color maps
    # by repeating them
    cmap = np.array(plt.cm.tab10(range(10), bytes=True))[:, :3]
    cmap = np.tile(cmap, (n // 10 + 1, 1))
    hex_cmap = ['#%02x%02x%02x' % tuple(triplet) for triplet in cmap]
    return hex_cmap
    
