import datashader
import bebi103
import bebi103.image
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import bokeh.io
import bokeh.plotting
import holoviews as hv
import skimage
import glob
bokeh.io.output_notebook()
hv.extension('bokeh')
import colorcet
from base64 import b16encode
import bokeh_catplot
from skimage import feature
import scipy
from panel.interact import fixed
from skimage import morphology

import scipy.stats as st
import warnings
import scipy
import scipy.optimize


# functions for each model
def linear_mod(a0, k, times):
    return a0*(1+k*times)

def exp_mod(a0, k, times):
    return a0*np.exp(k*times)

# residual for linear model
def resid_lin(params, areas, times):
    a0, k = params

    return areas - a0*(1+k*times)

# residual for exponential function
def resid_exp(params, areas, times):
    a0, k = params

    return areas - a0*np.exp(k*times)

# least squares regression for given residual function
def least_squares_regression(resid_fun, areas, times):
    res = scipy.optimize.least_squares(
        resid_fun, np.array([1, 0.1]), args=(areas, times), bounds=([0,-np.inf], [np.inf, np.inf])
    )

    # compute residual sum of squares from optimal params
    rss_mle = np.sum(resid_fun(res.x, areas, times)**2)

    # compute MLE for sigma
    sigma_mle = np.sqrt(rss_mle / len(areas))

    return (res.x, sigma_mle)

# let's generate area data
def gen_area_data_lin(params, times, size, rg):
    a0, k, sigma = params
    areas = linear_mod(a0, k, times)

    return np.vstack((times, rg.normal(areas, sigma))).transpose()

def gen_area_data_exp(params, times, size, rg):
    a0, k, sigma = params
    areas = exp_mod(a0, k, times)

    return np.vstack((times, rg.normal(areas, sigma))).transpose()

# the log likelihood of the linear function, given by
# the normal of the residual, which is defined as the
# difference between the estimated data and the measured data
def log_likelihood_lin_mod(params, areas, times):
    a0, k, sigma = params

    logpdf = st.norm.logpdf(areas, a0*(1+k*times), sigma)

    if True in np.isnan(logpdf):
        print('return minus inf')
        return -np.inf

    return np.sum(logpdf)

# the log likelihood of the exponential function, given by
# the normal of the residual, which is defined as the
# difference between the estimated data and the measured data
def log_likelihood_exp_mod(params, areas, times):
    a0, k, sigma = params

    # when sigma is equal to zero, we get that the logpdf of the
    # normal distribution returns a Nan value, so just set sigma
    # to be a very small number
    if sigma == 0:
        sigma = 10**-10

    logpdf = st.norm.logpdf(areas, a0*np.exp(k*times), sigma)

    if True in np.isnan(logpdf):
        print('return minus inf')
        return -np.inf

    return np.sum(logpdf)
