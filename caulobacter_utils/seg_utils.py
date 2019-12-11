import bebi103
import bebi103.image
import numpy as np
import pandas as pd
import skimage
import glob
import colorcet
from skimage import morphology
from skimage import feature
import scipy
import bokeh.io
import bokeh.plotting

# our segmentation algorithm function
def get_binary_im(im, thresh=0.001, show_each_step=False):
    # Convert image to float
    im_float = (im.astype(float) - im.min()) / (im.max() - im.min())

    # Make the structuring element 25 pixel radius disk
    selem = skimage.morphology.disk(25)

    # Do the mean filter
    im_mean = skimage.filters.rank.mean(im, selem)

    # Threshhold based on mean filter
    im_mask = im < 0.85 * im_mean

    im_float_masked = im_mask * im_float

    # do Laplacian of Gaussian
    im_LoG = scipy.ndimage.filters.gaussian_laplace(im_float_masked, 2.0)
    # 3x3 square structuring element
    selem = skimage.morphology.square(3)

    # Do max filter and min filter
    im_LoG_max = scipy.ndimage.filters.maximum_filter(im_LoG, footprint=selem)
    im_LoG_min = scipy.ndimage.filters.minimum_filter(im_LoG, footprint=selem)

    # Image of zero-crossings
    im_edge = ((im_LoG >= 0) & (im_LoG_min < 0)) | ((im_LoG <= 0) & (im_LoG_max > 0))

    # Using code from the lesson
    # Compute gradients using Sobel filter
    im_grad = skimage.filters.sobel(im)

    im_threshed = (im_edge & (im_grad >= thresh))

    im_skeleton = skimage.morphology.skeletonize(im_threshed)

    # Fill holes
    im_bw = scipy.ndimage.morphology.binary_fill_holes(im_skeleton)

    # Remove small objectes that are not bacteria
    im_bigguns = skimage.morphology.remove_small_objects(im_bw, min_size=400)

    im_cleared = skimage.segmentation.clear_border(im_bigguns, buffer_size=5)

    if show_each_step:
        transforms = [im_float, im_float_masked, im_LoG, im_edge, im_grad, im_threshed,
                      im_skeleton, im_bw, im_bigguns, im_cleared]

        [bokeh.io.show(bebi103.image.imshow(img)) for img in transforms]
    return im_cleared

# get area for a given binary image returned from the above function
def get_area_binary_im(im_binary):
    return im_binary.sum()
