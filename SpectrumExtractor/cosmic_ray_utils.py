""" This module contains the custom routines for comic ray filtering """

import numpy as np
from scipy import ndimage

from .laplace_filter import nlaplace_filter_intrace_super


def rebin(a, *args):
    """ Updated based on Scipy cookbook : https://scipy-cookbook.readthedocs.io/items/Rebinning.html
    rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    """
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape)//np.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
             ['/factor[%d]'%i for i in range(lenShape)]
    # print(''.join(evList))
    return eval(''.join(evList))

def fit_slope_1d(X,Y):
    """ Returns the slope and intercept of the the line Y = slope*X +alpha """
    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sxx = np.sum(np.power(X,2))
    Sxy = np.sum(X*Y)
    Syy = np.sum(np.power(Y,2))
    n = len(X)*1.
    slope = (n*Sxy - Sx*Sy)/(n*Sxx-Sx**2)
    alpha = Sy/n - slope*Sx/n
    return slope, alpha

def predict_quantile_value(data,q_to_predict=0.0001,q_nodes=(0.001,0.01,0.05,0.1)):
    """ Returns a robust prediction of the quantile value by extapolation .
    Note: input q_nodes are defined such that 1-q_nodes gives the actual quantile"""
    q_nodes = np.array(q_nodes)
    q_values = np.quantile(data,1-q_nodes)
    # Fitting a straight line in log space works well for many long tail distributions
    x = np.log10(q_nodes) - np.log10(q_to_predict)
    slope, intercept = fit_slope_1d(x,q_values)
    return intercept

def threshold_in_laplace_space(inputArray,thresh=1.4,pixelMask=True,TopEdgeCoord=None,BottomEdgeCoord=None):
    """ Threshold in the lapace space of the inputArray image
    Input:
        inputArray: 2D numpy array
                   Input array to threshold
        thresh: float
                  The factor to be multiplied to the lapalacian spread estimate to define the threshold
        pixelMask: numpy boolean mask
                  The mask for the pixels we care about in the inputArray
        TopEdgeCoord: 1D numpy array
                  The coordinates of the top (larger I value) edge of the region we care about in the inputArray
        BottomEdgeCoord: 1D numpy array
                  The coordinates of the bottom (smaller I value) edge of the region we care about in the inputArray

    Returns:
        outputmask: 1D numpy boolean mask
                   Mask of the pixes above the threshold
    """
    # Inorder to prevent damage around nonfinite pixels while calculating laplace, first extrapolate to those nearby pixels
    laplaceArray = fill_nearby_nonfinitepixels(inputArray,span=1,axis=1)
    laplaceArray = fill_nearby_nonfinitepixels(laplaceArray,span=1,axis=0)

    # # In order to avoid negative shadows from CR hits on neighbouring pixels, we need to supersample before Laplace filtering
    # original_img_size = inputArray.shape
    # # Supersample the image to make it twice the size in both axis
    # laplaceArray = ndimage.zoom(laplaceArray,2,order=0)
    # # Calculate the negative Lapace transform to obtain positive values for CR hits pixels
    # laplaceArray = ndimage.laplace(laplaceArray)*-1
    # # Set all the negative values to 0
    # laplaceArray[laplaceArray<0] = 0
    # # rebin back to original size
    # laplaceArray = rebin(laplaceArray,original_img_size[0],original_img_size[1])
    ###### Faster fortran implementation of the above steps ################
    laplaceArray = np.ascontiguousarray(nlaplace_filter_intrace_super(np.asfortranarray(laplaceArray),BottomEdgeCoord,TopEdgeCoord))

    # spread = np.std(laplaceArray[np.isfinite(laplaceArray)&pixelMask]) # Not robust enough
    # Use the quantile curve to predict the robust edge of the distribution
    spread = predict_quantile_value(laplaceArray[np.isfinite(laplaceArray)&pixelMask],q_to_predict=0.0001,q_nodes=(0.001,0.01,0.05,0.1))
    # Now identify the ouliers in this Lapace transformed image.
    outputmask = laplaceArray > thresh* spread

    return outputmask


def fill_nearby_nonfinitepixels(inputArray,span=1,axis=1):
    """ Extrapolate into nearby non finite value pixels along input axis """
    badpixmask = ~np.isfinite(inputArray)
    outputArray = np.copy(inputArray)
    # Use the np.roll to shift the array to fill in the not finite values with nearest entry
    for shift in range(1,span+1):
        for sign in (-1,1):
            shifted_inputArray = np.roll(outputArray,shift=shift*sign,axis=axis)
            recovered_pixmask = badpixmask & np.isfinite(shifted_inputArray)
            outputArray[recovered_pixmask] = shifted_inputArray[recovered_pixmask]
    return outputArray

def sum_filter_aroundnans(inputArray,pixsmoothing=5,axis=1):
    """ Return a sum filtered version of InputArray even if the array containa NaN or inf
    """
    outputArray = fill_nearby_nonfinitepixels(inputArray,span=int(pixsmoothing//2),axis=axis)
    # Set the remaining pixels which will not matter for smoothing to 0
    outputArray[~np.isfinite(outputArray)] = 0
    outputArray = ndimage.uniform_filter1d(outputArray,pixsmoothing,axis=axis)*pixsmoothing
    # Set back all the original non finite pixels back to what it was
    outputArray[badpixmask] = inputArray[badpixmask]
    return outputArray
