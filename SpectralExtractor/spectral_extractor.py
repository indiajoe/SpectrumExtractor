#!/usr/bin/env python
""" This tool is to extract 1D spectrum form a 2D image """
import numpy as np
import numpy.ma
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage import filters
from skimage import morphology
from scipy import ndimage
import scipy.interpolate as interp
import scipy.optimize as optimize
from functools32 import partial
from RVEstimator.interpolators import BandLimited2DInterpolator


def ImageThreshold(imgfile,bsize=401,offset=0, minarea=0,ShowPlot=False):
    """ Returns adaptive thresholded image mask """
    imgArray = fits.getdata(imgfile)
    # Adaptive thresholding..
    ThresholdedMask = imgArray > filters.threshold_local(imgArray, bsize,offset=offset)
    if minarea:
        print('Regions of area less than {0} are discarded'.format(minarea))
        ThresholdedMask = morphology.remove_small_objects(ThresholdedMask,minarea)
        
    if ShowPlot:
        plt.imshow(np.ma.array(imgArray,mask=~ThresholdedMask))
        plt.colorbar()
        plt.show()
    return ThresholdedMask

def LabelDisjointRegions(mask,DirectlyEnterRelabel= False):
    """ Interactively label disjoint regions in a mask """

    # Label the remaining regions
    labeled_array, num_features = ndimage.label(mask)

    if not DirectlyEnterRelabel:
        NewLabeledArray = labeled_array.copy()
        sugg = 1
        print('Enter q to discard all remaining regions')
        for label in np.unique(NewLabeledArray):
            if label == 0: # Skip 0 label
                continue
            plt.clf()
            # plt.imshow(np.ma.array(LampTargetArray,mask=~(labeled_array==label)))
            # plt.colorbar()
            plt.imshow(np.ma.array(labeled_array,mask=~(labeled_array==label)))
            plt.show(block=False)
            print('Area of region = {0}'.format(size))
            print('Current Label : {0}'.format(label))
            newlbl = sugg
            uinput = raw_input('Enter New Region label  (default: {0}): '.format(sugg)).strip()
            if uinput: #User provided input
                if uinput == 'q':
                    # Set all the labels current and above this value to Zero and exit
                    NewLabeledArray[labeled_array >= label]= 0
                    break
                else:
                    newlbl = int(uinput)

            NewLabeledArray[labeled_array==label] = newlbl
            sugg = newlbl + 1

    else:
        plt.imshow(labeled_array)
        plt.colorbar()
        plt.show(block=False)
        NewLabeledArray = np.zeros(labeled_array.shape)

        print('Create a file with the label renames to be executed.')
        print('File format should be 2 columns :  OldLabel NewLabel')
        print('Note: Only assigned labels will be kept..')
        uinputfile = raw_input('Enter the filename :').strip()
        if uinputfile: #User provided input
            with open(uinputfile,'r') as labelchangefile:
                labelchangelist = [tuple(int(i) for i in entry.rstrip().split()) \
                                   for entry in labelchangefile \
                                   if len(entry.rstrip().split()) == 2]
            print('Label Renaming :',labelchangelist)
            for oldlabel, newlabel in labelchangelist:
                # Now assign new label
                NewLabeledArray[labeled_array == oldlabel] = newlabel
                
        else:
            print('No input given. No relabelling done..')

        
        
    plt.clf()
    print('New labelled regions')
    plt.imshow(NewLabeledArray)
    plt.colorbar()
    plt.show()

    return NewLabeledArray

def errorfuncProfileFit(p,psf=None,xdata=None, ydata=None):
    """ Error function to minimise the profile psf(xdata) fit on to ydata """
    return p[0]*psf(xdata-p[1]) - ydata

def FitApertureCenters(SpectrumFile,ApertureLabel,apertures=None,
                           apwindow=(-7,+7),dispersion_Xaxis = True, ShowPlot=False):
    """ Fits the center of the apertures in the spectrum"""
    ApertureCenters = {}
    print('Extracting Aperture Centers')

    if apertures is None : 
        apertures = np.sort(np.unique(ApertureLabel))[1:] # Remove 0

    if isinstance(SpectrumFile,str):
        ImageArray = fits.getdata(SpectrumFile)
    else:
        ImageArray = SpectrumFile

    # Transpose the image if dispersion is not along X axis
    if not dispersion_Xaxis:
        ImageArray = ImageArray.T
    for aper in apertures:
        print('Aperture : {0}'.format(aper))
        # Crude center of the apperture along dispersion
        aperCenter = np.ma.mean(np.ma.array(np.indices(ApertureLabel.shape)[0], 
                                            mask=~(ApertureLabel==aper)),axis=0)
        # Masked values in the aperCenter means no pixel in the threshold map!

        # Shift and combine to create a PSF for the aperture
        # Pixels along the dispersion direction
        dpix = np.arange(len(aperCenter))[~aperCenter.mask]
        xdpixCenter = aperCenter[~aperCenter.mask].data
        xdapL2Upix = np.rint(xdpixCenter[:,np.newaxis]+\
                             np.arange(apwindow[0],apwindow[1]+1) ).astype(np.int) # round of to nearest integer
        # Extrapolate the coordinates at edges of array (for anyway useless orders) at the edge of detector
        xdapL2Upix[xdapL2Upix >= ImageArray.shape[0]] = ImageArray.shape[0]-1
        xdapL2Upix[xdapL2Upix < 0] = 0

        Rectifiedarray = ImageArray[[xdapL2Upix,
                                     np.repeat(dpix[:,np.newaxis],xdapL2Upix.shape[1],axis=1)]]
        mean_profile = np.mean(Rectifiedarray,axis=0)
        # PSF interpolated function
        psf = interp.InterpolatedUnivariateSpline(np.arange(apwindow[0],apwindow[1]+1), mean_profile)

        # Loop along the dispersion pixels
        # Fit the profile for each column to obtian the accurate centroid in each column
        CenterXDcoo = []
        
        ampl = 1
        for d,xd,flux in zip(dpix,xdapL2Upix,Rectifiedarray):
            # initial estimate
            p0 = [ampl,xd[len(xd)/2]]
            p,ier = optimize.leastsq(errorfuncProfileFit, p0, args=(psf,xd,flux))
            CenterXDcoo.append(p[1])
            ampl = p[0] # update with the lastest amplitude estimate

        ApertureCenters[aper] = np.array([dpix,CenterXDcoo])
        if ShowPlot:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax1.plot(ApertureCenters[aper][0,:],ApertureCenters[aper][1,:])
            ax2.plot(np.arange(apwindow[0],apwindow[1]+1), mean_profile)
            plt.show()

    return ApertureCenters


def Get_ApertureTraceFunction(ApertureCenters,deg=4):
    """ Returns dictionary of aperture trace functions (degree = deg) 
        based on the best fit of points in ApertureCenters """
    ApertureTraceFuncDic = {}
    for aper in ApertureCenters:
        # fit Chebyshev polynomial to the data to obtain cheb coeffs
        cc = np.polynomial.chebyshev.chebfit(ApertureCenters[aper][0,:], 
                                              ApertureCenters[aper][1,:], deg)
        ApertureTraceFuncDic[aper] = partial(np.polynomial.chebyshev.chebval, c= cc) 

    return ApertureTraceFuncDic

def Get_SlitShearFunction(ApertureCenters):
    """ Returns dictionary of the Dispersion direction shear coefficent for the slit """
    ApertureSlitShearFuncDic = {}
    for aper in ApertureCenters:
        ApertureSlitShearFuncDic[aper] = lambda x : x*0 # TODO: update with a proper function
 
    return ApertureSlitShearFuncDic

def RectifyCurvedApertures(SpectrumFile,Twidth,
                           ApertureTraceFuncDic,SlitShearFuncDic,
                           dispersion_Xaxis = True):
    """ Returns dictionary of rectified aperture image of width Twidth.
    Input:
      SpectrumFile: Spectrum image to rectify
      Twidth: (tuple) Range which gives the width of the strip of aperture to be rectified
      ApertureTraceFuncDic: Dictionary of function which traces the aperture
      SlitShearFuncDic: Dictionary of function which gives slit's shear
    Returns:
      RectifiedApertureDic : Dictionary of the Rectified strips of aperture
    """
    if isinstance(SpectrumFile,str):
        ImageArray = fits.getdata(SpectrumFile)
    else:
        ImageArray = SpectrumFile
    # Transpose the image if dispersion is not along X axis
    if not dispersion_Xaxis:
        ImageArray = ImageArray.T
 
    DispCoords = np.arange(ImageArray.shape[1])
    XDCoordsDelta = np.arange(Twidth[0],Twidth[1]+1)

    RectifiedApertureDic = {}
    for aper in ApertureTraceFuncDic:
        # First create the pixel mapping
        XDCoordsCenter = ApertureTraceFuncDic[aper](DispCoords)
        DCoords = DispCoords[:,np.newaxis] + SlitShearFuncDic[aper](DispCoords)[:,np.newaxis]*XDCoordsDelta[np.newaxis,:]
        XDCoords = XDCoordsCenter[:,np.newaxis] + XDCoordsDelta[np.newaxis,:]
        
        # Now use Bandlimited inteprolation to find values at the mapped pixels.
        # Initiate the interpolator
        BL2D = BandLimited2DInterpolator(filter_sizeX = 23,filter_sizeY = 23, kaiserBX=6, kaiserBY=6)
        # We need only the relevent strip of image data in Dispersion direction
        XDstart = max(0, int(np.min(XDCoords) - 20))
        XDend = min(ImageArray.shape[0], int(np.max(XDCoords) + 20))
        ImageArrayStrip = ImageArray[XDstart:XDend,:]
        NewXCoo = XDCoords.flatten() - XDstart
        NewYCoo = DCoords.flatten()
        Interpolated_values = BL2D.interpolate(NewXCoo,NewYCoo,ImageArrayStrip)        
        # TODO: Multiply by the area ratio of the transformation for flux preservation
        
        # Reshape back the flattned values to 2D array
        RectifiedApertureDic[aper] = Interpolated_values.reshape(XDCoords.shape)

    return RectifiedApertureDic


def main():
    """ Extracts 2D spectrum image into 1D spectrum """
    FlatFile = '/media/diskusers/ExtHDisk/joe/HPFSimulation/Kyle/input_data.fits'
    SpectrumFile = '/media/diskusers/ExtHDisk/joe/HPFSimulation/Kyle/input_data.fits'
    # Adaptively threshold the Flat to obtain the aperture masks
    FlatThresholdM = ImageThreshold(FlatFile,bsize=401,offset=0,minarea=1000, ShowPlot=True)
    # Label the apertures
    ApertureLabel = LabelDisjointRegions(FlatThresholdM,DirectlyEnterRelabel= True)
    # Trace the center of the pertures
    ApertureCenters = FitApertureCenters(FlatFile,ApertureLabel,apwindow=(-7,+7),
                                         dispersion_Xaxis = True, ShowPlot=True)
    # Obtain the tracing function of each aperture
    ApertureTraceFuncDic = Get_ApertureTraceFunction(ApertureCenters,deg=4)
    # Obtain the Slit Shear of each order 
    SlitShearFuncDic = Get_SlitShearFunction(ApertureCenters)

    # Get rectified 2D spectrum of each aperture of the spectrum file
    RectifiedSpectrum = RectifyCurvedApertures(SpectrumFile,(-30,+30),
                                               ApertureTraceFuncDic,SlitShearFuncDic,
                                               dispersion_Xaxis = True)
    

# if __name__ == '__main__':
#     main()
