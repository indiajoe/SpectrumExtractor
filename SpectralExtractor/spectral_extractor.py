#!/usr/bin/env python
""" This tool is to extract 1D spectrum form a 2D image """
import argparse
import sys
import os
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
import pickle
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
        ApertureSlitShearFuncDic[aper] = lambda x : x*0 #- 0.0351 # -0.0351 was for the tilt in HE at Oct Cooldown at HET
 
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
        BL2D = BandLimited2DInterpolator(filter_sizeX = 31,filter_sizeY = 31, kaiserBX=5, kaiserBY=5)
        # We need only the relevent strip of image data in Dispersion direction
        XDstart = max(0, int(np.min(XDCoords) - 17))
        XDend = min(ImageArray.shape[0], int(np.max(XDCoords) + 17))
        ImageArrayStrip = ImageArray[XDstart:XDend,:]
        NewXCoo = XDCoords.flatten() - XDstart
        NewYCoo = DCoords.flatten()
        Interpolated_values = BL2D.interpolate(NewXCoo,NewYCoo,ImageArrayStrip)        
        # TODO: Multiply by the area ratio of the transformation for flux preservation
        
        # Reshape back the flattned values to 2D array
        RectifiedApertureDic[aper] = Interpolated_values.reshape(XDCoords.shape)
    return RectifiedApertureDic


def SumApertures(RectifiedApertureDic, apwindow=(None,None), apertures=None, ShowPlot=False):
    """ Returns the sum of each rectified aperture inside the apwindow .
    If lower bound or upper bound of apwindow is None, sumation will be from 
    the begining or till the end of the XD axis of aperture array respectively"""
    ApertureSumDic = {}
    if apertures is None : 
        apertures = RectifiedApertureDic.keys()
    
    for aper in apertures:
        Llimit = -int(RectifiedApertureDic[aper].shape[1]/2.0) if apwindow[0] is None else apwindow[0]
        Ulimit = int(RectifiedApertureDic[aper].shape[1]/2.0) if apwindow[1] is None else apwindow[1]
        Lindx = int(RectifiedApertureDic[aper].shape[1]/2.0) + Llimit
        Uindx = int(RectifiedApertureDic[aper].shape[1]/2.0) + Ulimit
        

        ApertureSumDic[aper] = np.sum(RectifiedApertureDic[aper][:,Lindx:Uindx],axis=1)
        if ShowPlot:
            plt.plot(ApertureSumDic[aper])
            plt.ylabel('Sum of Counts')
            plt.xlabel('pixels')
            plt.title('Aperture : {0}'.format(aper))
            plt.show()

    return ApertureSumDic

def fix_badpixels(SpectrumImage,BPMask,replace_with_nan=False):
    """ Applies the bad mask (BPMask) on to the SpectrumImage to fix it """
    Mask = BPMask == 0
    if replace_with_nan:
        # No fixing to do. Just replace the pixels with np.nan
        SpectrumImage[Mask] = np.nan
    else:
        # Interpolate the values
        labeled_array, num_regions = ndimage.label(Mask)
        for region in range(1,num_regions+1):
            Xc,Yc = np.where(labeled_array==region)
            # Cut of a small tile with 4 pix border to speed up the interpolation
            TileX, TileY = np.meshgrid(range(np.min(Xc)-4,np.max(Xc)+4),range(np.min(Yc)-4,np.max(Yc)+4),indexing='ij')
            FullCoordsSet = set(zip(TileX.flatten(), TileY.flatten()))
            GoodPixelXY = np.array(list(FullCoordsSet - set(zip(*np.where(labeled_array != 0)))))
            # Create an interpolator to interpolate
            LinInterp = interp.LinearNDInterpolator(GoodPixelXY,SpectrumImage[GoodPixelXY[:,0],GoodPixelXY[:,1]])
            # Fill the values with the interpolated pixels
            SpectrumImage[Xc,Yc] = LinInterp(np.array([Xc,Yc]).T)

    return SpectrumImage
            

def WriteFitsFileSpectrum(FluxSpectrumDic, outfilename, fitsheader=None):
    """ Writes the FluxSpectrumDic into fits image with filename outfilename 
    with fitsheader as header."""
    # First create a 2D array to hold all apertures of the spectrum
    Spectrumarray = np.array([FluxSpectrumDic[i] for i in sorted(FluxSpectrumDic.keys())])
    hdu = fits.PrimaryHDU(Spectrumarray,header=fitsheader)
    hdu.writeto(outfilename)
    return outfilename


def parse_args():
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Spectral Extraction Tool")
    parser.add_argument('SpectrumFile', type=str,
                        help="The 2D Spectrum image fits file")
    # parser.add_argument('ConfigFile', type=str,
    #                     help="Configuration file which contains settings for extraction")
    parser.add_argument('--FlatNFile', type=str, default=None,
                        help="Normalized Flat file to be used for correcting pixel to pixel variation")
    parser.add_argument('--BadPixMask', type=str, default=None,
                        help="Bad Pixel Mask file to be used for fixing bad pixels")
    parser.add_argument('--ContinuumFile', type=str, default=None,
                        help="Continuum flat source to be used for aperture extraction")
    parser.add_argument('--ApertureLabel', type=str, default=None,
                        help="Array of labels for the aperture trace regions")
    parser.add_argument('OutputFile', type=str, 
                        help="Output filename to write extracted spectrum")
    args = parser.parse_args()
    return args

def main():
    """ Extracts 2D spectrum image into 1D spectrum """
    args = parse_args()
    # Config = create_configdict_from_file(args.ConfigFile)

    SpectrumFile = args.SpectrumFile
    ContinuumFile = args.ContinuumFile
    NFlatFile = args.FlatNFile
    BadPixMaskFile = args.BadPixMask
    OutputFile = args.OutputFile
    

    if os.path.isfile(OutputFile):
        print('WARNING: Output file {0} already exist'.format(OutputFile))
        print('Skipping this image extraction..')
        sys.exit(1)

    print('Extracting {0}..'.format(SpectrumFile))
    fheader = fits.getheader(SpectrumFile)

    if os.path.isfile(str(args.ApertureLabel)):
        ApertureLabel = np.load(args.ApertureLabel)
    else:
        # Adaptively threshold the ContinuumFile Flat to obtain the aperture masks
        CFlatThresholdM = ImageThreshold(ContinuumFile,bsize=51,offset=0,minarea=1000, ShowPlot=True)
        # Label the apertures
        ApertureLabel = LabelDisjointRegions(CFlatThresholdM,DirectlyEnterRelabel= True)
        # Save the aperture label if a non existing filename was provided as input
        if isinstance(args.ApertureLabel,str):
            np.save(args.ApertureLabel,ApertureLabel)

    ApertureTraceFilename = ContinuumFile+'_trace.pkl'
    if os.path.isfile(ApertureTraceFilename):
        print('Loading existing trace coordinates {0}'.format(ApertureTraceFilename))
        ApertureCenters = pickle.load(open(ApertureTraceFilename,'rb'))
    else:
        # Trace the center of the apertures
        ApertureCenters = FitApertureCenters(ContinuumFile,ApertureLabel,apwindow=(-7,+7),
                                             dispersion_Xaxis = True, ShowPlot=False)
        #Save for future
        with open(ApertureTraceFilename,'wb') as tracepickle:
            pickle.dump(ApertureCenters,tracepickle)

    fheader['APFILE'] = (os.path.basename(ApertureTraceFilename), 'Aperture Trace Filename')

    # Obtain the tracing function of each aperture
    ApertureTraceFuncDic = Get_ApertureTraceFunction(ApertureCenters,deg=4)
    # Obtain the Slit Shear of each order 
    SlitShearFuncDic = Get_SlitShearFunction(ApertureCenters)

    # Load the spectrum array 
    SpectrumImage = fits.getdata(SpectrumFile)
    # Apply flat correction for pixel to pixel variation
    if NFlatFile is not None:
        print('Doing Flat correction..')
        NFlat = fits.getdata(NFlatFile)
        SpectrumImage /= NFlat
        fheader['FLATFIL'] = (os.path.basename(NFlatFile), 'Flat Filename')
        fheader['HISTORY'] = 'Flat fielded the image'

    # Apply bad pixel correction
    if BadPixMaskFile is not None:
        print('Doing Bad pixel correction..')
        BPMask = fits.getdata(BadPixMaskFile)
        SpectrumImage = fix_badpixels(SpectrumImage,BPMask)
        fheader['BPMASK'] = (os.path.basename(BadPixMaskFile), 'Bad Pixel Mask File')
        fheader['HISTORY'] = 'Fixed the bad pixels in the image'
        
    # Get rectified 2D spectrum of each aperture of the spectrum file
    RectifiedSpectrum = RectifyCurvedApertures(SpectrumImage,(-8,+8),
                                               ApertureTraceFuncDic,SlitShearFuncDic,
                                               dispersion_Xaxis = True)
    
    # Sum the flux in XD direction of slit
    SumApFluxSpectrum = SumApertures(RectifiedSpectrum, apwindow=(-6,6), ShowPlot=False)

    # Write the extracted spectrum to output fits file
    fheader['HISTORY'] = 'Extracted spectrum to 1D'
    _ = WriteFitsFileSpectrum(SumApFluxSpectrum, OutputFile, fitsheader=fheader)
    print('Extracted {0} => {1} output file'.format(SpectrumFile,OutputFile))

if __name__ == '__main__':
    main()
