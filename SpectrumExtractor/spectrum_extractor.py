#!/usr/bin/env python
""" This tool is to extract 1D spectrum form a 2D image """
import argparse
import sys
import os
import logging
import numpy as np
import re
from astropy.io import fits
from astropy.stats import mad_std, biweight_location
from astropy.visualization import MinMaxInterval, SqrtStretch, PercentileInterval, ImageNormalize
import matplotlib.pyplot as plt
from skimage import filters
from skimage import morphology
import scipy
from scipy import ndimage, signal
import scipy.interpolate as interp
import scipy.optimize as optimize
import pickle
from .cosmic_ray_utils import threshold_in_laplace_space

from WavelengthCalibrationTool.recalibrate import (ReCalibrateDispersionSolution,
                                                   scale_interval_m1top1,
                                                   calculate_pixshift_with_phase_cross_correlation)
from WavelengthCalibrationTool.utils import calculate_cov_matrix_fromscipylsq

try:
    from ccdproc import cosmicray_lacosmic
except ImportError:
    logging.warning('Failed to import ccdproc module for cosmicray_lacosmic. DoCosmicRayClean option will not work without that.')

try:
    from RVEstimator.interpolators import BandLimited2DInterpolator
except ImportError:
    logging.warning('Failed to import RVEstimator module for BandLimited2D Interpolator. Rectification option will not work without that.')

try:
    from functools32 import partial
    import ConfigParser
except ImportError:
    from functools import partial
    import configparser as  ConfigParser


def ImageThreshold(imgfile,bsize=401,offset=0, minarea=0,ShowPlot=False):
    """ Returns adaptive thresholded image mask """
    imgArray = fits.getdata(imgfile)
    # Adaptive thresholding..
    ThresholdedMask = imgArray > filters.threshold_local(imgArray, bsize,offset=offset)
    if minarea:
        logging.info('Regions of area less than {0} are discarded'.format(minarea))
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
            uinput = str(input('Enter New Region label  (default: {0}): '.format(sugg))).strip()
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
        # Show the current labels of the regions
        # for label in np.unique(labeled_array):
        #     Xcenter = np.mean(np.where(labeled_array == label)[0])
        #     Ycenter = np.mean(np.where(labeled_array == label)[1])
        #     plt.text(Xcenter,Ycenter,str(label))
        plt.colorbar()
        plt.show(block=False)
        NewLabeledArray = np.zeros(labeled_array.shape)

        print('Create a file with the label renames to be executed.')
        print('File format should be 2 columns :  OldLabel NewLabel')
        print('Note: Only assigned labels will be kept..')
        uinputfile = input('Enter the filename :').strip()
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

def CreateApertureLabelByThresholding(ContinuumFile,BadPixMask=None,bsize=51,offset=0,minarea=2000, ShowPlot=True,DirectlyEnterRelabel= True):
    """ Creates an Aperture Label by Thresholding 2D image """
    # Adaptively threshold the ContinuumFile Flat to obtain the aperture masks
    CFlatThresholdM = ImageThreshold(ContinuumFile,bsize=bsize,offset=offset,minarea=minarea, ShowPlot=ShowPlot)

    if BadPixMask is not None:
        BPMask = np.load(BadPixMask) # fits.getdata(Config['BadPixMask'])
        # Remove bad pixels from Thresholded image
        CFlatThresholdM[~BPMask] = False
    # Label the apertures
    ApertureLabel = LabelDisjointRegions(CFlatThresholdM,DirectlyEnterRelabel=DirectlyEnterRelabel)
    return ApertureLabel


def NearestIndx(array,val):
    """ Returns the nearest index for val in sorted array"""
    return np.argmin(np.abs(array-val))


def RefineCentriodsInSignal(data,initial_centroids,Hwindow,Xpixels=None,profile=None,sigma=None):
    """ Returns refined centroids using data inside half width Hwindow """
    if Xpixels is None:
        Xpixels = np.arange(len(data))
    if sigma is None:
        sigma = np.sqrt(np.abs(data))
    if profile is None:
        # We are going to calculate the first moment, so set all negative data values to zero.
        datacopy = data.copy() # Preserve the input data from getting changed.
        datacopy[data<0] = 0
        data = datacopy

    NewCentroidList = []
    NewCentroidErrorList = []

    for i in range(len(initial_centroids)):
        icent = initial_centroids[i]
        indx = NearestIndx(Xpixels,icent)
        if profile is None:
            new_centroid = np.sum(data[indx-Hwindow:indx+Hwindow]*Xpixels[indx-Hwindow:indx+Hwindow])/np.sum(data[indx-Hwindow:indx+Hwindow])
            new_centroid_err = np.sqrt(np.sum((sigma[indx-Hwindow:indx+Hwindow]*(Xpixels[indx-Hwindow:indx+Hwindow]-np.mean(Xpixels[indx-Hwindow:indx+Hwindow])))**2)/np.sum(data[indx-Hwindow:indx+Hwindow])**2)
        else:
            RefSpectrum = np.array([data[indx-Hwindow:indx+Hwindow],Xpixels[indx-Hwindow:indx+Hwindow]]).T
            if isinstance(profile[0],list): # if profile is a list of list
                newX, popt, pconv = ReCalibrateDispersionSolution(profile[i],RefSpectrum,method='p0',sigma=sigma[indx-Hwindow:indx+Hwindow],cov=True)
            else:
                newX, popt, pconv = ReCalibrateDispersionSolution(profile,RefSpectrum,method='p0',sigma=sigma[indx-Hwindow:indx+Hwindow],cov=True)
            # scale the shifts from -1 to 1 domain to real pixel domain
            new_centroid = scale_interval_m1top1(popt[1], a=min(RefSpectrum[:,0]),b=max(RefSpectrum[:,0]), inverse_scale=True)
            new_centroid_err = scale_interval_m1top1(np.sqrt(pconv[1,1]), a=min(RefSpectrum[:,0]),b=max(RefSpectrum[:,0]), inverse_scale=True)

        NewCentroidErrorList.append(new_centroid_err)
        NewCentroidList.append(new_centroid)
    return NewCentroidList, NewCentroidErrorList

def CreateApertureLabelByXDFitting(ContinuumFile,BadPixMask=None,startLoc=None,avgHWindow=21,TraceHWidth=5,trace_fit_deg=4,
                                   extrapolate_thresh=0.4,extrapolate_order=2,
                                   dispersion_Xaxis=True,ShowPlot=True,return_trace=False):
    """Creates the Aperture Trace labels by shifting and fitting profiles in Cross dispersion columns """
    if isinstance(ContinuumFile  ,str):
        ContinuumFile = fits.getdata(ContinuumFile)
        if not dispersion_Xaxis:
            ContinuumFile = ContinuumFile.T
    if BadPixMask is not None:
        BPMask = fits.getdata(BadPixMask) if BadPixMask[-5:] == '.fits' else np.load(BadPixMask)
        if not dispersion_Xaxis:
            BPMask = BPMask.T

    if startLoc is None:
        startLoc = ContinuumFile.shape[1]//2
    # Starting labelling Reference XD cut data
    RefXD = np.nanmedian(ContinuumFile[:,startLoc-avgHWindow:startLoc+avgHWindow],axis=1)
    Refpixels = np.arange(len(RefXD))
    Bkg = signal.order_filter(RefXD,domain=[True]*TraceHWidth*5,rank=int(TraceHWidth*5/10))
    Flux = np.abs(RefXD -Bkg)
    ThreshMask = RefXD > (Bkg + np.abs(mad_std(Flux))*6)
    labeled_array, num_traces = ndimage.label(ThreshMask)
    logging.info('Detected {0} order traces'.format(num_traces))
    LabelList = []
    XDCenterList = []
    for label in np.sort(np.unique(labeled_array)):
        if label == 0: # Skip 0 label
            continue
        M = labeled_array==label
        centerpix = np.sum(Flux[M]*Refpixels[M])/np.sum(Flux[M])
        LabelList.append(label)
        XDCenterList.append(centerpix)
    # Refine again using the TraceHWidth window
    XDCenterList, XDCenterList_err = RefineCentriodsInSignal(Flux,XDCenterList,TraceHWidth,Xpixels=Refpixels)

    if ShowPlot:
        plt.plot(Refpixels,Flux)
        plt.plot(Refpixels,labeled_array)
        plt.plot(XDCenterList,LabelList,'.')
        plt.xlabel('XD pixels')
        plt.ylabel('Trace labels')
        plt.show()

    print('Trace_number     PixelCoord   PixelError')
    print('\n'.join(['{0}  {1}  {2}'.format(l,p,e) for l,p,e in zip(LabelList,XDCenterList,XDCenterList_err)]))

    print('Create a file with the label names and pixel coords to use.')
    print('File format should be 3 columns :  OrderLabel PixelCoords PixelError')
    print('Note: Trace labels should start with 1 and not zero')
    uinputfile = input('Enter the filename :').strip()
    if uinputfile: #User provided input
        with open(uinputfile,'r') as labelchangefile:
            LabelList = []
            XDCenterList = []
            for entry in labelchangefile:
                if entry[0]=='#':
                    continue
                entry = entry.rstrip().split()
                if len(entry)<2:
                    continue
                LabelList.append(int(entry[0]))
                XDCenterList.append(float(entry[1]))
                XDCenterList_err.append(float(entry[2]))
        if ShowPlot:
            print('New labelled traces')
            plt.plot(Refpixels,Flux)
            plt.plot(Refpixels,labeled_array)
            plt.plot(XDCenterList,LabelList,'.')
            plt.xlabel('XD pixels')
            plt.ylabel('New Trace labels')
            plt.show()

    else:
        print('No input given. No relabelling done..')

    # Create a index list sorted by the pixel error as a proxy for brightness and goodness of the line to use for extrapolation
    SortedErrorIndices = np.argsort(XDCenterList_err)

    # Now start fitting the XD cut plot across the dispersion

    # Create a dictionary to save dcoordinates of each order
    FullCoorindateOfTraceDic = {o:[[d],[xd],[xde]] for o,d,xd,xde in zip(LabelList,[startLoc]*len(LabelList),XDCenterList,XDCenterList_err)}

    # First step to higher pixels from startLoc position and then step to lower positions
    for stepDLoc in [avgHWindow,-1*avgHWindow]:
        newDLoc = startLoc + max(1,np.abs(stepDLoc)//2)*np.sign(stepDLoc)
        newpixels = np.arange(len(Flux))
        newRefFlux = np.vstack([newpixels,Flux]).T
        newRefXDCenterList = XDCenterList
        while (newDLoc < ContinuumFile.shape[1]-avgHWindow) and (newDLoc > avgHWindow):
            newXD = np.nanmedian(ContinuumFile[:,newDLoc-avgHWindow:newDLoc+avgHWindow],axis=1)
            newBkg = signal.order_filter(newXD,domain=[True]*TraceHWidth*5,rank=int(TraceHWidth*5/10))
            newFlux = np.abs(newXD -newBkg)
            SigmaArrayWt = np.sqrt(np.abs(newFlux))
            try:
                shifted_pixels,fitted_driftp = ReCalibrateDispersionSolution(newFlux,
                                                                             newRefFlux,method='p1',
                                                                             sigma = SigmaArrayWt)
            except (RuntimeError,ValueError) as e:
                logging.warning(e)
                logging.warning('Failed fitting.. Skipping {0} pixel position'.format(newDLoc))
            else:
                # Calculate the new pixel coordinates of previous centroids
                newXDCenterList = [NearestIndx(shifted_pixels,icent) for icent in newRefXDCenterList]
                newXDCenterList, newXDCenterList_err  = RefineCentriodsInSignal(newFlux,newXDCenterList,TraceHWidth,Xpixels=newpixels)
                # Make sure there is atleast extrapolate_order trace which do not need to be interpolated to for this scheme to work
                NoOfGoodTraceFits = np.sum(np.array(newXDCenterList_err) < extrapolate_thresh)
                if NoOfGoodTraceFits > extrapolate_order :
                    # Identify poorly constrained centers and extrapolate from the nearby good points.
                    PositionDiffArray = np.array(newXDCenterList)-np.array(XDCenterList)
                    newSortedErrorIndices = np.argsort(newXDCenterList_err)
                    for i in range(len(newSortedErrorIndices)):
                        ic = newSortedErrorIndices[i]
                        if newXDCenterList_err[ic] < extrapolate_thresh:
                            continue
                        else:
                            # Identify nearby good points better than this bad point
                            GoodpointsSuperArray = np.array(newSortedErrorIndices[:i])
                            extrapolate_points_tofit = extrapolate_order *3
                            # Find nearest extrapolate_points_tofit points fron the GoodpointsSuperList
                            NearestGoodPoints = GoodpointsSuperArray[np.argsort(np.abs(GoodpointsSuperArray-ic))[:extrapolate_points_tofit]]
                            logging.debug('Identified Traces {0} to extrapolate for trace {1} with error {2} at pixel pos {3}'.format(NearestGoodPoints,ic,newXDCenterList_err[ic],newDLoc))
                            # Fit the polynomial to extrapolate to obtain ic trace location
                            extrp_p = np.polyfit(NearestGoodPoints,PositionDiffArray[NearestGoodPoints],extrapolate_order)
                            new_pos_diff = np.polyval(extrp_p,ic)
                            PositionDiffArray[ic] = new_pos_diff
                            newXDCenterList[ic] = XDCenterList[ic] + new_pos_diff
                    # update the Dictionary
                    for i,o in enumerate(LabelList):
                        if (0 < newXDCenterList[i] < ContinuumFile.shape[0]) and (newXDCenterList_err[i] < 0.5):
                            FullCoorindateOfTraceDic[o][0].append(newDLoc)
                            FullCoorindateOfTraceDic[o][1].append(newXDCenterList[i])
                            FullCoorindateOfTraceDic[o][2].append(max(0.05,newXDCenterList_err[i])) # min error is set to 0.05

                    #Change the Reference to the new DLoc position
                    newRefFlux = np.vstack([newpixels,newFlux]).T
                    newRefXDCenterList = newXDCenterList
                else:
                    logging.debug('Skipping pixel pos {0} since number of good traces {1} < extrapolation poly order {2}'.format(newDLoc,NoOfGoodTraceFits, extrapolate_order))
            finally:
                newDLoc = newDLoc + max(1,np.abs(stepDLoc)//2)*np.sign(stepDLoc)

    # Finally fit a trace function for each order and create an Aperture Label array
    ApertureLabel = np.zeros(ContinuumFile.shape)
    # First conver the dictionary values to a numpy array
    for o in LabelList:
        FullCoorindateOfTraceDic[o] = np.array(FullCoorindateOfTraceDic[o])
    pix_scale_function = partial(scale_interval_m1top1,a=0,b=ContinuumFile.shape[1])
    ApertureTraceFuncDic = Get_ApertureTraceFunction(FullCoorindateOfTraceDic,deg=trace_fit_deg,domain_scale_function=pix_scale_function)
    # Now loop through each trace for setting the label
    for o in reversed(sorted(LabelList)):
        boundinside = partial(boundvalue,ll=0,ul=ApertureLabel.shape[0])
        for j in np.arange(ApertureLabel.shape[1]):
            mini,maxi = int(np.rint(boundinside(ApertureTraceFuncDic[o](j)-TraceHWidth))), int(np.rint(boundinside(ApertureTraceFuncDic[o](j)+TraceHWidth+1)))
            ApertureLabel[mini:maxi,j] = o

    if ShowPlot:
        plt.imshow(np.ma.array(ApertureLabel,mask=ApertureLabel==0),cmap='hsv')
        norm = ImageNormalize(ContinuumFile, interval=PercentileInterval(95.),stretch=SqrtStretch())
        plt.imshow(ContinuumFile,norm=norm,alpha=0.5)
        plt.colorbar()
        for o in LabelList:
            plt.plot(FullCoorindateOfTraceDic[o][0],FullCoorindateOfTraceDic[o][1],marker='.',alpha=0.5,color='k')
        plt.show()
    if return_trace:
        return ApertureLabel, FullCoorindateOfTraceDic
    else:
        return ApertureLabel



def errorfuncProfileFit(p,psf=None,xdata=None, ydata=None):
    """ Error function to minimise the profile psf(xdata) fit on to ydata """
    return p[0]*psf(xdata-p[1]) - ydata

def boundvalue(x,ll,ul):
    """ Returns x if ll<= x <= ul. Else the limit values """
    if ll <= x <= ul:
        return x
    elif x < ll:
        return ll
    elif x > ul :
        return ul

def FitApertureCenters(SpectrumFile,ApertureLabel,apertures=None,
                           apwindow=(-7,+7),dispersion_Xaxis = True, ShowPlot=False):
    """ Fits the center of the apertures in the spectrum"""
    ApertureCenters = {}
    logging.info('Extracting Aperture Centers')

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
        logging.info('Aperture : {0}'.format(aper))
        # Crude center of the apperture along dispersion
        aperCenter = np.ma.mean(np.ma.array(np.indices(ApertureLabel.shape)[0],
                                            mask=~(ApertureLabel==aper)),axis=0)
        # Masked values in the aperCenter means no pixel in the threshold map!

        # Shift and combine to create a PSF for the aperture
        # Pixels along the dispersion direction
        dpix = np.arange(len(aperCenter))[~aperCenter.mask]
        xdpixCenter = aperCenter[~aperCenter.mask].data
        xdapL2Upix = np.rint(xdpixCenter[:,np.newaxis]+\
                             np.arange(apwindow[0],apwindow[1]+1) ).astype(int) # round of to nearest integer
        # Extrapolate the coordinates at edges of array (for anyway useless orders) at the edge of detector
        xdapL2Upix[xdapL2Upix >= ImageArray.shape[0]] = ImageArray.shape[0]-1
        xdapL2Upix[xdapL2Upix < 0] = 0

        Rectifiedarray = ImageArray[(xdapL2Upix,
                                     np.repeat(dpix[:,np.newaxis],xdapL2Upix.shape[1],axis=1))]
        weights_foravg = np.sum(Rectifiedarray,axis=1)
        weights_foravg = np.clip(weights_foravg,0,np.percentile(weights_foravg,98))  # remove -ve weights and clip 2 percentile max points
        weights_foravg[~np.isfinite(weights_foravg)] = 0 # Set any non finite pixel weights to 0
        masked_Rectifiedarray = np.ma.array(Rectifiedarray,mask=~np.isfinite(Rectifiedarray)) # Mask out any non finite pixels
        mean_profile = np.ma.average(masked_Rectifiedarray,axis=0,weights=weights_foravg)  # weighted propotional to flux level
        # PSF interpolated function
        psf = interp.InterpolatedUnivariateSpline(np.arange(apwindow[0],apwindow[1]+1), mean_profile)

        # Loop along the dispersion pixels
        # Fit the profile for each column to obtian the accurate centroid in each column
        CenterXDcoo = []
        CenterXDcooErr = []
        ampl = 1
        for d,xd,flux in zip(dpix,xdapL2Upix,Rectifiedarray):
            m = np.isfinite(flux) # good pixel mask
            # initial estimate
            p0 = [ampl,xd[len(xd)//2]]
            fitoutput = optimize.least_squares(errorfuncProfileFit, p0,
                                               bounds=([0,np.min(xd)],[np.inf,np.max(xd)]),
                                               args=(psf,xd[m],flux[m]))
            p = fitoutput.x
            pcov = calculate_cov_matrix_fromscipylsq(fitoutput,absolute_sigma=False)
            # p,ier = optimize.leastsq(errorfuncProfileFit, p0, args=(psf,xd,flux))
            # CenterXDcoo.append(boundvalue(p[1],np.min(xd),np.max(xd))) # use the boundry values incase the fitted p[1] is outside the bounds
            CenterXDcoo.append(p[1])
            CenterXDcooErr.append(np.sqrt(pcov[1,1]))
            ampl = p[0] # update with the lastest amplitude estimate

        # Clip the errors less than 0.05 pixel to 0.05
        ApertureCenters[aper] = np.array([dpix,CenterXDcoo,np.clip(CenterXDcooErr,0.05,None)])
        if ShowPlot:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax1.errorbar(ApertureCenters[aper][0,:],ApertureCenters[aper][1,:],yerr=ApertureCenters[aper][2,:])
            ax1.set_ylim((np.min(ApertureCenters[aper][1,:])-5,np.max(ApertureCenters[aper][1,:])+5))
            ax2.plot(np.arange(apwindow[0],apwindow[1]+1), mean_profile)
            plt.show()

    return ApertureCenters

def function_eval_afterdomainscaling(x,func,domain_scale_function=None,**kwargs):
    if domain_scale_function is None:
        return func(x,**kargs)
    else:
        return func(domain_scale_function(x),**kwargs)

def Get_ApertureTraceFunction(ApertureCenters,deg=4,return_coeff=False,domain_scale_function=None):
    """ Returns dictionary of aperture trace functions (degree = deg)
        based on the best fit of points in ApertureCenters
        if return_coeff is True, returns a dictionary of the coefficents as well"""
    if domain_scale_function is None:
        domain_scale_function = lambda x: x
    ApertureTraceFuncDic = {}
    ApertureTraceCoeffDic = {}
    for aper in ApertureCenters:
        try:
            weights = 1/ApertureCenters[aper][2,:]
        except IndexError:  # No weights provided
            weights = None # Ignore to support legacy files
        else:
            weights[~np.isfinite(weights)] = 0
        # fit Chebyshev polynomial to the data to obtain cheb coeffs
        cc = np.polynomial.chebyshev.chebfit(domain_scale_function(ApertureCenters[aper][0,:]),
                                             ApertureCenters[aper][1,:], deg,
                                             w=weights)
        ApertureTraceFuncDic[aper] = partial(function_eval_afterdomainscaling,
                                             func=np.polynomial.chebyshev.chebval,
                                             domain_scale_function=domain_scale_function,
                                             c=cc)
        ApertureTraceCoeffDic[aper] = cc
    if return_coeff:
        return ApertureTraceFuncDic, ApertureTraceCoeffDic
    else:
        return ApertureTraceFuncDic

def get_aperture_trace_function_fromChebcoeff(ApertureCentersCoeff,domain_scale_function=None):
    """ Returns dictionary of aperture trace functions 
        based on the polynomial coeffs in ApertureCenters """
    ApertureTraceFuncDic = {}
    for aper in ApertureCentersCoeff:
        # Use chebyshev polynomimal with domain scaling
        ApertureTraceFuncDic[aper] = partial(function_eval_afterdomainscaling,
                                             func=np.polynomial.chebyshev.chebval,
                                             domain_scale_function=domain_scale_function,
                                             c=ApertureCentersCoeff[aper])
    return ApertureTraceFuncDic



def get_aperture_trace_function_fromcoeff(ApertureCentersCoeff):
    """ Returns dictionary of aperture trace functions 
        based on the polynomial coeffs in ApertureCenters """
    ApertureTraceFuncDic = {}
    for aper in ApertureCentersCoeff:
        # currently use the standard polynomial
        ApertureTraceFuncDic[aper] = partial(np.polynomial.polynomial.polyval, c= ApertureCentersCoeff[aper]) 

    return ApertureTraceFuncDic
######################################################

def Get_SlitShearFunction(ApertureCenters):
    """ Returns dictionary of the Dispersion direction shear coefficent for the slit """
    ApertureSlitShearFuncDic = {}
    for aper in ApertureCenters:
        ApertureSlitShearFuncDic[aper] = lambda x : x*0 #- 0.0351 # -0.0351 was for the tilt in HE at Oct Cooldown at HET

    return ApertureSlitShearFuncDic

def CalculateShiftInXD(SpectrumImage, RefImage=None, XDshiftmodel='p0', DWindowToUse=None, StripWidth=50,
                       Apodize=True, bkg_medianfilt=False,dispersion_Xaxis=True,ShowPlot=False):
    """ Calculates the avg shift in XD to match SpectrumImage to RefImage
    Returns Avg_XD_shift coeffiencts in the domain the XD pixels are scaled to -1 to 1 """
    if isinstance(RefImage,str):
        RefImage = fits.getdata(RefImage)
    else:
        RefImage = RefImage

    # Transpose the image if dispersion is not along X axis
    if not dispersion_Xaxis:
        SpectrumImage = SpectrumImage.T
        RefImage = RefImage.T

    ApodizingWindow = scipy.signal.windows.hamming(StripWidth)  # We use HAmming window to apodise the edges of each stripe

    if DWindowToUse is None:
        DWindowToUse = (1,-1)
    NoOfXDstripes = int(SpectrumImage[:,DWindowToUse[0]:DWindowToUse[1]].shape[1]/StripWidth)

    XDShiftList = []
    for i,(XDSliceSpec,XDSliceRef) in enumerate(zip(np.split(SpectrumImage[:,DWindowToUse[0]:DWindowToUse[0]+NoOfXDstripes*StripWidth],
                                                           NoOfXDstripes,axis=1),
                                                  np.split(RefImage[:,DWindowToUse[0]:DWindowToUse[0]+NoOfXDstripes*StripWidth],
                                                           NoOfXDstripes,axis=1))):
        SumApFluxSpectrum = np.sum(XDSliceSpec*ApodizingWindow,axis=1)
        SumApRefSpectrum = np.sum(XDSliceRef*ApodizingWindow,axis=1)
        SigmaArrayWt = np.sqrt(np.abs(SumApFluxSpectrum))
        SigmaArrayWt = np.clip(SigmaArrayWt,np.percentile(SigmaArrayWt,95),None) # put lower limit on sigma at 95 percentile to avoid zero sigmas
        if bkg_medianfilt: # Subtract a median filtered bkg in the XD slice
            SumApFluxSpectrum -= signal.medfilt(SumApFluxSpectrum,kernel_size=bkg_medianfilt)
            SumApRefSpectrum -= signal.medfilt(SumApRefSpectrum,kernel_size=bkg_medianfilt)

        newRefFlux = np.vstack([np.arange(len(SumApRefSpectrum)),SumApRefSpectrum]).T
        # Get a quick estimate for the pixel shift
        guess_pshift = calculate_pixshift_with_phase_cross_correlation(SumApFluxSpectrum,SumApRefSpectrum,upsample_factor=10)
        DomainRange = (min(newRefFlux[:,0]), max(newRefFlux[:,0]))
        guess_params = [np.percentile(SumApFluxSpectrum,98)/np.percentile(SumApRefSpectrum,98) ,guess_pshift*2./(DomainRange[1]-DomainRange[0])]
        if int(XDshiftmodel[1:]) == 1:
            guess_params.extend([1])
        elif int(XDshiftmodel[1:]) > 1:
            guess_params.extend([1]+[0]*(int(XDshiftmodel[1:])-1))

        try:
            shifted_pixels, fitted_driftp = ReCalibrateDispersionSolution(SumApFluxSpectrum,newRefFlux,
                                                                          method=XDshiftmodel,
                                                                          sigma = SigmaArrayWt,
                                                                          initial_guess=guess_params)
        except (RuntimeError,ValueError) as e:
            logging.warning(e)
            logging.warning('Failed Refitting aperture at {0} D pixel position'.format(DWindowToUse[0]+i*StripWidth))
        else:
            logging.debug('XD offset fit {0}:{1}'.format(i,fitted_driftp))
            XDShiftList.append(fitted_driftp)
            if ShowPlot:
                plt.plot(newRefFlux[:,0],newRefFlux[:,1]*fitted_driftp[0],color='k',alpha=0.3)
                plt.plot(shifted_pixels,SumApFluxSpectrum,color='g',alpha=0.3)

    Avg_XD_shift = biweight_location(np.array(XDShiftList),axis=0)[1:] #remove the flux scale coeff

    if ShowPlot:
        plt.title('{0}:  {1}'.format(tuple(Avg_XD_shift),tuple(DomainRange)))
        plt.xlabel('XD pixels')
        plt.ylabel('Apodized counts')
        plt.show()

    return Avg_XD_shift, DomainRange

def ApplyXDshiftToApertureCenters(ApertureCenters,Avg_XD_shift,PixDomain):
    """ Returns the shifted Aperture centers after applying the shift """
    ShiftedApertureCenters = {}
    for aper in ApertureCenters:
        mean_y = np.nanmedian(ApertureCenters[aper][1,:])
        scaled_y = scale_interval_m1top1(mean_y, a=PixDomain[0],b=PixDomain[1])
        shifted_scaled_y = np.polynomial.polynomial.polyval(scaled_y, Avg_XD_shift)
        shifted_mean_y = scale_interval_m1top1(shifted_scaled_y,
                                               a=PixDomain[0],b=PixDomain[1],inverse_scale=True)
        y_offset = shifted_mean_y - mean_y
        ShiftedApertureCenters[aper] = np.copy(ApertureCenters[aper])
        ShiftedApertureCenters[aper][1,:] = ShiftedApertureCenters[aper][1,:] - y_offset
    return ShiftedApertureCenters


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

def LagrangeInterpolateArray(newX,X,Y):
    """ Returns an interpolation at newX location using a Lagrange polynomial of order X.shape[0] on X,Y data.
    See formula in https://en.wikipedia.org/wiki/Polynomial_interpolation#Constructing_the_interpolation_polynomial
    Note newX should be an array of size X.shape[1] . (Or float if X and Y are 1D arrays)
    ie, X and Y will also be a 2D array. And the funtion will return the newX values for each of the X[:,i],Y[:,i] pairs.
 """
    xmX_j = newX-X
    TermsList = []
    for i in range(X.shape[0]):
        xmX_jmi = np.delete(xmX_j,i,axis=0)  # x-X_j without i=j
        X_imX_j = X[i] - np.delete(X,i,axis=0)
        xmX_jmi_by_X_imX_j = np.true_divide(xmX_jmi,X_imX_j)
        Pij = np.prod(xmX_jmi_by_X_imX_j,axis=0)
        TermsList.append(Pij*Y[i])
    return np.sum(TermsList,axis=0)

def SumSubpixelAperturewindow(ImageArrayStrip,TopEdgeCoord,BottomEdgeCoord,EdgepixelOrder):
    """ Returns sum of the subpixel aperture window on top of ImageArrayStrip, where the window is defined by the TopEdgeCoord, BottomEdgeCoord.
    The contribution of subpixel at the edge is calculated by LagrangeInterpolation of the cumulative flux in the pixel grid.
    The order of the polynomial used for interpolation is determinaed by EdgepixelOrder.
    Note: The sub-pixel aperture window width should be wider than 1 pixel.
    INPUT:
        ImageArrayStrip : Numpy 2D array strip, where columns are the axis along which the subpixel aperture to sum is defined.
        TopEdgeCoord: Numpy 1D array, with the same length as columns in ImageArrayStrip, ie. ImageArrayStrip.shape[1].
                      This array defines the top edge subpixel coordinate of the mask.
        BottomEdgeCoord: Numpy 1D array, with the same length as columns in ImageArrayStrip, ie. ImageArrayStrip.shape[1].
                      This array defines the bottom edge subpixel coordinate of the mask.
                      Note:  TopEdgeCoord - BottomEdgeCoord should be > 1. i.e., The aperture window should be larger than a pixel
        EdgepixelOrder: The order of the polynomial used for interpolation of the ub-pixel aperture on the edge pixel.
    OUTPUTS:
        SumArray : Numpy 1D array, which contains the sum of the flux inside th sub-pixel aperture along each column of ImageArrayStrip

    """

    if np.any((TopEdgeCoord - BottomEdgeCoord) < 1):
        raise ValueError('TopEdgeCoord - BottomEdgeCoord should be > 1. i.e., The aperture window should be larger than a pixel')

    # To sum up all the pixels fully inside the aperture, create a mask
    Igrid,Jgrid = np.mgrid[0:ImageArrayStrip.shape[0],0:ImageArrayStrip.shape[1]]

    FullyInsidePixelMask = (Igrid >= np.int_(BottomEdgeCoord)+1 ) & (Igrid <= np.int_(TopEdgeCoord)-1 )
    FullyInsidePixelSum = np.ma.sum(np.ma.array(ImageArrayStrip,mask=~FullyInsidePixelMask),axis=0).data

    # Now calculate the interpolated flux in edge pixel

    ### Below is where all the magic happens, don't mess with it without understanding what you are dealing with.

    # Coorindate of pixel boundaries
    TopEdgeXrows = np.vstack([np.int_(TopEdgeCoord)+i for i in range(-EdgepixelOrder//2,EdgepixelOrder//2 +1)])
    # Assign the value of a pixel below to the pixel boundry
    TopEdgeYvaluerows = np.vstack([ImageArrayStrip[np.int_(TopEdgeCoord)+i-1,np.arange(ImageArrayStrip.shape[1])] for i in range(-EdgepixelOrder//2,EdgepixelOrder//2 +1)])
    # Calculate the cumulative sum values for interpolation
    CumSumTopEdgeYvaluerows = np.cumsum(TopEdgeYvaluerows,axis=0)
    # Set the value at edge of fully inside pixel to be zero
    CumSumTopEdgeYvaluerows -= CumSumTopEdgeYvaluerows[(EdgepixelOrder+1)//2,:]
    # Now obtain the Top edge interpolated value
    TopEdgePixelContribution = LagrangeInterpolateArray(TopEdgeCoord,TopEdgeXrows,CumSumTopEdgeYvaluerows)

    # Simillarly get the flux contrinution from Bottom Edge as well
    # Coorindate of pixel boundaries
    BottomEdgeXrows = np.vstack([np.int_(BottomEdgeCoord)+i for i in range(-EdgepixelOrder//2+1,EdgepixelOrder//2 +1+1)])
    # Assign the value of a pixel above to the pixel boundry
    BottomEdgeYvaluerows = np.vstack([ImageArrayStrip[np.int_(BottomEdgeCoord)+i,np.arange(ImageArrayStrip.shape[1])] for i in range(-EdgepixelOrder//2+1,EdgepixelOrder//2 +1+1)])
    # Calculate the cumulative sum values for interpolation in up to down direction
    CumSumBottomEdgeYvaluerows = np.cumsum(BottomEdgeYvaluerows[::-1,:],axis=0)[::-1,:]
    # Set the value at edge of fully inside pixel to be zero
    CumSumBottomEdgeYvaluerows -= CumSumBottomEdgeYvaluerows[(EdgepixelOrder-1)//2 +1,:]
    # Now obtain the Top edge interpolated value
    BottomEdgePixelContribution = LagrangeInterpolateArray(BottomEdgeCoord,BottomEdgeXrows,CumSumBottomEdgeYvaluerows)

    #Add both the top and bottom edge pixel contrinution to FullInside Pixel Sum
    return FullyInsidePixelSum + TopEdgePixelContribution + BottomEdgePixelContribution

def SumCurvedApertures(SpectrumFile, ApertureTraceFuncDic, apwindow=(None,None), EdgepixelOrder=3, apertures=None, dispersion_Xaxis=True, ShowPlot=False):
    """ Returns the sum of each unrectified curved aperture inside the apwindow.
    Input:
      SpectrumFile: Spectrum image to rectify
      ApertureTraceFuncDic: Dictionary of function which traces the aperture
      EdgepixelOrder : Order of the polynomial to be used for interpolating the edge pixel
    """
    if isinstance(SpectrumFile,str):
        ImageArray = fits.getdata(SpectrumFile)
    else:
        ImageArray = SpectrumFile
    # Transpose the image if dispersion is not along X axis
    if not dispersion_Xaxis:
        ImageArray = ImageArray.T

    if apertures is None :
        apertures = ApertureTraceFuncDic.keys()

    ApertureSumDic = {}
    DispCoords = np.arange(ImageArray.shape[1])

    for aper in apertures:
        # First create the pixel mapping
        XDCoordsCenter = ApertureTraceFuncDic[aper](DispCoords)
        # Clip any pixel coordinates which goes outside ImageArray during extraction
        XDCoordsCenter = np.clip(XDCoordsCenter,
                                 0-(apwindow[0]-EdgepixelOrder),
                                 ImageArray.shape[0]-(apwindow[1]+EdgepixelOrder))
        # Cut out a Rectangle strip of interest for lower memory and faster calculations
        StripStart = int(np.rint(np.min(XDCoordsCenter)+apwindow[0]-EdgepixelOrder))
        StripEnd = int(np.rint(np.max(XDCoordsCenter)+apwindow[1]+EdgepixelOrder))
        ImageArrayStrip = ImageArray[StripStart:StripEnd,:]
        # New coordinates inside the Strip, Add 0.5 to convert pixel index to pixel center coordinate
        XDCoordsCenterStrip = XDCoordsCenter - StripStart +0.5
        TopEdgeCoord = XDCoordsCenterStrip+apwindow[1]
        BottomEdgeCoord = XDCoordsCenterStrip+apwindow[0]
        # Sum the flux inside the sub-pixel aperture window
        ApertureSumDic[aper] = SumSubpixelAperturewindow(ImageArrayStrip,TopEdgeCoord,BottomEdgeCoord,EdgepixelOrder)

    if ShowPlot:
        plt.plot(ApertureSumDic[aper])
        plt.ylabel('Sum of Counts')
        plt.xlabel('pixels')
        plt.title('Aperture : {0}'.format(aper))
        plt.show()

    return ApertureSumDic


def FlatRelativeOptimatExtractionAperturewindow(ImageArrayStrip,FlatArrayStrip,VarianceArrayStrip,TopEdgeCoord,BottomEdgeCoord,CRsigma=0,maxCRiter=7,do_non_optimal=False,verbose=True,logprefix=''):
    """ Returns flat-relative optimal extracted values from the aperture window on top of ImageArrayStrip, where the window is defined by the TopEdgeCoord, BottomEdgeCoord.
        Only full pixels which fall completely inside the aperture window is used
        Ref: https://arxiv.org/pdf/1311.5263.pdf, Equation 7 and 9
    INPUT:
        ImageArrayStrip : Numpy 2D array strip, where columns are the axis along which the aperture to sum is defined.
        FlatArrayStrip : Numpy 2D array strip of the Flat, where columns are the axis along which the aperture to sum is defined.
        VarianceArrayStrip : Numpy 2D array strip of the Variance image, where columns are the axis along which the aperture to sum is defined.
        TopEdgeCoord: Numpy 1D array, with the same length as columns in ImageArrayStrip, ie. ImageArrayStrip.shape[1].
                      This array defines the top edge subpixel coordinate of the mask.
        BottomEdgeCoord: Numpy 1D array, with the same length as columns in ImageArrayStrip, ie. ImageArrayStrip.shape[1].
                      This array defines the bottom edge subpixel coordinate of the mask.
                      Note:  TopEdgeCoord - BottomEdgeCoord should be > 1. i.e., The aperture window should be larger than a pixel
        CRsigma : float (default: 0)
                      If CRsigma > 0, then cosmic ray rejection is done using the CRsigma as threshold parmeter.
                      Note: The sufix sigma is a misnomer to this term. It doesnot represent stdev.
                      Recommended value is CRsigma = 1.4
        maxCRiter: int (default: 7)
                      Maximum number of iterations to do to identify CR hits.
        do_non_optimal: True/False (default: False)
                      Do a non optimal extraction where instead of weighting by the varicane of the data, use the flat for relative weighting.
                      This is equivalent to sum extraction of flux divided by sum extracted flat with the benefits of CR rejection.
                      This is useful for avoiding S/N dependent extraction artefacts on circular fibers
        logprefix: str (default : '')
                      Optional prefix to add to logs

    OUTPUTS:
        FlatRelativeFlux : Numpy 1D array, which contains the flat relative flux inside the pixel aperture along each column of ImageArrayStrip
        VarFlatRelativeFlux : Numpy 1D array, which contains the variance of the flat relative flux inside the pixel aperture along each column of ImageArrayStrip
    """

    if np.any((TopEdgeCoord - BottomEdgeCoord) < 1):
        raise ValueError('TopEdgeCoord - BottomEdgeCoord should be > 1. i.e., The aperture window should be larger than a pixel')
    if (CRsigma > 0): # if CR hit is being detected and masked, preserve the original input varance from getting modified
        VarianceArrayStrip = np.copy(VarianceArrayStrip)

    # To sum up all the pixels fully inside the aperture, create a mask
    Igrid,Jgrid = np.mgrid[0:ImageArrayStrip.shape[0],0:ImageArrayStrip.shape[1]]
    FullyInsidePixelMask = (Igrid >= np.int_(BottomEdgeCoord)+1 ) & (Igrid <= np.int_(TopEdgeCoord)-1 )

    # Calculate the FS and FF terms
    wFS = FlatArrayStrip*ImageArrayStrip/VarianceArrayStrip
    wFF = FlatArrayStrip*FlatArrayStrip/VarianceArrayStrip

    for it in range(maxCRiter):
        FinitePixelMask = np.isfinite(ImageArrayStrip) & np.isfinite(VarianceArrayStrip) & np.isfinite(FlatArrayStrip)

        FullyInsidePixelSum_wFS = np.ma.filled(np.ma.sum(np.ma.array(wFS,mask=~(FullyInsidePixelMask & FinitePixelMask)),axis=0),fill_value=np.nan)
        FullyInsidePixelSum_wFF = np.ma.filled(np.ma.sum(np.ma.array(wFF,mask=~(FullyInsidePixelMask & FinitePixelMask)),axis=0),fill_value=np.nan)

        FlatRelativeFlux = FullyInsidePixelSum_wFS/FullyInsidePixelSum_wFF
        VarFlatRelativeFlux = 1./FullyInsidePixelSum_wFF

        if CRsigma > 0:
            FluxMinusModel =  ImageArrayStrip - FlatArrayStrip * FlatRelativeFlux[np.newaxis,:]
            FluxMinusModel_Var = VarianceArrayStrip - FlatArrayStrip*FlatArrayStrip * VarFlatRelativeFlux[np.newaxis,:]
            # Model / std image
            FluxModelbystdev = FluxMinusModel / np.sqrt(FluxMinusModel_Var)
            # Identify the regions of CR hts by 2d technique
            CRpixelMask = threshold_in_laplace_space(FluxModelbystdev,thresh=CRsigma,pixelMask=FullyInsidePixelMask,
                                                     TopEdgeCoord=TopEdgeCoord,BottomEdgeCoord=BottomEdgeCoord)
            # Following lines are needed only if the above step is not sufficent
            # struct2 = ndimage.generate_binary_structure(2, 2) # Dilate and fill holes in the CR region
            # CRregion = ndimage.binary_fill_holes(ndimage.binary_dilation(CRregion,structure=struct2,mask=FullyInsidePixelMask))
            # # Find the real CR hit pixels inside the CR regions
            # CRpixelMask = CRregion & (FluxModelbystdev>CRsigma)

            NewCRmask = FullyInsidePixelMask & FinitePixelMask & CRpixelMask
            if verbose:
                logging.info(logprefix+'Number of CR pixels to be rejected in iteration {0}: {1}'.format(it,np.sum(NewCRmask)))
            if np.any(NewCRmask): # Detected CR hits
                VarianceArrayStrip[NewCRmask] = -np.inf
            else:
                logging.info(logprefix+'Total number of CR pixels rejected after final iteration {0}: {1}'.format(it,np.sum(np.isneginf(VarianceArrayStrip) & FullyInsidePixelMask)))
                CRmask = np.isneginf(VarianceArrayStrip) & FullyInsidePixelMask  # Change this to np.isinf to preserve +infinite variance pixels also
                break # Skip rest of the for loop
        else:
            # No CR correction beign done
            CRmask = None
            break # Escape and exit the for loop
    else:
        # for loop exited without converging on all the CR hit pixels
        logging.warning(logprefix+'Max number of CR hit removal iteration reached ({0}). Incomplete CR hit masking : {1}'.format(it,np.sum(np.isneginf(VarianceArrayStrip) & FullyInsidePixelMask)))
        CRmask = np.isneginf(VarianceArrayStrip) & FullyInsidePixelMask

    if do_non_optimal:
        # Recalculate the FlatRelativeFlux in a non-optimal way
        NonOptimalFullyInsidePixelSum_wFS = np.ma.filled(np.ma.sum(np.ma.array(ImageArrayStrip,mask=~(FullyInsidePixelMask & FinitePixelMask)),axis=0),fill_value=np.nan)
        NonOptimalFullyInsidePixelSum_wFF = np.ma.filled(np.ma.sum(np.ma.array(FlatArrayStrip,mask=~(FullyInsidePixelMask & FinitePixelMask)),axis=0),fill_value=np.nan)
        FlatRelativeFlux = NonOptimalFullyInsidePixelSum_wFS/NonOptimalFullyInsidePixelSum_wFF
        VarFlatRelativeFlux = np.ma.filled(np.ma.sum(np.ma.array(VarianceArrayStrip,mask=~(FullyInsidePixelMask & FinitePixelMask)),axis=0),fill_value=np.nan)/np.power(NonOptimalFullyInsidePixelSum_wFF,2)

    return FlatRelativeFlux, VarFlatRelativeFlux, CRmask

def FlatRelativeOptimatExtraction(SpectrumFile, FlatFile, ApertureTraceFuncDic, VarianceImage=None,
                                  apwindow=(None,None), CRsigma=0, do_non_optimal=False, apertures=None, dispersion_Xaxis=True, ShowPlot=False,verbose=True,logprefix=''):
    """ Returns the flat-relative optimal extrcation from unrectified curved aperture using pixels which are fully inside the apwindow.
        Ref: https://arxiv.org/pdf/1311.5263.pdf
    Input:
      SpectrumFile: Spectrum image to extract
      FlatFile: Continuum Flat relative to which extraction is to be done.
      ApertureTraceFuncDic: Dictionary of function which traces the aperture
      VarianceImage: VarianceImage of the array.
      apwindow: The aperture window to use for extraction. Only pixels which are fully inside are used.
      CRsigma : float (default: 0)
                If CRsigma > 0, then cosmic ray rejection is done using the CRsigma as threshold
      do_non_optimal: True/False (default: False)
                      Do a non optimal extraction where instead of weighting by the varicane of the data, use the flat for relative weighting.
                      This is equivalent to sum extraction of flux divided by sum extracted flat with the benefits of CR rejection.
                      This is useful for avoiding S/N dependent extraction artefacts on circular fibers
     Returns:
       ApertureSumDic, ApertureVarDic, CRimage
    """
    # Load from fits files if they are strings
    if isinstance(SpectrumFile,str):
        ImageArray = fits.getdata(SpectrumFile)
    elif isinstance(SpectrumFile,fits.HDUList):
        ImageArray = SpectrumFile[0].data
    elif isinstance(SpectrumFile,np.ndarray):
        ImageArray = SpectrumFile
    else:
        raise ValueError('Input SpectrumFile in unknown format')

    if isinstance(FlatFile,str):
        FlatArray = fits.getdata(FlatFile)
    elif isinstance(FlatFile,fits.HDUList):
        FlatArray = FlatFile[0].data
    elif isinstance(FlatFile,np.ndarray):
        FlatArray = FlatFile
    else:
        raise ValueError('Input FlatFile in unknown format')

    # If VarianceImage is not provided assume poission noise
    VarianceArray = VarianceImage if (VarianceImage is not None) else np.copy(SpectrumFile)

    # Transpose the image if dispersion is not along X axis
    if not dispersion_Xaxis:
        ImageArray = ImageArray.T
        FlatArray = FlatArray.T
        VarianceArray = VarianceArray.T

    if CRsigma > 0:
        CRimage = np.zeros(ImageArray.shape,dtype=np.bool)
    else:
        CRimage = None
    if apertures is None :
        apertures = ApertureTraceFuncDic.keys()

    ApertureSumDic = {}
    ApertureVarDic = {}
    DispCoords = np.arange(ImageArray.shape[1])

    for aper in apertures:
        # First create the pixel mapping
        XDCoordsCenter = ApertureTraceFuncDic[aper](DispCoords)
        # Clip any pixel coordinates which goes outside ImageArray during extraction
        XDCoordsCenter = np.clip(XDCoordsCenter,
                                 0-apwindow[0],
                                 ImageArray.shape[0]-apwindow[1])
        # Cut out a Rectangle strip of interest for lower memory and faster calculations
        StripStart = int(np.rint(np.min(XDCoordsCenter)+apwindow[0]))
        StripEnd = int(np.rint(np.max(XDCoordsCenter)+apwindow[1]))
        ImageArrayStrip = ImageArray[StripStart:StripEnd,:]
        FlatArrayStrip = FlatArray[StripStart:StripEnd,:]
        VarianceArrayStrip = VarianceArray[StripStart:StripEnd,:]
        # New coordinates inside the Strip, Add 0.5 to convert pixel index to pixel center coordinate
        XDCoordsCenterStrip = XDCoordsCenter - StripStart +0.5
        TopEdgeCoord = XDCoordsCenterStrip + apwindow[1]
        BottomEdgeCoord = XDCoordsCenterStrip + apwindow[0]
        # Sum the flux inside the sub-pixel aperture window
        ApertureSumDic[aper], ApertureVarDic[aper], CRmask = FlatRelativeOptimatExtractionAperturewindow(ImageArrayStrip,
                                                                                                         FlatArrayStrip,
                                                                                                         VarianceArrayStrip,
                                                                                                         TopEdgeCoord,
                                                                                                         BottomEdgeCoord,
                                                                                                         CRsigma=CRsigma,
                                                                                                         do_non_optimal=do_non_optimal,
                                                                                                         verbose=verbose,
                                                                                                         logprefix=logprefix+'O '+str(aper)+' ')
        if CRmask is not None:
            CRimage[StripStart:StripEnd,:] |= CRmask

    if ShowPlot:
        plt.plot(ApertureSumDic[aper])
        plt.ylabel('Sum of Counts')
        plt.xlabel('pixels')
        plt.title('Aperture : {0}'.format(aper))
        plt.show()
        if CRmask is not None:
            plt.imshow(CRimage)
            plt.title('Masked CR hit pixels')
            plt.show()

    return ApertureSumDic, ApertureVarDic, CRimage


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
            WindowminX = np.max(( np.min(Xc)-4, 0))
            WindowmaxX = np.min(( np.max(Xc)+4, SpectrumImage.shape[0]))
            WindowminY = np.max(( np.min(Yc)-4, 0))
            WindowmaxY = np.min(( np.max(Yc)+4, SpectrumImage.shape[1]))
            TileX, TileY = np.meshgrid(range(WindowminX,WindowmaxX),range(WindowminY,WindowmaxY),indexing='ij')
            # TileX, TileY = np.meshgrid(range(np.min(Xc)-4,np.max(Xc)+4),range(np.min(Yc)-4,np.max(Yc)+4),indexing='ij')
            FullCoordsSet = set(zip(TileX.flatten(), TileY.flatten()))
            GoodPixelXY = np.array(list(FullCoordsSet - set(zip(*np.where(labeled_array != 0)))))
            # Create an interpolator to interpolate
            LinInterp = interp.LinearNDInterpolator(GoodPixelXY,SpectrumImage[GoodPixelXY[:,0],GoodPixelXY[:,1]])
            # Fill the values with the interpolated pixels
            SpectrumImage[Xc,Yc] = LinInterp(np.array([Xc,Yc]).T)

    return SpectrumImage


##########################################################################

def add_extracted_spectra_dicts(list_of_extrated_spectra_dicts):
    """ Returns the summed output of the list of extracted spectra
    Input:
        list_of_extrated_spectra_dicts: (list of dict)
                List of the extracted spectra dictionaries
    Returns:
        output_sum_spectrum
          sum of the input spectra
    """
    output_sum_spectrum = {}
    for spectrumDict in list_of_extrated_spectra_dicts:
        # Loop through the orders to sum
        for i in sorted(spectrumDict.keys()):
            try:
                output_sum_spectrum[i] += spectrumDict[i]
            except KeyError:
                output_sum_spectrum[i] = spectrumDict[i].copy()

    return output_sum_spectrum

def add_scaled_variance_to_variance_dictlist(inputVariance_list,varianceToAdd_list,scaling_list=None):
    """ Returns the new list of variance dictionary after adding the  varianceToAdd_list/scaling_list**2 to inputVariance_list
    Input:
         inputVariance_list : (list of dict)
                List of the flat relative extracted spectra's varaince
         varianceToAdd_list : (list of dict)
                List of extra variance to add
         scaling_list : (list of dict)
                List of the flat's continuum model to scale

    Returns:
         outputVariance_list : (list of dict)
                Returns inputVariance_list + varianceToAdd_list/scaling_list**2
    """
    outputVariance_list = []
    for spectrum_varDict, spectrum_varDict2, continuum_spectrumDict in zip(inputVariance_list,
                                                                           varianceToAdd_list,
                                                                           scaling_list):
        # Loop through the orders to multiply
        ApVarianceDict = {}
        for i in sorted(spectrum_varDict.keys()):
            ApVarianceDict[i] = spectrum_varDict[i] + spectrum_varDict2[i]/np.power(continuum_spectrumDict[i],2)
        outputVariance_list.append(ApVarianceDict)

    return outputVariance_list

##########################################################################

def WriteFitsFileSpectrum(FluxSpectrumDic, outfilename, VarianceSpectrumDic=None, fitsheader=None,
                          BkgFluxSpectrumList=(), BkgFluxVarSpectrumList=()):
    """ Writes the FluxSpectrumDic into fits image with filename outfilename
    with fitsheader as header."""
    # First create a 2D array to hold all apertures of the spectrum
    Spectrumarray = np.array([FluxSpectrumDic[i] for i in sorted(FluxSpectrumDic.keys())])
    hdu = fits.PrimaryHDU(Spectrumarray,header=fitsheader)
    hdulist =fits.HDUList([hdu])
    for i,bkgfluxdic in enumerate(BkgFluxSpectrumList):
        bkgfluxSpectrumarray = np.array([bkgfluxdic[i] for i in sorted(bkgfluxdic.keys())])
        hdu_bkg = fits.ImageHDU(bkgfluxSpectrumarray,name='Bkg Flux {0}'.format(i))
        hdulist.append(hdu_bkg)
    if VarianceSpectrumDic is not None:
        VarianceSpectrumarray = np.array([VarianceSpectrumDic[i] for i in sorted(VarianceSpectrumDic.keys())])
        hdu_var = fits.ImageHDU(VarianceSpectrumarray,name='Variance')
        hdulist.append(hdu_var)
        for i,bkgvardic in enumerate(BkgFluxVarSpectrumList):
            bkgvarSpectrumarray = np.array([bkgvardic[i] for i in sorted(bkgvardic.keys())])
            hdu_bkg_var = fits.ImageHDU(bkgvarSpectrumarray,name='Bkg Variance {0}'.format(i))
            hdulist.append(hdu_bkg_var)

    hdulist.writeto(outfilename)
    return outfilename


####################################################################################

def write_crmask(crmask,fname=None,outputDirectory=None,fitsheader=None):
    """ Save the boolean crmask fits file.
    If the file already exists, it will be combined with the new mask by OR operation.
    Output filename will be generated  in the outputDirectory
    """
    if fname is None:
        # Load the existing file name form header
        fname = fitsheader['FILENAME']
    try:
        outputmask = fits.getdata(os.path.join(outputDirectory,fname)).astype(np.bool) | crmask
    except IOError:
        outputmask = crmask

    fits.PrimaryHDU(data=outputmask.astype(np.uint8),header=fitsheader).writeto(os.path.join(outputDirectory,fname),overwrite=True)

    return os.path.join(outputDirectory,fname)

def write_2dimage_fitsfile(image_dict,fname=None,outputDirectory=None,fitsheader=None,fname_prefix='ScatterModel',append=True):
    """ Save the images in the input image_dict as a multiextension fits file with the name fname.
    Output filename will be generated in the outputDirectory.
    if append is True and output file already exists then it will append to an existing file.
    """

    if fname is None:
        # Load the existing the L1 file
        fname = fitsheader['FILENAME']

    if append:
        try:
            outhdulist = fits.open(os.path.join(outputDirectory,fname))
        except IOError:
            outhdulist = fits.HDUList([fits.PrimaryHDU(header=fitsheader)])
    else:
        outhdulist = fits.HDUList([fits.PrimaryHDU(header=fitsheader)])

    for extname in image_dict:
        outhdulist.append(fits.ImageHDU(image_dict[extname].astype(np.float32),name=extname))

    outhdulist.writeto(os.path.join(outputDirectory,fname),overwrite=True)

    return os.path.join(outputDirectory,fname)

#######################################################################################
def parse_str_to_types(string):
    """ Converts string to different object types they represent.
    Supported formats: True,Flase,None,int,float,list,tuple"""
    string = string.strip() # remove any extra white space paddings
    if string == 'True':
        return True
    elif string == 'False':
        return False
    elif string == 'None':
        return None
    elif string.lstrip('-+ ').isdigit():
        return int(string)
    elif (string[0] in '[(') and (string[-1] in ')]'): # Recursively parse a list/tuple into a list
        if len(string[1:-1]) == 0:
            return []
        else:
            return [parse_str_to_types(s) for s in re.split(r',\s*(?=[^)]*(?:\(|$))', string[1:-1])]  # split at comma unless it is inside a ( )
    else:
        try:
            return float(string)
        except ValueError:
            return string



def create_configdict_from_file(configFilename,listOfConfigSections=None,flattenSections=True):
    """ Returns a configuration object as a dictionary by loading the config file.
        Values in the config files are parsed appropriately to python objects.
    Parameters
    ----------
    configFilename : str
                    File name of the config file to load
    listOfConfigSections : list (default:None)
                    Only the sections in the listOfConfigSections will be loaded to the dictionary.
                    if listOfConfigSections is None (default), all the sections in config file will be loaded.
    flattenSections: (bool, default=True)
                    True: Flattens the sections in the config file into a single level Config dictionary.
                    False: will return a dictionary of dictionaries for each section.
    Returns
    -------
    configDictionary : dictionary
                    if `flattenSections` is True (default): Flattened {key:value,..} dictionary is returned
                    else: A dictionary of dictionary is returned {Section:{key:Value},..}
    """
    configLoader = ConfigParser.ConfigParser()
    configLoader.optionxform = str  # preserve the Case sensitivity of keys
    with open(configFilename) as cfgFile:
        try:
            configLoader.read_file(cfgFile)
        except AttributeError: # python 2 environment doesnot have .rad_file attribute
            configLoader.readfp(cfgFile)

    # Create a Config Dictionary
    config = {}
    if isinstance(listOfConfigSections,str): # Convert the single string to a list
        listOfConfigSections = [listOfConfigSections]
    elif listOfConfigSections is None:
        listOfConfigSections = configLoader.sections()

    for configSection in listOfConfigSections:
        if flattenSections:  # Flatten the sections
            for key,value in configLoader.items(configSection):
                config[key] = parse_str_to_types(value)
        else:
            subConfig = {}
            for key,value in configLoader.items(configSection):
                subConfig[key] = parse_str_to_types(value)
            config[configSection] = subConfig
    return config

def parse_args(raw_args=None):
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Spectrum Extraction Tool")
    parser.add_argument('SpectrumFile', type=str,
                        help="The 2D Spectrum image fits file")
    parser.add_argument('ConfigFile', type=str,
                         help="Configuration file which contains settings for extraction")
    parser.add_argument('--FlatNFile', type=str, default=None,
                        help="Normalized Flat file to be used for correcting pixel to pixel variation")
    parser.add_argument('--BadPixMask', type=str, default=None,
                        help="Bad Pixel Mask file to be used for fixing bad pixels")
    parser.add_argument('--ApertureTraceFilename', type=str, default=None,
                        help="Aperture trace file to be used for aperture extraction")
    parser.add_argument('--ContinuumFile', type=str, default=None,
                        help="Continuum flat source to be used for aperture extraction")
    parser.add_argument('--ApertureLabel', type=str, default=None,
                        help="Array of labels for the aperture trace regions")
    parser.add_argument('--ExtractionMethod', type=str, default=None,
                        help="Extraction method to use : Sum, FlatRelativeOptimal")
    parser.add_argument('--Do_non_optimal', type=str, default=None,
                        help="True/False Whether to do non optimal extraction while doing FlatRelativeOptimal extraction. Useful for non rectangular fibers.")
    parser.add_argument('--VarianceExt', type=int, default=None,
                        help="Provide extension of Variance array if needs to be extracted")
    parser.add_argument('OutputFile', type=str,
                        help="Output filename to write extracted spectrum")
    parser.add_argument('--SaveCRmaskDirectory', type=str,default=None,
                        help="Optional directory path to save the Cosmic Ray Mask created during the extraction")
    parser.add_argument('--logfile', type=str, default=None,
                        help="Log Filename to write logs during the run")
    parser.add_argument("--loglevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")
    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    """ Extracts 2D spectrum image into 1D spectrum """
    args = parse_args(raw_args)

    if args.logfile is None:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.getLevelName(args.loglevel))
    else:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.getLevelName(args.loglevel),
                            filename=args.logfile, filemode='a')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # Sent info to the stdout as well

    # disable matplotlib's debug logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    Config = create_configdict_from_file(args.ConfigFile,listOfConfigSections=['processing_settings',
                                                                               'tracing_settings',
                                                                               'extraction_settings'])


    SpectrumFile = args.SpectrumFile
    OutputFile = args.OutputFile

    logprefix = os.path.basename(SpectrumFile)+':'
    # Override the Config file with command line arguments
    if args.ApertureTraceFilename is not None:
        Config['ApertureTraceFilename'] = args.ApertureTraceFilename
    if args.ContinuumFile is not None:
        Config['ContinuumFile'] = args.ContinuumFile
    if args.FlatNFile is not None:
        Config['FlatNFile'] = args.FlatNFile
    if args.BadPixMask is not None:
        Config['BadPixMask'] = args.BadPixMask
    if args.ApertureLabel is not None:
        Config['ApertureLabel'] = args.ApertureLabel
    if args.VarianceExt is not None:
        Config['VarianceExt'] = args.VarianceExt
    if args.ExtractionMethod is not None:
        Config['ExtractionMethod'] = args.ExtractionMethod
    if args.SaveCRmaskDirectory is not None:
        Config['SaveCRmaskDirectory'] = args.SaveCRmaskDirectory
    if args.Do_non_optimal is not None:
        Config['Do_non_optimal'] = parse_str_to_types(args.Do_non_optimal)

    if os.path.isfile(OutputFile):
        logging.warning('WARNING: Output file {0} already exist'.format(OutputFile))
        logging.warning('Skipping this image extraction..')
        sys.exit(1)

    ################################################################################
    # Starting Extraction process
    ################################################################################
    logging.info('Extracting {0}..'.format(SpectrumFile))
    fheader = fits.getheader(SpectrumFile)

    ################################################################################
    # Get/Create the apertrue trace centers for extraction
    ################################################################################
    if Config['ApertureTraceFilename'] is not None:
        ApertureTraceFilename = Config['ApertureTraceFilename']
    else:
        ApertureTraceFilename = Config['ContinuumFile']+'_trace.pkl'
    if os.path.isfile(ApertureTraceFilename):  # Save time and load a pre-existing aperture trace
        logging.info('Loading existing trace coordinates {0}'.format(ApertureTraceFilename))
        try:
            ApertureCenters = pickle.load(open(ApertureTraceFilename,'rb'))
        except UnicodeDecodeError: # Support for old python2 .pkl trace file
            ApertureCenters = pickle.load(open(ApertureTraceFilename,'rb'),encoding='latin1')
    else:
        # First Load/Create the Aperture Label file to label and extract trace from continuum file
        ###########
        if os.path.isfile(str(Config['ApertureLabel'])):
            ApertureLabel = np.load(Config['ApertureLabel'])
        else:
            # ApertureLabel = CreateApertureLabelByThresholding(Config['ContinuumFile'],BadPixMask=Config['BadPixMask'],bsize=51,offset=0,minarea=2000, ShowPlot=True,DirectlyEnterRelabel= True)
            ApertureLabel, ApertureCenters_Trace1 = CreateApertureLabelByXDFitting(Config['ContinuumFile'],BadPixMask=Config['BadPixMask'],
                                                                                   startLoc=Config['Start_Location'],avgHWindow=Config['AvgHWindow_forTrace'],
                                                                                   TraceHWidth=Config['HWidth_inXD'],trace_fit_deg=Config['ApertureTraceFuncDegree'],
                                                                                   dispersion_Xaxis=Config['dispersion_Xaxis'],extrapolate_thresh=Config['extrapolate_thresh_forTrace'],
                                                                                   extrapolate_order=Config['extrapolate_order_forTrace'], ShowPlot=Config['ShowPlot_Trace'],
                                                                                   return_trace=True)
            # Save the aperture label if a non existing filename was provided as input
            if isinstance(Config['ApertureLabel'],str):
                np.save(Config['ApertureLabel'],ApertureLabel)
        ###########

        # Trace the center of the apertures
        ApertureCenters = FitApertureCenters(Config['ContinuumFile'],ApertureLabel,
                                             apwindow=(-Config['HWidth_inXD'],Config['HWidth_inXD']),
                                             dispersion_Xaxis = Config['dispersion_Xaxis'], ShowPlot=Config['ShowPlot_Trace'])
        #Save for future
        with open(ApertureTraceFilename,'wb') as tracepickle:
            pickle.dump(ApertureCenters,tracepickle)

    fheader['APFILE'] = (os.path.basename(ApertureTraceFilename), 'Aperture Trace Filename')


    ################################################################################
    # Load and pre process the spectrum before extraction
    ################################################################################

    # Load the spectrum array
    SpectrumImage = fits.getdata(SpectrumFile)
    if Config['VarianceExt'] is not None:
        VarianceImage = fits.getdata(SpectrumFile,ext=Config['VarianceExt'])
    else:
        VarianceImage = None

    # Apply flat correction for pixel to pixel variation
    if Config['FlatNFile'] is not None:
        logging.info('Doing Flat correction..')
        NFlat = fits.getdata(Config['FlatNFile'])
        SpectrumImage /= NFlat
        fheader['FLATFIL'] = (os.path.basename(Config['FlatNFile']), 'Flat Filename')
        fheader['HISTORY'] = 'Flat fielded the image'
        if Config['VarianceExt'] is not None:
            VarianceImage /= NFlat**2

    # Apply bad pixel correction
    if Config['BadPixMask'] is not None:
        logging.info('Doing Bad pixel correction..')
        BPMask = fits.getdata(Config['BadPixMask']) if Config['BadPixMask'][-5:] == '.fits' else np.load(Config['BadPixMask'])
        SpectrumImage = fix_badpixels(SpectrumImage,BPMask)
        fheader['BPMASK'] = (os.path.basename(Config['BadPixMask']), 'Bad Pixel Mask File')
        fheader['HISTORY'] = 'Fixed the bad pixels in the image'
        if Config['VarianceExt'] is not None:
            #update the VarianceImage to reflect the fixing of bad pixels
            VarianceImage[BPMask == 0] = np.inf

    # Also use cosmic_lacomic to fix any spiky CR hits in data
    if Config['DoCosmicRayClean']:
        crgain = fheader[Config['CosmicRayCleanGain']] if isinstance(Config['CosmicRayCleanGain'],str) else Config['CosmicRayCleanGain']
        SpectrumImage , cmask = cosmicray_lacosmic(SpectrumImage,
                                                   sigclip=Config['CosmicRayCleanSigclip'],
                                                   gain=crgain,gain_apply=False)
        logging.info('Cleaned {0} cosmic ray pixels'.format(np.sum(cmask)))
        SpectrumImage = SpectrumImage.value
        fheader['LACOSMI'] = (np.sum(cmask),'Number of CR pixel fix by L.A. Cosmic')
        fheader['HISTORY'] = 'Fixed the CosmicRays using LACosmic'
        if Config['VarianceExt'] is not None:
            pass  # In future update the VarianceImage to reflect the fixing of cosmic rays
    else:
        cmask = None

    ################################################################################
    # Create the final aperture trace function for the image to be extracted
    ################################################################################
    # If requested to refit the apertures in the XD direction
    if Config['ReFitApertureInXD']:
        if isinstance(Config['ReFitApertureInXD'],(list,tuple)):
            Avg_XD_shift, PixDomain = Config['ReFitApertureInXD'][0], Config['ReFitApertureInXD'][1]
            logging.info('Applying shift in XD position :{0} in -1to1 domain of {1}'.format(tuple(Avg_XD_shift),tuple(PixDomain)))
        else:
            if Config['ReFitApertureInXD'] is True:
                XDshiftmodel = 'p0'
            else:
                XDshiftmodel = Config['ReFitApertureInXD']
            logging.info('ReFitting the XD position of aperture to the spectrum using model {0}'.format(XDshiftmodel))
            # Get the XD shift required between the ContinuumFile based aperture and Spectrum to extract
            Avg_XD_shift, PixDomain = CalculateShiftInXD(SpectrumImage,RefImage=Config['ContinuumFile'],
                                                         XDshiftmodel=XDshiftmodel,DWindowToUse=Config['ReFitApertureInXD_DWindow'],
                                                         StripWidth=4*Config['AvgHWindow_forTrace'],Apodize=True,
                                                         bkg_medianfilt=Config['ReFitApertureInXD_BkgMedianFilt'],
                                                         dispersion_Xaxis=Config['dispersion_Xaxis'],ShowPlot=Config['ShowPlot_Trace'])
            logging.info('Fitted shift in XD position :{0} in -1to1 domain of {1}'.format(tuple(Avg_XD_shift),tuple(PixDomain)))
        # Apply the XD shift to the aperture centers
        ApertureCenters = ApplyXDshiftToApertureCenters(ApertureCenters,Avg_XD_shift,PixDomain)

    # Obtain the tracing function of each aperture
    pix_scale_function = partial(scale_interval_m1top1,a=0,b=SpectrumImage.shape[1])
    ApertureTraceFuncDic = Get_ApertureTraceFunction(ApertureCenters,
                                                     deg=Config['ApertureTraceFuncDegree'],
                                                     domain_scale_function=pix_scale_function)
    # Obtain the Slit Shear of each order
    SlitShearFuncDic = Get_SlitShearFunction(ApertureCenters)

    if Config['ShowPlot_Trace']:
        norm = ImageNormalize(SpectrumImage, interval=PercentileInterval(95.),
                              stretch=SqrtStretch())
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if Config['dispersion_Xaxis']:
            im = ax.imshow(SpectrumImage, origin='lower', norm=norm, cmap='gray')
        else:
            im = ax.imshow(SpectrumImage.T, origin='lower', norm=norm, cmap='gray')
        fig.colorbar(im)
        ylim = ax.get_ylim()
        for order in ApertureTraceFuncDic:
            x = np.arange(SpectrumImage.shape[0])
            # Plot the center of the trace
            ax.plot(x,ApertureTraceFuncDic[order](x),ls=':',color='r')
            # Plot the Extraction aperture window
            ax.plot(x,ApertureTraceFuncDic[order](x)+Config['ApertureWindow'][0],color='deeppink')
            ax.plot(x,ApertureTraceFuncDic[order](x)+Config['ApertureWindow'][1],color='deeppink')
            if Config['BkgWindows'] is not None:
                # Plot the Bkg window
                for bkgw_offset in np.array(Config['BkgWindows']).flatten():
                    ax.plot(x,ApertureTraceFuncDic[order](x)+bkgw_offset,ls='--',color='yellowgreen')
        ax.set_title('Fitted Aperture: Star in Pink & Bkg in YellowGreen')
        ax.set_ylim(ylim)
        plt.show()
    ################################################################################
    # Spectral Extraction starts here
    ################################################################################

    BkgFluxSpectrumList = []   # Lists to store optian bkg spectrum
    BkgFluxVarSpectrumList = []

    if Config['RectificationMethod'] == 'Bandlimited':
        logging.info('Doing Rectification :{0}'.format(Config['RectificationMethod']))
        # Get rectified 2D spectrum of each aperture of the spectrum file
        RectifiedSpectrum = RectifyCurvedApertures(SpectrumImage,Config['RectificationWindow'],
                                                   ApertureTraceFuncDic,SlitShearFuncDic,
                                                   dispersion_Xaxis = Config['dispersion_Xaxis'])
        if Config['VarianceExt'] is not None:
            RectifiedVariance = RectifyCurvedApertures(VarianceImage,Config['RectificationWindow'],
                                                       ApertureTraceFuncDic,SlitShearFuncDic,
                                                       dispersion_Xaxis = Config['dispersion_Xaxis'])

        # Do the post rectification extraction
        if Config['ExtractionMethod'] == 'Sum':
            # Sum the flux in XD direction of slit
            SumApFluxSpectrum = SumApertures(RectifiedSpectrum, apwindow=Config['ApertureWindow'],
                                             ShowPlot=False)
            if Config['VarianceExt'] is not None:
                SumApVariance = SumApertures(RectifiedVariance, apwindow=Config['ApertureWindow'],
                                             ShowPlot=False)

        else:
            raise NotImplementedError('Unknown Extraction method {0}'.format(Config['ExtractionMethod']))
    elif Config['RectificationMethod'] is None:
        # No rectification needs to be done.
        # Directly extract from curved data
        if Config['ExtractionMethod'] == 'Sum':
            # Sum the flux in XD direction of slit
            SumApFluxSpectrum = SumCurvedApertures(SpectrumImage, ApertureTraceFuncDic,
                                                   apwindow=Config['ApertureWindow'],
                                                   EdgepixelOrder = 3,
                                                   dispersion_Xaxis = Config['dispersion_Xaxis'],
                                                   ShowPlot=False)
            if Config['VarianceExt'] is not None:
                SumApVariance = SumCurvedApertures(VarianceImage, ApertureTraceFuncDic,
                                                   apwindow=Config['ApertureWindow'],
                                                   EdgepixelOrder = 3,
                                                   dispersion_Xaxis = Config['dispersion_Xaxis'],
                                                   ShowPlot=False)

        elif Config['ExtractionMethod'] == 'FlatRelativeOptimal':
            # estimate the flux in XD direction of slit relative to flat
            SumApFluxSpectrum, SumApVariance, CRimg = FlatRelativeOptimatExtraction(SpectrumImage, 
                                                                                    Config['ContinuumFile'],
                                                                                    ApertureTraceFuncDic,
                                                                                    VarianceImage=VarianceImage,
                                                                                    apwindow=Config['ApertureWindow'],
                                                                                    dispersion_Xaxis=Config['dispersion_Xaxis'],
                                                                                    CRsigma=Config['OptimalExt_CRsigma'],
                                                                                    do_non_optimal=Config['Do_non_optimal'],
                                                                                    ShowPlot=False,
                                                                                    verbose=False,
                                                                                    logprefix=logprefix+':')
            if CRimg is not None:
                cmask = (cmask | CRimg) if cmask is not None else CRimg

        else:
            raise NotImplementedError('Unknown Extraction method {0}'.format(Config['ExtractionMethod']))

        # Now do sum extraction of Bkg pixel window if provided
        if Config['BkgWindows'] is not None:
            logging.info('Doing Bkg extraction from :{0}'.format(Config['BkgWindows']))
            # No rectification needs to be done.
            # Directly extract from curved data
            if isinstance(Config['BkgWindows'][0],(list,tuple)):
                BkgApertrueWindowList = Config['BkgWindows']
            else:
                BkgApertrueWindowList = [Config['BkgWindows']]
            for bkg_aperture in BkgApertrueWindowList:
                # Sum the flux in XD direction of slit
                SumBkgFluxSpectrum = SumCurvedApertures(SpectrumImage, ApertureTraceFuncDic,
                                                       apwindow=bkg_aperture,
                                                       EdgepixelOrder = 2,
                                                       dispersion_Xaxis = Config['dispersion_Xaxis'],
                                                       ShowPlot=False)
                BkgFluxSpectrumList.append(SumBkgFluxSpectrum)
                if Config['VarianceExt'] is not None:
                    SumBkgVariance = SumCurvedApertures(VarianceImage, ApertureTraceFuncDic,
                                                       apwindow=bkg_aperture,
                                                       EdgepixelOrder = 2,
                                                       dispersion_Xaxis = Config['dispersion_Xaxis'],
                                                       ShowPlot=False)
                    BkgFluxVarSpectrumList.append(SumBkgVariance)

    else:
        raise NotImplementedError('Unknown Rectification method {0}'.format(Config['RectificationMethod']))

    # Write the extracted spectrum to output fits file
    fheader['RECTMETH'] = (str(Config['RectificationMethod']), 'Rectification method used')
    fheader['EXTRMETH'] = (str(Config['ExtractionMethod']), 'Extraction method used')
    fheader['APERTURE'] = (str(Config['ApertureWindow']), 'Aperture window used for extraction')
    if Config['ExtractionMethod'] == 'FlatRelativeOptimal':
        fheader['NONOPTIM'] = (str(Config['Do_non_optimal']), 'T/F NonOptimal extraction was done')

    fheader['HISTORY'] = 'Extracted spectrum to 1D'
    if Config['VarianceExt'] is None:
        SumApVariance = None
    _ = WriteFitsFileSpectrum(SumApFluxSpectrum, OutputFile, VarianceSpectrumDic=SumApVariance, fitsheader=fheader,
                              BkgFluxSpectrumList=BkgFluxSpectrumList, BkgFluxVarSpectrumList=BkgFluxVarSpectrumList)
    # Save diagnostic frames
    if (cmask is not None) and (Config['SaveCRmaskDirectory'] is not None):
        _ = write_crmask(cmask,fname=SpectrumFile+'_CRmask.fits',outputDirectory=Config['SaveCRmaskDirectory'],fitsheader=fheader)

    logging.info('Extracted {0} => {1} output file'.format(SpectrumFile,OutputFile))

    if Config['ReFitApertureInXD']:
        return OutputFile, Avg_XD_shift, PixDomain
    else:
        return OutputFile

if __name__ == '__main__':
    _ = main()
