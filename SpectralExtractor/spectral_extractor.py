#!/usr/bin/env python
""" This tool is to extract 1D spectrum form a 2D image """
import argparse
import ConfigParser
import sys
import os
import numpy as np
import numpy.ma
from astropy.io import fits
from astropy.stats import mad_std
import matplotlib.pyplot as plt
from skimage import filters
from skimage import morphology
from scipy import ndimage, signal
import scipy.interpolate as interp
import scipy.optimize as optimize
from functools32 import partial
import pickle
from ccdproc import cosmicray_lacosmic 
from RVEstimator.interpolators import BandLimited2DInterpolator
from WavelengthCalibrationTool.recalibrate import ReCalibrateDispersionSolution, scale_interval_m1top1


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
        BPMask = np.load(BadPixMask) # fits.getdata(Config['BadPixMask'])
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
    print('Detected {0} order traces'.format(num_traces))
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
    uinputfile = raw_input('Enter the filename :').strip()
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
    FullCoorindateOfTraceDic = {o:[[d],[xd]] for o,d,xd in zip(LabelList,[startLoc]*len(LabelList),XDCenterList)}

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
                print(e)
                print('Failed fitting.. Skipping {0} pixel position'.format(newDLoc))
            else:
                # Calculate the new pixel coordinates of previous centroids 
                newXDCenterList = [NearestIndx(shifted_pixels,icent) for icent in newRefXDCenterList]
                newXDCenterList, newXDCenterList_err  = RefineCentriodsInSignal(newFlux,newXDCenterList,TraceHWidth,Xpixels=newpixels)
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
                        print('Identified Traces {0} to extrapolate for trace {1} with error {2}'.format(NearestGoodPoints,ic,newXDCenterList_err[ic]))
                        # Fit the polynomial to extrapolate to obtain ic trace location
                        extrp_p = np.polyfit(NearestGoodPoints,PositionDiffArray[NearestGoodPoints],extrapolate_order)
                        new_pos_diff = np.polyval(extrp_p,ic)
                        PositionDiffArray[ic] = new_pos_diff
                        newXDCenterList[ic] = XDCenterList[ic] + new_pos_diff
                # update the Dictionary
                for i,o in enumerate(LabelList):
                    FullCoorindateOfTraceDic[o][0].append(newDLoc)
                    FullCoorindateOfTraceDic[o][1].append(newXDCenterList[i])

                #Change the Reference to the new DLoc position
                newRefFlux = np.vstack([newpixels,newFlux]).T
                newRefXDCenterList = newXDCenterList
            finally:
                newDLoc = newDLoc + max(1,np.abs(stepDLoc)//2)*np.sign(stepDLoc)

    # Finally fit a trace function for each order and create an Aperture Label array
    ApertureLabel = np.zeros(ContinuumFile.shape)
    # First conver the dictionary values to a numpy array
    for o in LabelList:
        FullCoorindateOfTraceDic[o] = np.array(FullCoorindateOfTraceDic[o])
    ApertureTraceFuncDic = Get_ApertureTraceFunction(FullCoorindateOfTraceDic,deg=trace_fit_deg)        
    # Now loop through each trace for setting the label
    for o in reversed(sorted(LabelList)):
        boundinside = partial(boundvalue,ll=0,ul=ApertureLabel.shape[0])
        for j in np.arange(ApertureLabel.shape[1]):
            mini,maxi = int(np.rint(boundinside(ApertureTraceFuncDic[o](j)-TraceHWidth))), int(np.rint(boundinside(ApertureTraceFuncDic[o](j)+TraceHWidth+1)))
            ApertureLabel[mini:maxi,j] = o

    if ShowPlot:
        plt.imshow(np.ma.array(ApertureLabel,mask=ApertureLabel==0),cmap='hsv')
        plt.imshow(np.log(ContinuumFile),alpha=0.5)
        plt.colorbar()
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
            CenterXDcoo.append(boundvalue(p[1],np.min(xd),np.max(xd))) # use the boundry values incase the fitted p[1] is outside the bounds 
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
            

def WriteFitsFileSpectrum(FluxSpectrumDic, outfilename, fitsheader=None):
    """ Writes the FluxSpectrumDic into fits image with filename outfilename 
    with fitsheader as header."""
    # First create a 2D array to hold all apertures of the spectrum
    Spectrumarray = np.array([FluxSpectrumDic[i] for i in sorted(FluxSpectrumDic.keys())])
    hdu = fits.PrimaryHDU(Spectrumarray,header=fitsheader)
    hdu.writeto(outfilename)
    return outfilename

#######################################################################################
def parse_str_to_types(string):
    """ Converts string to different object types they represent.
    Supported formats: True,Flase,None,int,float,list,tuple"""
    if string == 'True':
        return True
    elif string == 'False':
        return False
    elif string == 'None':
        return None
    elif string.lstrip('-+ ').isdigit():
        return int(string)
    elif (string[0] in '[(') and (string[-1] in ')]'): # Recursively parse a list/tuple into a list
        return [parse_str_to_types(s) for s in string.strip('()[]').split(',')]
    else:
        try:
            return float(string)
        except ValueError:
            return string
        
        

def create_configdict_from_file(configfilename):
    """ Returns a configuration object by loading the config file """
    Configloader = ConfigParser.SafeConfigParser()
    Configloader.optionxform = str  # preserve the Case sensitivity of keys
    Configloader.read(configfilename)
    # Create a Config Dictionary
    Config = {}
    for key,value in Configloader.items('processing_settings'):
        Config[key] = parse_str_to_types(value)
    for key,value in Configloader.items('extraction_settings'):
        Config[key] = parse_str_to_types(value)
    return Config

def parse_args():
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Spectral Extraction Tool")
    parser.add_argument('SpectrumFile', type=str,
                        help="The 2D Spectrum image fits file")
    parser.add_argument('ConfigFile', type=str,
                         help="Configuration file which contains settings for extraction")
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
    Config = create_configdict_from_file(args.ConfigFile)

    SpectrumFile = args.SpectrumFile
    OutputFile = args.OutputFile

    # Override the Config file with command line arguments
    if args.ContinuumFile is not None:
        Config['ContinuumFile'] = args.ContinuumFile
    if args.FlatNFile is not None:
        Config['FlatNFile'] = args.FlatNFile
    if args.BadPixMask is not None:
        Config['BadPixMask'] = args.BadPixMask
    if args.ApertureLabel is not None:
        Config['ApertureLabel'] = args.ApertureLabel

    if os.path.isfile(OutputFile):
        print('WARNING: Output file {0} already exist'.format(OutputFile))
        print('Skipping this image extraction..')
        sys.exit(1)

    print('Extracting {0}..'.format(SpectrumFile))
    fheader = fits.getheader(SpectrumFile)

    ApertureTraceFilename = Config['ContinuumFile']+'_trace.pkl'
    if os.path.isfile(ApertureTraceFilename):
        print('Loading existing trace coordinates {0}'.format(ApertureTraceFilename))
        ApertureCenters = pickle.load(open(ApertureTraceFilename,'rb'))
    else:
        # First Load/Create the Aperture Label file to label and extract trace form continuum file
        ###########
        if os.path.isfile(str(Config['ApertureLabel'])):
            ApertureLabel = np.load(Config['ApertureLabel'])
        else:
            # ApertureLabel = CreateApertureLabelByThresholding(Config['ContinuumFile'],BadPixMask=Config['BadPixMask'],bsize=51,offset=0,minarea=2000, ShowPlot=True,DirectlyEnterRelabel= True)
            ApertureLabel, ApertureCenters_Trace1 = CreateApertureLabelByXDFitting(Config['ContinuumFile'],BadPixMask=Config['BadPixMask'],startLoc=None,avgHWindow=21,TraceHWidth=5,dispersion_Xaxis=Config['dispersion_Xaxis'],extrapolate_thresh=0.4, extrapolate_order=2, ShowPlot=True, return_trace=True) 
            # Save the aperture label if a non existing filename was provided as input
            if isinstance(Config['ApertureLabel'],str):
                np.save(Config['ApertureLabel'],ApertureLabel)
        ###########
        
        # Trace the center of the apertures
        ApertureCenters = FitApertureCenters(Config['ContinuumFile'],ApertureLabel,apwindow=(-7,+7),
                                             dispersion_Xaxis = Config['dispersion_Xaxis'], ShowPlot=False)
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
    if Config['FlatNFile'] is not None:
        print('Doing Flat correction..')
        NFlat = fits.getdata(Config['FlatNFile'])
        SpectrumImage /= NFlat
        fheader['FLATFIL'] = (os.path.basename(Config['FlatNFile']), 'Flat Filename')
        fheader['HISTORY'] = 'Flat fielded the image'

    # Apply bad pixel correction
        print('Doing Bad pixel correction..')
        BPMask = fits.getdata(BadPixMaskFile)
        SpectrumImage = fix_badpixels(SpectrumImage,BPMask)
    if Config['BadPixMask'] is not None:
        fheader['BPMASK'] = (os.path.basename(Config['BadPixMask']), 'Bad Pixel Mask File')
        fheader['HISTORY'] = 'Fixed the bad pixels in the image'
        # Also use cosmic_lacomic to fix any spiky CR hits in data
        SpectrumImage , cmask = cosmicray_lacosmic(SpectrumImage,sigclip=5,gain=fheader['EXPLNDR'])
        fheader['LACOSMI'] = (np.sum(cmask),'Number of CR pixel fix by L.A. Cosmic')
        
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
