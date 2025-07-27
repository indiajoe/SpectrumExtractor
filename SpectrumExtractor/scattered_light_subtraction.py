#!/usr/bin/env python
""" This module is to subtract scattered background light in the 2D image """
import sys
import argparse
import logging
import numpy as np
from astropy.io import fits
import astropy.stats
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import scipy.stats
import skimage.measure


def set_ref_pixels(img_array,value=0):
    """ Returns the imgArray with al the reference pixel set to value """
    new_img_array = img_array.copy()
    new_img_array[:4,:] = value
    new_img_array[-4:,:] = value
    new_img_array[:,:4] = value
    new_img_array[:,-4:] = value
    return new_img_array


def get_badpixel_mask_laplace_thresholding(image,lower_thresh_percentile=0.1,upper_thresh_percentile=99.9,
                                           bad_pixel_mask=None,do_plot=False):
    """ Returns new badpixels by percentile thresholding in laplace space
    Parameters
    ----------
    image: numpy 2D array 
           Input image
    lower_thresh_percentile : float (default: 0.1)
                              lower percentile below which the pixels should be marked badpixels in the lapace(image)
    upper_thresh_percentile :float (default: 99.9)
                              upper percentile above which the pixels should be marked badpixels in the lapace(image)
    bad_pixel_mask : numpy bool 2D array 
                     Array of known badpixels set to True

    Returns
    -------
    bad_pix_in_image_mask: numpy bool 2D array 
                         Array of newly identified badpixels set to True
    """
    laplace_img = scipy.ndimage.laplace(image)
    l_thresh = np.nanpercentile(laplace_img[~bad_pixel_mask.astype(bool)],lower_thresh_percentile)
    u_thresh = np.nanpercentile(laplace_img[~bad_pixel_mask.astype(bool)],upper_thresh_percentile)
    logging.info('Lower and upper thresholds in lapace image {0}, {1}'.format(l_thresh,u_thresh))
    bad_pix_in_image_mask = (laplace_img < l_thresh) | (laplace_img > u_thresh) | ~np.isfinite(laplace_img)
    logging.info('No of new bad pixels identified by laplace threshold of image: {0}'.format(np.sum(bad_pix_in_image_mask)))
    if do_plot :
        plt.imshow(laplace_img,vmin=np.percentile(laplace_img,1),vmax=np.percentile(laplace_img,99))
        plt.title('Laplace image')
        plt.colorbar()
        plt.show()
    return bad_pix_in_image_mask

def create_locally_scaled_bkg_template(input_image,template_image,background_pixel_mask,kernel,do_plot=False):
    """ Creates a locally scaled background template which can be subtracted from the image
    Parameters
    ----------
    input_image: numpy 2D array 
                 input image
    template_image: numpy 2D array 
                 The high signal scattered light template image 
    background_pixel_mask : numpy bool 2D array 
                 Array of Background pixels set to True
    kernel : numpy 2D array 
             Weighting kernel for smoothing the ratio
             Example for HPF HR mode
             # Create a Weight kernal of rectangular box of size (31,11)  [31 is in XD direction and 11 in dispersion direction].
             # The weights are a product of a XD Gaussian with 5 pix sigma, and a D direction Gaussian with 1.3 pixel sigma. 
             # (1.3 pixel sigma comes from the HR mode resolution, this will have to be changed for HE mode data)
             kernel = np.tile(scipy.stats.norm.pdf(np.arange(-5,6),0,1.3),[31,1]) * scipy.stats.norm.pdf(np.arange(-15,16),0,5)[:,np.newaxis]
               
    Returns
    -------
    scaled_template: numpy 2D array 
                 The locally scaled scattered light `template_image` to match `input_image`
    scale_img : numpy 2D array 
                 The scaling function applied to create the `scaled_template`
    """

    # Ratio image for the scaling between image and Template
    ratio_image = input_image/template_image


    if do_plot:
        plt.imshow(ratio_image*background_pixel_mask.astype(float),vmin=np.percentile(ratio_image,5),vmax=np.percentile(ratio_image,95))
        plt.title('Raw Ratio of Image and Template in Bkg region')
        plt.colorbar()
        plt.show()

    # Any bad pixel not masked or CR hit will significantly affect the weighed averaging of the scale image.
    # Hence we shall du a lapacian filtering of the Bkg region to mask out spurious values

    bad_pix_in_ratio_mask = get_badpixel_mask_laplace_thresholding(ratio_image,lower_thresh_percentile=0.1,upper_thresh_percentile=99.9,
                                                                   bad_pixel_mask=~background_pixel_mask,do_plot=do_plot)

    # Create the new background region mask by removing the newly found bad pixels
    clean_bkg_mask = (~bad_pix_in_ratio_mask) & background_pixel_mask.astype(bool)
    clean_bkg_mask = clean_bkg_mask.astype(float)
    if do_plot:
        plt.imshow(clean_bkg_mask)
        plt.title('Cleaned final background region')
        plt.show()


    # We shall use convolution to calulte neumerator and denominator of weighted average 
    smooth_scatter_bkg = scipy.ndimage.convolve(ratio_image*clean_bkg_mask,kernel) # Numerator
    smooth_scatter_bkg_norm = scipy.ndimage.convolve(clean_bkg_mask,kernel) # Denominator

    scale_img = smooth_scatter_bkg / smooth_scatter_bkg_norm

    # Set reference pixels scaling to 1
    scale_img = set_ref_pixels(scale_img,value=1)

    if do_plot:
        plt.imshow(scale_img,vmin=np.percentile(scale_img,5),vmax=np.percentile(scale_img,95))
        plt.title('Scaling image for Template')
        plt.colorbar()
        plt.show()
    
    scaled_template = template_image * scale_img

    return scaled_template, scale_img


def get_clean_bkg_mask(badpix_mask_file,aper_mask_file,bkg_erosion_iter=1,
                       extra_dilation_label_dic=None,do_plot=False):
    """
    Returns clean background mask for the instrument
    Parameters
    ----------
    badpix_mask_file : str of the .fits file, 2D numpy array
                     Badpixel mask filename
    aper_mask_file : str of the .npy file, 2D numpy array
                     Aperture Mask filename. Background region should be labelled by 0 in this.
    bkg_erosion_iter: int (default: 1)
                     Number of the iterations of erosion to do to the background region mask
    extra_dilation_label_dic: dict {label: niter,...}  (default: None)
                     Dictionary containing the label regions in the `aper_mask_file` and the number of iterations 
                     of dilation to be performed on that region to remove those pixels from the background

    Returns
    -------
    clean_bkg_mask: numpy 2D array
    
    """
    if isinstance(badpix_mask_file,str):
        bad_pixel_mask = fits.getdata(badpix_mask_file)
    else:
        bad_pixel_mask = badpix_mask_file
    if isinstance(aper_mask_file,str):
        aper_mask = np.load(aper_mask_file)
    else:
        aper_mask = aper_mask_file

    # Load the bad pixel mask and aperture mask
    bkg_mask = aper_mask == 0   # Creat the background region mask


    # Create a safer bkg mask by eroding the region of pixels
    safe_bkg_mask = scipy.ndimage.binary_erosion(bkg_mask,iterations=bkg_erosion_iter)

    if extra_dilation_label_dic is not None:
        # Dilate and add the extra labels/traces
        extra_mask = np.sum([scipy.ndimage.binary_dilation(aper_mask == i,iterations=n) for i,n in extra_dilation_label_dic.items()],axis=0).astype(bool)
        # Combine the dialated extra mask to safe_bkg_mask
        safe_bkg_mask = (safe_bkg_mask & ~extra_mask)

    # Remove the bad pixels in the Background region
    clean_bkg_mask = safe_bkg_mask.astype(float)* bad_pixel_mask

    if do_plot:
        plt.imshow(clean_bkg_mask)
        plt.title('Clean Background mask')
        plt.show()

    return clean_bkg_mask

###################################################################################################
        
def scale_and_subtract_template(input_image_file,template_image_file,output_image_file=None,
                                badpix_mask_file=None,aper_mask_file=None,append_model=False,do_plot=True):
    """
    Scale and Subtract template to remove the scattered background light
    Parameters
    ----------
    input_image_file :str of the .fits file
                      input filename
    template_image_file :str of the .fits file
                      scattered light template file
    output_image_file : str (optional)
                      outputfile to write

    Returns
    -------
    output_image : output hdulist object 
    """
    if badpix_mask_file is None:
        badpix_mask_file = 'HPFmask_liberal_v2.fits'
    if aper_mask_file is None:
        aper_mask_file = 'ApertureLabel_HPF_HR_28Nov2017.npy'
    # To remove bright star contamination. dilate the star fiber 3 more times
    extra_dilation_label_dic = {4 + (i*3): 3 for i in range(28)}
    # Before passing on the badpixel mask, remove all the reference pixels from the edges by setting them as badpixels
    # Get a clean mask for background pixels
    clean_bkg_mask = get_clean_bkg_mask(set_ref_pixels(fits.getdata(badpix_mask_file),value=0),
                                        aper_mask_file,
                                        extra_dilation_label_dic=extra_dilation_label_dic)

    # Create a Weight kernel of rectangular box of size (31,11)  [31 is in XD direction and 11 in dispersion direction].
    # The weights are a product of a XD Gaussian with 5 pix sigma, and a D direction Gaussian with 1.3 pixel sigma. 
    # (1.3 pixel sigma comes from the HR mode resolution, this will have to be changed for HE mode data)

    kernel = np.tile(scipy.stats.norm.pdf(np.arange(-5,6),0,1.3),[31,1]) * scipy.stats.norm.pdf(np.arange(-15,16),0,5)[:,np.newaxis]

    input_image = fits.getdata(input_image_file)
    template_image = fits.getdata(template_image_file)

    scaled_template, scale_img = create_locally_scaled_bkg_template(input_image,template_image,
                                                                    clean_bkg_mask,kernel,do_plot=do_plot)
    output_image = input_image - scaled_template
    if output_image_file is not None:
        # Create and write the scaled Template subtracted image
        with fits.open(input_image_file) as hdulist:
            hdulist[0].data = output_image
            hdulist[0].header['HISTORY'] = 'Scaled and Subtracted {0}'.format(template_image_file)
            if append_model:
                hdu_model = fits.ImageHDU(scaled_template,name='BkgModel')
                hdulist.append(hdu_model)
            hdulist.writeto(output_image_file)
        return output_image_file
    else:
        return output_image

#########################################################################################################
def scale_fiberflux_and_subtract_template(input_image_file,template_image_file,fiberaper_mask_file,
                                          output_image_file=None,poly_deg=0,
                                          append_model=False,do_plot=True):
    """
    Scale the fiber flux and Subtract template to remove the scattered background light
    Parameters
    ----------
    input_image_file :str of the .fits file or numpy array
                      input filename
    template_image_file :str of the .fits file or numpy array
                      scattered light template file
    fiberaper_mask_file: str of the .fits file or numpy array
                      fiber aperture mask to use for scaling the template to input image
    output_image_file : str (optional)
                      outputfile to write
    poly_deg : int
              Degress of the polynomial to use to scale the template in cross disperion axis (=0)

    Returns
    -------
    output_image : output hdulist object 
    """

    if isinstance(input_image_file,str):
        input_image = fits.getdata(input_image_file)
    else:
        input_image = input_image_file

    if isinstance(template_image_file,str):
        template_image = fits.getdata(template_image_file)
    else:
        template_image = template_image_file

    if isinstance(fiberaper_mask_file,str):
        fiberaper_mask = fits.getdata(fiberaper_mask_file).astype(bool)
    else:
        fiberaper_mask = fiberaper_mask_file

    # Calculate the ratio image
    ratio_image = input_image/template_image
    
    # Cross dispersion coordinates
    icoords = np.meshgrid(np.arange(ratio_image.shape[0]),np.arange(ratio_image.shape[1]))[0].T[fiberaper_mask]
    # ratio values
    rvalues = ratio_image[fiberaper_mask]
    
    # Fit polynomial to the ratio values
    scale_p = np.polyfit(icoords,rvalues,poly_deg)

    scale_vector = np.polyval(scale_p,np.arange(ratio_image.shape[0]))
    # Clip any negative values or factors above 2 times the 99 percentile ratio
    scale_vector = np.clip(scale_vector,a_min=0,a_max=2*np.percentile(rvalues,99))

    scaled_template = template_image*scale_vector[:,np.newaxis]

    output_image = input_image - scaled_template

    if do_plot:
        plt.imshow(scaled_template)
        plt.colorbar()
        plt.title('Scattered Light Model')
        plt.show()

    if output_image_file is not None:
        if isinstance(input_image_file,str):
            # Create and write the scaled Template subtracted image
            with fits.open(input_image_file) as hdulist:
                hdulist[0].data = output_image
                hdulist[0].header['HISTORY'] = 'Scaled and Subtracted {0}'.format(template_image_file)
                if append_model:
                    hdu_model = fits.ImageHDU(scaled_template,name='BkgModel')
                    hdulist.append(hdu_model)
                hdulist.writeto(output_image_file)
        else:
            # Save the output as a numpy array 
            np.save(output_image_file,output_image)
        return output_image_file
    else:
        return output_image


#########################################################################################################

def create_background_spline_model(input_image, background_pixel_mask,downsample=(8,16),txy=(16,8),do_plot=False):
    """
    Creates a smooth background model based on 2D splines
    Parameters
    ----------
    input_image: numpy 2D array 
                 input image
    background_pixel_mask : numpy bool 2D array 
                 Array of Background pixels set to True
    downsample: two element tuple of int (default : (8,16)
                 Block size to downsample the image with median.
                 If you do not downsample enough, the spline model cannot be fitted.
    txy:   two element tuple of int (default : (16,8)
                  Number of equally spaced spline nodes in each i and j dimension

    Returns
    -------
    scatterd_light_model: numpy 2D array
                Interpolated spline based scattered background model
    scatterd_light_tck : tck output from scipy.interpolate.bisplrep 

    """
    #Downsample the large image array to enable fitting spline
    input_image_downsampled = skimage.measure.block_reduce(np.ma.array(input_image,mask=~background_pixel_mask).filled(np.nan), 
                                                           downsample, func=np.nanmedian)
    #Indices of the downsampled image 
    i, j = np.mgrid[0:input_image.shape[0],0:input_image.shape[1]]  # original indices
    i_ds = skimage.measure.block_reduce(i,downsample, func=np.mean)
    j_ds = skimage.measure.block_reduce(j,downsample, func=np.mean)

    mask = np.isfinite(input_image_downsampled)
    try:
        scatterd_light_tck = scipy.interpolate.bisplrep(i_ds[mask], j_ds[mask], input_image_downsampled[mask],
                                                        tx=np.linspace(0,input_image.shape[0],txy[0]),
                                                        ty=np.linspace(0,input_image.shape[1],txy[1]),task=-1)
    except OverflowError as e:
        logging.warning(e)
        logging.info('Too many points to fit spline. Current downsampling:{0} is not sufficent'.format(downsample))
        logging.info('Downsample the image further')
        return None
    # Now create the interpolated scattered light model
    scatterd_light_model = scipy.interpolate.bisplev(i[:,0],j[0,:],scatterd_light_tck)

    if do_plot:
        plt.imshow(scatterd_light_model)
        plt.colorbar()
        plt.title('Scattered Light Model')
        plt.show()

    return scatterd_light_model, scatterd_light_tck
    
def fit_and_subtract_spline_model(input_image_file,output_image_file=None,downsample=(8,16),txy=(16,8),
                                  badpix_mask_file=None,aper_mask_file=None,niter=0,append_model=False,do_plot=True):
    """
    Scale and Subtract template to remove the scattered background light
    Parameters
    ----------
    input_image_file :str of the .fits file
                      input filename
    output_image_file : str (optional)
                      outputfile to write
    downsample: two element tuple of int (default : (8,16)
                 Block size to downsample the image with median.
                 If you do not downsample enough, the spline model cannot be fitted.
    txy:   two element tuple of int (default : (16,8)
                  Number of equally spaced spline nodes in each i and j dimension
    niter: int (default 0)
                 Number of iterations of sigma clipping to do to remove outliers in final bkg region
    Returns
    -------
    output_image : output hdulist object 
    """
    input_image = fits.getdata(input_image_file)
    if badpix_mask_file is None:
        badpix_mask_file = 'HPFmask_liberal_v2.fits'
    if aper_mask_file is None:
        aper_mask_file = 'ApertureLabel_HPF_HR_28Nov2017.npy'
    # Get a dialtated background mask for HPF
    # To remove bright order nearby contamination. dilate the orders with bright light 3 more times
    aper_mask = np.load(aper_mask_file)
    extra_dilation_label_dic = {i:3 for i in np.sort(np.unique(aper_mask))[1:] if np.nanpercentile(input_image[aper_mask==i],95) > 30}
    # Before passing on the badpixel mask, remove all the reference pixels from the edges by setting them as badpixels
    badpix_mask = set_ref_pixels(fits.getdata(badpix_mask_file),value=0)
    # Get a clean mask for background pixels
    clean_bkg_mask = get_clean_bkg_mask(badpix_mask, aper_mask,
                                        extra_dilation_label_dic=extra_dilation_label_dic)

    # Get a mask to identify bad pixels int he input_image in the background region
    bad_pix_in_bkg_mask = get_badpixel_mask_laplace_thresholding(input_image,lower_thresh_percentile=0.1,
                                                                 upper_thresh_percentile=99.9,
                                                                 bad_pixel_mask=~clean_bkg_mask.astype(bool),do_plot=do_plot)
    # Update the bkg mask
    clean_bkg_mask = clean_bkg_mask.astype(bool) & ~bad_pix_in_bkg_mask

    # Further threshold the background region to remove outliers
    l_flux_thresh = np.percentile(input_image[clean_bkg_mask.astype(bool)],0.01)
    u_flux_thresh = np.percentile(input_image[scipy.ndimage.binary_erosion(clean_bkg_mask.astype(bool),iterations=2)],99.5)
    good_flux_mask = (input_image > l_flux_thresh) & (input_image < u_flux_thresh)
    # Combine to create the final bkg region mask
    final_bkg_mask = clean_bkg_mask.astype(bool) & good_flux_mask

    for i in range(niter+1):
        scatterd_light_model, scatterd_light_tck = create_background_spline_model(input_image, final_bkg_mask,
                                                                                  downsample=downsample,txy=txy,do_plot=do_plot)
        output_image = input_image - scatterd_light_model

        if niter > i: # create a new bkg mask for next iteration
            final_bkg_mask = ~astropy.stats.sigma_clip(np.ma.array(output_image,mask=~final_bkg_mask),sigma=4.).mask
        


    median_offset = np.nanmedian(output_image[final_bkg_mask])
    logging.info('Median offset in Bkg after subtraction = {0}'.format(median_offset))

    # Correct for a pedestal offset to bring the median bkg region pixels to zero
    output_image -= median_offset 

    if do_plot:
        plt.imshow(np.ma.array(input_image,mask=~final_bkg_mask))
        plt.colorbar()
        plt.title('Before Background subtraction')
        plt.show()
        plt.imshow(np.ma.array(output_image,mask=~final_bkg_mask))
        plt.colorbar()
        plt.title('Background subtracted image')
        plt.show()
        _ = plt.hist(output_image[final_bkg_mask],bins=100,alpha=1,label='After Bkg Subtraction')
        _ = plt.hist(input_image[final_bkg_mask],bins=100,alpha=0.5,label='Before Bkg Subtraction')
        plt.legend()
        plt.title('Histogram of Bkg pixels')
        plt.show()
    if output_image_file is not None:
        # Create and write the background subtracted image
        with fits.open(input_image_file) as hdulist:
            hdulist[0].data = output_image
            hdulist[0].header['HISTORY'] = 'Subtracted Bspline Bkg Model minus {0}'.format(median_offset)
            if append_model:
                hdu_model = fits.ImageHDU(scatterd_light_model+median_offset,name='BkgModel')
                hdulist.append(hdu_model)
            hdulist.writeto(output_image_file)
        return output_image_file
    else:
        return output_image

###################################################################################################

def parse_args(raw_args=None):
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Background Subtraction Tool")
    parser.add_argument('image_file', type=str,
                        help="The 2D image fits file for background subtraction")
    # parser.add_argument('ConfigFile', type=str,
    #                      help="Configuration file which contains settings for extraction")
    parser.add_argument('--subtract_spline', action='store_true',
                        help="Subtract spline background model")
    parser.add_argument('--template_file', type=str, default=None,
                        help="Scattered light template file to be used for scaling and subtracting from the data")
    parser.add_argument('--bad_pix_mask', type=str, default=None,
                        help="Bad Pixel Mask fits file to be used for masking bad pixels")
    parser.add_argument('--aperture_label', type=str, default=None,
                        help="Array of labels for the aperture trace regions")
    parser.add_argument('output_file', type=str, 
                        help="Output filename to write background subtracted image")
    parser.add_argument('--do_plot', action='store_true',
                        help="Show plots at each step for debugging")
    parser.add_argument('--append_model', action='store_true',
                        help="Append the background model as a fits extension to the end of output_file fits")
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

    if (args.template_file is not None) and (not args.subtract_spline):
        logging.info('Doing Background subtraction by Template scaling.')
        _ = scale_and_subtract_template(args.image_file,args.template_file,output_image_file=args.output_file,
                                        badpix_mask_file=args.bad_pix_mask,aper_mask_file=args.aperture_label,
                                        append_model=args.append_model,do_plot=args.do_plot)
    elif args.subtract_spline and (args.template_file is None):
        logging.info('Doing Background subtraction by Spline subraction.')
        # downample image by 8 in XD dispertion and 16 in Dispersion direction
        # Since more light variability in XD, we shll use 16 nodes, and only 8 nodes in dispersion direction
        _ = fit_and_subtract_spline_model(args.image_file,output_image_file=args.output_file,downsample=(8,16),txy=(16,8),
                                          badpix_mask_file=args.bad_pix_mask,aper_mask_file=args.aperture_label,niter=1,
                                          append_model=args.append_model,do_plot=args.do_plot)
    else:
        logging.error('Either the --template_file for template subtraction or --subtract_spline for spline subtraction needs to be provided.')

if __name__ == '__main__':
    main()



