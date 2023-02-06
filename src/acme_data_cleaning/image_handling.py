"""Functions for processing images from the fastCCD at 7.0.1.2 of the ALS

Author: Abe Levitan, alevitan@mit.edu

This code is mostly rewritten, but many parts are based on a previous
codebase which was written by Pablo Enfadaque and Stefano Marchesini

All of these functions are designed to accept either single images or stacks
of images, so feel free to batch-process with abandon!
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
import warnings

__all__ = [
    'FastCCDFrameCleaner',
    'combine_exposures',
    'InterpolatingResampler',
    'NonInterpolatingResampler',
    'make_resampler'
]

#
# First we have some generally useful function definitions
#

def convolve2d(a, b):
    """A wrapper for t.nn.functional.convolve to avoid some nonsense

    The nonsense is that convolutions for neural network layers tend to need
    a bunch of bells and whistles, which we don't need here, but to use
    the pytorch convolution you need to set all those explicitly to "1", which
    is annoying to repeat all over the place.

    This expects "b" to be a 2D convolution kernel, and "a" is an arbitrarily
    shaped array of at least 2 dimensions.
    
    Note that this method will always use direct convolution, with 'same'
    boundary conditions.
    """
    leading_dimensions = list(a.shape[:-2])
    final_dimensions = list(a.shape[-2:])
    reshaped = a.reshape([int(np.prod(leading_dimensions)),1]+final_dimensions)
    expanded_kernel = b.reshape([1,1] + list(b.shape))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="Using padding='same'")
        convolved = t.nn.functional.conv2d(reshaped, expanded_kernel,
                                           padding='same')
    output = convolved.reshape(leading_dimensions+final_dimensions)
    return output


#
# Now we have the classes defined for each detector type
#


class FrameCleaner():
    """ A base class to run background subtraction routines on raw frames

    This class should be subclassed for each kind of detector available
    at 7.0.1.2. The basic pattern is that the class is instantiated once
    per each scan, using the available information (the darks, etc.).
    
    Subclasses are required to expose the function process_frame(), which
    accepts a raw frame or stack of raw frames, and outputs a background
    subtracted, processed frame (or stack of frames) and a saturation mask
    (or stack of saturation masks) indicating where the detector was
    saturated
    """
    
    def process_frame():
        raise NotImplementedError(
            'Subclasses of FrameCleaner must implement process_frame')


class FastCCDFrameCleaner(FrameCleaner):

    #
    # Here we define a few key numbers relating to the detector geometry. Note
    # That all the terminology here refers to the orientation where the central
    # seam of the detector runs horizontally.
    #
    
    full_tile_height = 487 # The height of each tile, including the overscan.
    full_tile_width = 12 # The width of each tile, including the overscan.
    
    num_tiles_per_half = 96 # The number of tiles in each detector half
    
    # This slice extracts the real data from a tile which include overscan. 
    # Hi Peter Denes: Yes, I know that the overscan SHOULD be on the last two
    # columns, but it is not. So, David, whenever that gets fixed, this is where
    # you'll need to update the code. There are some other places where things
    # will need to be updated: they're marked with the comment OVERSCANPROBLEMS
    tile_slice = np.s_[...,6:-1,1:-1]
    
    # This is the saturation value in the images coming from the framegrabber.
    # The ADC is 16 bits, and the top three bits are always set to "110". The
    # detector itself only has 13 bits of dynamic range. Thus, the images
    # saturate at "0b1101111111111111, a.k.a. 2^15 + 2^14 + (2^13 - 1)"
    saturation_level = 2**15 + 2**14 + 2**13 - 1     
    
    def __init__(self, darks, fix_that_one_tile=True):
        # This saves the darks, pre-moved to tile format 
        self.darks = \
            tuple(self.map_raw_to_tiles(dark) for dark in darks)
        #self.dark_tiles = self.map_raw_to_tiles(
        #    darks, fix_that_one_tile=fix_that_one_tile)


    def stack_tiles(self, half):
        """Creates a tile stack from a detector half-image, in raw format

        This function exists to be called from map_raw_to_tiles, which
        separately stacks the two halves of the detector into a list of
        tiles
        """
        transposed = half.swapaxes(-1,-2)
        extra_axes = transposed.shape[:-2]
        final_shape = (extra_axes +
                       (self.num_tiles_per_half,
                        self.full_tile_width,
                        self.full_tile_height))
        stacked = transposed.reshape(final_shape)
        return stacked.swapaxes(-1,-2)

    
    def map_raw_to_tiles(self, data, fix_that_one_tile=True):
        """Maps images from the raw format output by the framegrabber to a stack of tiles
        
        This function accepts either a single frame, or an array of frames.
        
        The tiles produced by this function include the overscan columns which
        are used for exposure-to-exposure background subtraction of each tile.
        They also include an extra row of data at the "seam", which may be
        useful for background subtraction, along with 6 extra rows of data\
        at the top and bottom of the detector.
        
        In other words - this strips out all the regions of the detector which
        are just filled with zeros, but leaves all the non-physical regions,
        whichcan possibly be used for background subtraction.
        """
        
        top_half = data[...,:self.full_tile_height,:]
        
        # The bottom half is rotated 180 degrees so that the detector seam is
        # always at the bottom. This is useful because detector artifacts tend
        # to propagate toward the seam, and blooming from the saturation also
        # tends to propagate clockwise around the seam, so having a consistent
        # orientation of the tiles is useful.
        bottom_half = data[...,-self.full_tile_height:,:].flip(-1,-2)


        top_tiles = self.stack_tiles(top_half)
        bottom_tiles = self.stack_tiles(bottom_half)
        
        tiles = t.cat([top_tiles, bottom_tiles], dim=-3)
        
        # OVERSCANPROBLEMS This fixes a single tile, which appears to have it's
        # overscan columns different from the rest. If the overscan ever gets
        # fixed, you can fix that here
        if fix_that_one_tile:
            tiles[...,59,:,:] = tiles[...,59,:,:].roll(-1,dims=-1)
        
        return tiles


    def map_tiles_to_frame(self, tiles, include_overscan=False):
        """Maps from a stack of tiles to a full, stitched detector frame
        
        This function accepts either a single frame, or an array of frames.
        
        By default, this removes all of the overscan regions, so the final image
        reflects the actual geometry of the fccd detector. If you set
        include_overscan=True, it will include all the overscan regions
        
        The output images are laid out so they appear to a viewer as though
        you are looking downstream, from the sample toward the detector.
        """
        
        # This removes all the extra stuff
        if not include_overscan:
            tiles = tiles[self.tile_slice]
        
        top_tiles = tiles[...,:self.num_tiles_per_half,:,:]
        bottom_tiles = tiles[...,self.num_tiles_per_half:,:,:]

        # Look at map_raw_to_tiles to understand what these are doing
        top_half = t.cat(list(top_tiles.swapaxes(-3,0)), dim=-1)
        bottom_half = t.cat(list(bottom_tiles.swapaxes(-3,0)), dim=-1)
        bottom_half = bottom_half.flip(-1,-2)

        frame = t.cat((top_half, bottom_half), dim=-2)
    
        #The last thing which needs to be done is a rotation 
        return frame.swapaxes(-1,-2).flip(-2,-1)



    def make_saturation_mask(self, exp_tiles, radius=1,
                             include_wing_shadows=False):
        """Generates a map of saturated pixels in an exposure.
        
        The actual mask returned is the set of all saturated pixels, binary
        dilated by a convolution element with a radius of (by default) 1.
        
        This function needs to take the non-background-subtracted images,
        because the variable background means that the saturation has a
        constant value before background subtraction, but not after
        
        Radius defines the size of a convolution element to dilate the mask
        with, which is useful because the saturation tends to affect
        neighboring pixels.

        Finally, if include_wing_shadows is set to True, the mask will also
        include the regions of the detector which have fake "shadows" due
        to the saturation in the 0th order. These occur two tiles outward
        from the saturated regions.
        """

        mask = t.zeros(exp_tiles.shape, device=exp_tiles.device)
        
        # We want to exclude the overscan here, because there are some overscan
        # pixels at the corner points of the tiles which saturate, not because
        # any nearby signal pixels are saturated. If we include those and then
        # dilate the mask, we end up masking off nearby good pixels.
        mask[(np.s_[:],) + self.tile_slice] = \
            exp_tiles[(np.s_[:],) + self.tile_slice] >= self.saturation_level
        
        kernel = t.ones([radius*2+1,radius*2+1], device=exp_tiles.device)
        mask = convolve2d(mask, kernel) >= 1
        
        if include_wing_shadows:
            # This also masks off the regions that have the  wing shadows,
            # which show up two tiles outward from saturated regions
            mask[:46] = mask[:46] + mask[2:48]
            mask[50:96] = mask[50:96] + mask[48:96-2]
            mask[96:96+46] = mask[96:96+46:] + mask[96+2:96+48]
            mask[96+50:] = mask[96+50:] + mask[96+48:-2]
            mask = mask >= 1

        return mask


    def apply_overscan_background_subtraction(
            self, tiles, med_width=10, max_correction=50):
        """Applies a frame-to-frame background correction based on the overscan.
        
        Thus function uses the first and last column of each tile to estimate a
        variable background correction. The background is estimated as the min
        of the two columns, because saturation effects can often cause one or
        both of the columns to become saturated as well. Also, a small offset
        of 0.55 is applied, to account for the fact that using a minimum of two
        random variables will slightly bias the estimate of the background.
        The strength of the bias (0.55) was measured empirically.
        
        Finally, to reduce the noise in the background estimate, a median filter
        is used along the column direction. A median is used rather than a mean,
        or (as was done in previous versions) an exponentially decaying
        convolution because the median filter does better at avoiding "bleed"
        from saturation within the 0th order to outside the 0th order.

        The median will be calculated over 2*med_width+1 pixels
        """
        
        # OVERSCANPROBLEMS This would become the last two columns if the
        # overscan problem is fixed
        background_estimate = t.minimum(tiles[...,:,0], tiles[...,:,11])
        
        # This will pad only the last dimension (the dimension along the cols)
        # The padding will alleviate edge effects with the median filter
        pad_shape = (med_width, med_width)   
        padded_bk = t.nn.functional.pad(background_estimate,
                                        pad_shape, mode='replicate')
        
        # This places sequential slices of the background estimate along a new
        # dimension, so the median can be calculated as a single kernel
        unfolded_background = padded_bk.unfold(-1, 2*med_width+1,1)
        background_estimate = t.median(unfolded_background, dim=-1)[0]

        # We apply a max to the background estimate, because sometimes saturated
        # pixels will bloom into the overscan, causing the background estimate
        # to become unreasonably large. However, we don't apply a min, because
        # saturation can also cause large negative changes to the background
        # which are real, and do show up in the overscan.
        background_estimate = t.clamp(background_estimate, max=max_correction)

        # The method above slightly underestimates the background, because we
        # estimate it using the min of the two overscan columns. Because these
        # measurements themselves have noise, the minimum of the two will tend
        # to be less than the true background level. Simply put, the minimum
        # is a biased estimator. We correct for that below
    
        # 0.55 comes from an empirical measurement of that underestimation, done
        # by comparing the estimated background with the mean pixel value in a
        # region which is not illuminated. If the detector changes, then this
        # value would change as well.
        return t.clamp(tiles -  background_estimate[...,None] - 0.55, min=-5)


    def process_frame(self, exp, idx, mask=None,
                      med_width=10,
                      max_correction=50,
                      include_wing_shadows=None,
                      include_overscan=False):
        """Processes a single frame from raw data to final result

        This takes an experimental frame, and the index in the sequence of
        exposures. The index is used to select the correct background image
        for this exposure time
        """
        # TODO: Find a more natural way to match frames to darks than an
        # index. Maybe key it by exposure time?
        
        tiles = self.map_raw_to_tiles(exp)
        saturation_mask = self.make_saturation_mask(tiles)
        mask = saturation_mask if mask is None else mask * saturation_mask
        subtracted = tiles - self.darks[idx]

        cleaned = self.apply_overscan_background_subtraction(
            subtracted, med_width=med_width,  max_correction=max_correction)

        frames = self.map_tiles_to_frame(
            cleaned, include_overscan=include_overscan)

        mask = self.map_tiles_to_frame(
            mask, include_overscan=include_overscan)
        
        return (frames, mask)


def combine_exposures(frames, masks, exposure_times, use_all_exposures=False):
    """Combines a set of exposures of different lengths

    This expects the input to be in the form of arrays with the exposures /
    masks stacked along the first dimension. I.e., it will fail when given
    a list of frames as input.

    If the use_all_exposures flag is set to True, then each pixel will be a
    properly normalized sum of the data from all the exposures for which
    that pixel was not masked. This is appropriate for, for example, single
    photon counting detectors where the detector readout noise is small.

    If the use_all_exposures flag is set to false, each pixel will only contain
    data from the longest exposure for which it was not saturated. This is
    best for the case where detector readout noise is significant, so adding in
    data from a short exposure will simply increase the readout noise. This
    option is selected by default, as the fccd has significant readout noise.

    The output is both a synthesized frame, and a synthesized mask which
    indicates which pixels were invalid across all exposures.
    
    Any region which is masked off in all exposures will be set to zero.
    In the future, I want to use the value from the shortest exposure. In
    the case of ties, the first exposure with the minimum time will be used.
    """
    exposure_times = t.as_tensor(exposure_times,
                                 dtype=frames.dtype, device=frames.device)
    # This sets up the masks that we need if we plan to use all the exposures
    num_trailing_dimensions = len(frames[0].shape)
    inverse_masks = ~masks
    exposure_weighted_masks = (inverse_masks.to(dtype=t.float32)
                               * exposure_times.reshape(
                [len(exposure_times),] + [1,]*num_trailing_dimensions))
    
    # If we don't want to use all the exposures, we modify the masks to
    # just select a single exposure per pixel
    if use_all_exposures==False:
        # This contains the index of the exposure to use for each pixel
        mask_idx = t.argmax(exposure_weighted_masks,axis=0)
        for idx, exp_time in enumerate(exposure_times):
            # And this updates the masks so only the mask at that index
            # is set to nonzero
            mask_at_this_index = (mask_idx == idx)
            exposure_weighted_masks[idx] = exp_time * mask_at_this_index
            inverse_masks[idx] = mask_at_this_index
    
    # Finally, we use these masks to generate the output data
    total_data = t.sum((inverse_masks) * frames, dim=0)
    total_exposure = t.sum(exposure_weighted_masks, dim=0)

    synthesized_frame = (total_data / total_exposure) * t.max(exposure_times)

    # This sets things which were fully masked off to zero instead of nan
    synthesized_frame = t.nan_to_num(synthesized_frame)
    
    synthesized_mask = t.prod(masks,dim=0)

    return synthesized_frame, synthesized_mask


# We need to have two methods for resampling the raw diffraction patterns:
# one method which produces defined pixel sizes in real space, at the cost
# of doing bad things to the data, and the other which treats the data with
# respect but inevitably produces images with a non-idea real space pixel
# size

class InterpolatingResampler():

    def __init__(self, sample_input, center, output_shape, binning_factor):
        """This resamples diffraction patterns to produce arbitrary pixel sizes
        
        This class implements the method used in the cosmicp preprocessor, in
        that it will interpolate the original diffraction pattern to produce an
        output which captures a stable sampling in reciprocal space. This is
        best for focused-probe ptychography, when the speckles are not near the
        pixel size and having easily inrepretable images as a function of energy
        is important.

        The resampling is defined in the following manner:

        The input images to the resampler are interpreted as defining a
        continuous image, which is bilinearly interpolated between the pixels
        of the input. The locations of the input pixels are (0,1,...), so
        asking for output at (0,0) produces exactly the value of the
        (0,0)th pixel.
        
        On top of this continuous image, we place a grid of sampling points.
        This grid is defined so that the zero-frequency pixel (with the
        pixels arranged to be interpreted as an "FFTshifted" array) is placed
        precisely at the "center" location. The zero-frequency pixel is the
        pixel in position output_shape//2 along each dimension. The spacing
        between the sampling points is equal to binning_factor.
        
        As an example, to get the original data from a 960x960 image, use
        a center of (480,480), an output shape of (960,960), and a binning
        factor of 1. 

        One further note is that, before the resampling is done, the image is
        convolved with a uniform element of shape floor(binning_factor) by
        floor(binning_factor). This allows information from pixels that are
        not near any sampling point to propagate into the final image.

        As a precaution, note that because we choose to prioritize the
        central location as the zero-frequency pixel of the output array, the
        edges of the sliced location can vary slightly from how you may expect.
        For example, a binning factor of 3, and an output shape of (320,320),
        requires a center of (481,481) to match what you would get by simply
        downsampling the original image by a factor of 3. With a center of
        (480,480), it will miss one line of pixels on the right, and include
        a line outside the detector on the left.
        
        Finally, the interpolation will happily include information from
        outside the detector area (below 0, and above input_shape-1). In this
        case, the data itself is set to zero, and the resampled masks will
        indicate that the region of fake data should be masked off.        
        """

        # We get the key info from the sample input
        input_shape = sample_input.shape[-2:]
        
        # The convolution kernel can only have an integer size, so we set it
        # To the floor of the binning factor (but never less than 1!)
        self.conv_kernel = t.ones(
            (int(np.maximum(np.floor(binning_factor), 1)),) * 2,
            dtype=sample_input.dtype, device=sample_input.device)

        # This makes the convolution a mean instead of a sum
        self.conv_kernel /= np.prod(self.conv_kernel.shape)

        # Just needed to make ptyorch happy
        self.output_shape = tuple(t.as_tensor(val) for val in output_shape)

        # First we get the locations of the sampling points,
        # in the coordinate system described in the docstring
        i = (t.arange(self.output_shape[0], device=sample_input.device)
             - self.output_shape[0] // 2) * binning_factor + center[0]
        j = (t.arange(self.output_shape[0], device=sample_input.device)
              - self.output_shape[0] // 2) * binning_factor + center[1]
        I, J = t.meshgrid((i,j), indexing='ij')

        # And we convert to the coordinate system expected by grid_sample
        I = (I + 0.5) * 2 / input_shape[0] - 1
        J = (J + 0.5) * 2 / input_shape[1] - 1
        self.locations = t.stack((J,I), dim=-1)

        
    def resample(self, images, masks=None):
        """Applies the resmpling operation defined by this object

        This can either accept an image (or stack of images) to resample,
        or it can also accept a mask (or stack of masks) to resample using the
        same operation. If the masks are resampled, the output masks are
        defined to mask off all pixels which contain information from masked
        pixels on the underlying detector.
        """
        
        # This just helps to deal with some quirks of pytorch
        leading_dimensions = images.shape[:-2]
        final_dimensions = images.shape[-2:]

        convolved_images = convolve2d(images, self.conv_kernel)

        n_leading_channels = int(np.prod(leading_dimensions))

        # Now we reshape the data and locations, to make pytorch happy
        reshaped_input = convolved_images.reshape(
            [n_leading_channels,1] + list(final_dimensions))
        # This doesn't actually touch the underlying memory, so it should
        # be a pretty efficient approach.
        reshaped_locations = self.locations.expand(
            (n_leading_channels,-1,-1,-1))

        # And we finally do the resampling
        # Should we be aligning corners??
        resampled = t.nn.functional.grid_sample(reshaped_input,
                                                reshaped_locations,
                                                align_corners=False,
                                                padding_mode='zeros')

        # Before converting back to the original shape        
        output = resampled.reshape(leading_dimensions+tuple(self.output_shape))

        if masks is None:
            return output

        # If we have masks, we process the masks now
        leading_dimensions = masks.shape[:-2]
        final_dimensions = masks.shape[-2:]
        
        float_masks = masks.to(dtype=self.conv_kernel.dtype)
        convolved_masks = convolve2d(float_masks, self.conv_kernel)
        
        n_leading_channels = int(np.prod(leading_dimensions))
        
        # Now we reshape the data and locations, to make pytorch happy
        reshaped_masks = convolved_masks.reshape(
            [n_leading_channels,1] + list(final_dimensions))
        
        # This doesn't actually touch the underlying memory, so it should
        # be a pretty efficient approach.
        reshaped_locations = self.locations.expand(
            (n_leading_channels,-1,-1,-1))
        
        # And we finally do the resampling
        # TODO: mask off all unphysical regions of the detector!
        resampled_masks = t.nn.functional.grid_sample(reshaped_masks,
                                                      reshaped_locations,
                                                      align_corners=True,
                                                      padding_mode='zeros')
        
        # Before converting back to the original shape
        output_masks = resampled_masks.reshape(
            leading_dimensions+tuple(self.output_shape))
        
        
        output_masks = (output_masks != 0).to(dtype=masks.dtype)
        
        return output, output_masks

        

class NonInterpolatingResampler():

    def __init__(self, sample_input, center, output_shape, binning_factor):
        """This resamples diffraction patterns  without interpolating

        This class implements a simple method for resampling images which
        is less flexible than the interpolating resampler but treats the
        raw data more respectfully. If your diffraction pattern has structure
        close to the pixel size, for example, using the interpolating
        resampler can have undesireable effects which this resampler will
        avoid.
        
        The binning_factor must be an integer value, so we set it to
        floor(binning_factor) in order to match the behavior of
        InterpolatingResampler.

        With n = output_shape, and bin=binning_factor,
        this method will select the pixels in the range from:

        [ center - n // 2 * bin - (bin-1) // 2
         : center - n // 2 * bin + n * bin - (bin-1) // 2]

        along each dimension. In other words, the pixel indicated by "center"
        will wind up as the zero-frequency pixel of an "FFTshifted" style array.
        As the binning factor increases, this pixel will also stay centered
        within the binned zero frequency pixel.

        This output will match the output provided by InterpolatingResampler
        given the same input, provided that all the inputs are integers.
        However, InterpolatingResampler will introduce numerical error due
        to the resampling step, and risks corrupting the data unintentionally
        when non-integer inputs are given.
        
        As a precaution, note that because we choose to prioritize the
        central location as the zero-frequency pixel of the output array, the
        edges of the sliced location can vary slightly from how you may expect.
        For example, a binning factor of 3, and an output shape of (320,320),
        requires a center of (481,481) to match what you would get by simply
        downsampling the original image by a factor of 3. With a center of
        (480,480), it will miss one line of pixels on the right, and include
        a line outside the detector on the left.

        """
        # We get the key info from the sample input
        input_shape = sample_input.shape[-2:]

        self.binning_factor = int(binning_factor) # floor

        # Just needed to make ptyorch happy
        self.output_shape = tuple(t.as_tensor(val) for val in output_shape)

        n_pix = tuple(n for n in self.output_shape)
        bin_fact = self.binning_factor
        
        limits = tuple((c - n // 2 * bin_fact - (bin_fact-1)//2,
                        c - (n // 2 - n ) * bin_fact - (bin_fact-1)//2)
                       for c, n in zip(center, n_pix))

        required_pads = tuple((np.maximum(-lim[0], 0),
                               np.maximum(lim[1] - in_n, 0))
                              for lim, in_n in zip(limits, input_shape))
        

        pad = np.maximum(*tuple(np.maximum(*p) for p in required_pads))
        self.pad = (pad,)*4
        
        self.sl = np.s_[...,
                        limits[0][0] + pad : limits[0][1] +  pad,
                        limits[1][0] + pad : limits[1][1] + pad ]

        
    def resample(self, images, masks=None):
        """Applies the resmpling operation defined by this object

        This can either accept an image (or stack of images) to resample,
        or it can also accept a mask (or stack of masks) to resample using the
        same operation. If the masks are resampled, the output masks are
        defined to mask off all pixels which contain information from masked
        pixels on the underlying detector.
        """

        # We pad the images with enough zeroes to cover the requested area
        padded_images = t.nn.functional.pad(images, self.pad,
                                            mode='constant', value=0)

        # We extract the asked-for slice
        sliced_images = padded_images[self.sl]

        # This just helps to deal with some quirks of pytorch
        leading_dimensions = sliced_images.shape[:-2]
        final_dimensions = sliced_images.shape[-2:]

        n_leading_channels = int(np.prod(leading_dimensions))

        # Now we reshape the data and locations, to make pytorch happy
        reshaped_input = sliced_images.reshape(
            [n_leading_channels,1] + list(final_dimensions))
                
        # And then we do the binning
        output = t.nn.functional.avg_pool2d(reshaped_input, self.binning_factor)

        # Before converting back to the original shape        
        output = output.reshape(leading_dimensions+tuple(self.output_shape))

        
        if masks is None:
            return output

        # Now, we do the masks

        # TODO: update the masks so they actually report saturation vs
        # bad pixels, and set the mask in the padded region to 0x00000200,
        # meaning pixel is missing. Saturation is 0x00000002.
        # We pad the mask with enough zeros to cover the requested area
        padded_masks = t.nn.functional.pad(masks, self.pad,
                                           mode='constant', value=0)

        # We extract the asked-for slice
        sliced_masks = padded_masks[self.sl]

        # This just helps to deal with some quirks of pytorch
        leading_dimensions = sliced_masks.shape[:-2]
        final_dimensions = sliced_masks.shape[-2:]

        n_leading_channels = int(np.prod(leading_dimensions))

        # We make the masks floats, so pytorch is happy
        float_masks = sliced_masks.to(dtype=images.dtype)
        # Now we reshape the data and locations, to make pytorch happy
        reshaped_masks = float_masks.reshape(
            [n_leading_channels,1] + list(final_dimensions))
                
        # And then we do the binning
        output_masks = t.nn.functional.avg_pool2d(
            reshaped_masks, self.binning_factor)

        # Before converting back to the original shape        
        output_masks = output_masks.reshape(
            leading_dimensions + tuple(self.output_shape))

        output_masks = (output_masks != 0).to(dtype=masks.dtype)
        
        return output, output_masks
        

def make_resampler(metadata, config, dummy_im):
    """Makes the appropriate resampler

    It chooses the resampler and sets the parameters based on the metadata,
    config, and a dummy image
    """

    output_shape = (config['output_pixel_count'],)*2
    
    if config['interpolate']:
        # We define an interpolating resampler
        
        hc = 1.986446e-25 # in Joule-meters
        energy = metadata['energy'] * 1.60218e-19 # convert to Joules
        wavelength = hc / energy
        pixel_pitch = metadata['geometry']['psize']
        det_distance = metadata['geometry']['distance']
        
        # In this case, we define the binning_factor using the
        # output_pixel_size
        binning_factor = ( wavelength * det_distance / 
                           ( output_shape[0] * pixel_pitch
                             * (config['output_pixel_size'] * 1e-9)))

        # We have to remember to update the pixel size for the output
        metadata['geometry']['psize'] *= float(binning_factor)
        if 'basis_vectors' in metadata['geometry']:
            metadata['geometry']['basis_vectors'] = \
                (np.array(metadata['geometry']['basis_vectors'])
                 * float(binning_factor)).tolist()
        
        return InterpolatingResampler(
            dummy_im, config['center'], output_shape,
            binning_factor)
    else:
        # We define a non-interpolating resampler
        
        # In this case, we ignore the output_pixel_size and just use the
        # manually defined binning factor
        metadata['geometry']['psize'] *= float(config['binning_factor'])
        
        if 'basis_vectors' in metadata['geometry']:
            metadata['geometry']['basis_vectors'] = \
                (np.array(metadata['geometry']['basis_vectors'])
                 * float(config['binning_factor'])).tolist()
                 
        return NonInterpolatingResampler(
            dummy_im, config['center'], output_shape,
            config['binning_factor'])
        

