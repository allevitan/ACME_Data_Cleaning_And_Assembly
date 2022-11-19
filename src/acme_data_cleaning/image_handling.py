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

__all__ = [
    'map_raw_to_tiles',
    'map_tiles_to_frame',
    'make_saturation_mask',
    'apply_overscan_background_subtraction',
    'process_frame',
    'combine_exposures'
]

#
# Here we define a few key numbers relating to the detector geometry
#

full_tile_height = 487 # The height of each tile, including the overscan.
full_tile_width = 12 # The width of each tile, including the overscan.

num_tiles_per_half = 96 # The number of tiles in each detector half

# This slice extracts the real data from a tile which include the overscan. 
tile_slice = np.s_[...,6:-1,1:-1]

# This is the saturation value in the images that come from the framegrabber.
# The ADC is 16 bits, and the top three bits are always set to "110". The
# detector itself only has 13 bits of dynamic range. Thus, the images
# saturate at "0b1101111111111111, a.k.a. 2^15 + 2^14 + (2^13 - 1)"
saturation_level = 2**15 + 2**14 + 2**13 - 1 


def map_raw_to_tiles(data, fix_that_one_tile=True):
    """Maps images from the raw format output by the fccd to a stack of tiles

    This function accepts either a single frame, or an array of frames.
    
    The tiles produced by this function include the overscan columns which are
    used for exposure-to-exposure background subtraction of each tile.
    They also include an extra row of data at the "seam", which may be useful
    for background subtraction, along with 6 extra rows of data at the top and
    bottom of the detector.

    In other words - this strips out all the regions of the detector which are
    just filled with zeros, but leaves all the non-physical regions, which
    can possibly be used for background subtraction.
    """
    #from matplotlib import pyplot as plt
    #from cdtools.tools import plotting as p
    #p.plot_real(data)
    #plt.show()
    top_half = data[...,:full_tile_height,:]

    # The bottom half is rotated 180 degrees so that the detector seam is
    # always at the bottom. This is useful because detector artifacts tend
    # to propagate toward the seam, and blooming from the saturation also
    # tends to propagate clockwise around the seam, so having a consistent
    # orientation of the tiles is useful.
    bottom_half = data[...,-full_tile_height:,:].flip(-1,-2)

    def stack_tiles(half):
        # The transposes here allow the data to flow properly into the tiles
        # during the reshape
        transposed = half.swapaxes(-1,-2)
        extra_axes = transposed.shape[:-2]
        final_shape = extra_axes + (96,full_tile_width,full_tile_height)        
        stacked = transposed.reshape(final_shape)
        return stacked.swapaxes(-1,-2)

    top_tiles = stack_tiles(top_half)
    bottom_tiles = stack_tiles(bottom_half)

    tiles = t.cat([top_tiles, bottom_tiles], dim=-3)
    
    tiles[...,59,:,:] = tiles[...,59,:,:].roll(-1,dims=-1)
    return tiles


def map_tiles_to_frame(tiles, include_overscan=False):
    """Maps from a stack of tiles to a full, stitched detector frame

    This function accepts either a single frame, or an array of frames.
    
    This function removes all of the overscan regions, so the final image
    reflects the actual geometry of the fccd detector
    """

    # This removes all the extra stuff
    if not include_overscan:
        tiles = tiles[tile_slice]
    top_tiles = tiles[...,:96,:,:]
    bottom_tiles = tiles[...,96:,:,:]

    top_half = t.cat(list(top_tiles.swapaxes(-3,0)), dim=-1)
    bottom_half = t.cat(list(bottom_tiles.swapaxes(-3,0)), dim=-1)
    bottom_half = bottom_half.flip(-1,-2)

    frame = t.cat((top_half, bottom_half), dim=-2)
    #The last thing which needs to be done is a rotation 
    return frame.swapaxes(-1,-2).flip(-2,-1)


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
    reshaped = a.reshape([np.prod(leading_dimensions),1]+final_dimensions)
    expanded_kernel = b.reshape([1,1] + list(b.shape))
    convolved = t.nn.functional.conv2d(reshaped, expanded_kernel,
                                       padding='same')
    output = convolved.reshape(leading_dimensions+final_dimensions)
    return output


def make_saturation_mask(exp_tiles, radius=1, include_wing_shadows=False):
    """Generates a map of saturated pixels in an exposure.

    This function needs to take the non-background-subtracted images, because
    the variable background means that the saturation has a constant value
    before background subtraction, but not after background subtraction.

    Radius defines the size of a convolution element to dilate the mask with,
    which is useful because the saturation tends to affect neighboring pixels.

    Finally, if include_wing_shadows is set to True, the mask will also include
    the regions of the detector which have fake "shadows" due to the saturation
    in the 0th order. These occur two tiles outward from the saturated regions.
    """

    mask = t.zeros(exp_tiles.shape, device=exp_tiles.device)

    # We want to exclude the overscan here, because there are some overscan
    # pixels at the corner points of the tiles which saturate, not because
    # any nearby signal pixels are saturated. If we include those and then
    # dilate the mask, we end up masking off nearby good pixels.
    mask[...,1:-1,1:-1] = exp_tiles[...,1:-1,1:-1] >= saturation_level
    
    kernel = t.ones([radius*2+1,radius*2+1], device=exp_tiles.device)
    mask = convolve2d(mask, kernel) >= 1

    if include_wing_shadows:
        # This also masks off the regions that have the bizarre wing shadows,
        # which show up two tiles outward from saturated regions
        mask[:46] = mask[:46] + mask[2:48]
        mask[50:96] = mask[50:96] + mask[48:96-2]
        mask[96:96+46] = mask[96:96+46:] + mask[96+2:96+48]
        mask[96+50:] = mask[96+50:] + mask[96+48:-2]
        mask = mask >= 1

    return mask

def apply_overscan_background_subtraction(tiles, max_correction=50):
    """Applies a frame-to-frame background correction based on the overscan.

    Thus function uses the first and last column of each tile to estimate a
    variable background correction. The background is estimated as the minimum
    of the two columns, because saturation effects can often cause one or both
    of the columns to become saturated as well. Additionally, a small offset
    of 0.55 is applied, to account for the fact that using a minimum of two
    random variables will slightly bias the estimate of the background.
    The strength of the bias (0.55) was measured empirically.

    Finally, to reduce the noise in the background estimate, a median filter
    is used along the column direction. A median is used rather than a mean,
    or (as was done in previous versions) an exponentially decaying convolution
    because the median filter does better at avoiding "bleed" from saturation
    within the 0th order to outside the 0th order.
    """
    #background_estimate = t.min(tiles, dim=-1)[0]
    background_estimate = t.minimum(tiles[...,:,0], tiles[...,:,11])
    
    med_width = 10 # The median will be calculated over 2*med_width+1 pixels
    
    # This will pad only the last dimension (the dimension along the columns).
    # The padding will alleviate edge effects with the median filter
    pad_shape = (med_width, med_width)   
    padded_bk = t.nn.functional.pad(background_estimate,
                                    pad_shape, mode='replicate')
    
    # This places sequential slices of the background estimate along a new
    # dimension, so the median can be calculated as a single kernel
    unfolded_background = padded_bk.unfold(-1, 2*med_width+1,1)
    background_estimate = t.median(unfolded_background, dim=-1)[0]

    # We apply a maximum to the background estimate, because sometimes saturated
    # pixels will bloom into the overscan, causing the background estimate to
    # become unreasonably large. However, we don't apply a minimum, because
    # saturation can also cause large negative changes to the background which
    # are real, and do show up in the overscan.
    background_estimate = t.clamp(background_estimate, max=max_correction)

    # 0.55 is emperically the difference between the average minimum value
    # of the two outer pixels in a row, and the mean of that row when
    # in a region which is not illuminated. This arises because the minimum is
    # a biased estimator, and this accounts for that bias

    #threshold = 1
    #thing = tiles - background_estimate[...,None]
    #return t.clamp(thing - 0.55, min=threshold) - threshold * (thing < threshold)
    #return t.clamp(tiles - background_estimate[...,None] - 2, min=0)
    return t.clamp(tiles -  background_estimate[...,None] - 0.55, min=-5)


def process_frame(exp, dark, mask=None,
                  max_correction=50,
                  include_wing_shadows=None,
                  include_overscan=False):
    """Processes a single frame from raw data to final result

    This takes the exposure and dark separately to be able to accurately
    determine where the image is saturated.

    Note that while the exposure is expected to be input in "raw" format,
    the dark should be input in "tiles" format to avoid needing to reformat
    the dark with every new experimental frame.    
    """
    tiles = map_raw_to_tiles(exp)
    saturation_mask = make_saturation_mask(tiles)
    mask = saturation_mask if mask is None else mask * saturation_mask
    subtracted = tiles - dark

    cleaned = apply_overscan_background_subtraction(
        subtracted, max_correction=max_correction)

    return (map_tiles_to_frame(cleaned, include_overscan=include_overscan),
            map_tiles_to_frame(mask, include_overscan=include_overscan))

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



