"""Functions for processing images from the fastCCD at 7.0.1.2 of the ALS

Author: Abe Levitan, alevitan@mit.edu

This code is mostly rewritten, but many parts are based on a previous
codebase which was written by Filipe Maia.

All of these functions are designed to accept either single images or stacks
of images, so feel free to batch-process with abandon!
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import jax.numpy as np
import jax
from jax import lax
from jax import scipy
from functools import partial

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


@jax.jit
def map_raw_to_tiles(data):
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
    
    top_half = data[...,:full_tile_height,:]

    # The bottom half is rotated 180 degrees so that the detector seam is
    # always at the bottom. This is useful because detector artifacts tend
    # to propagate toward the seam, and blooming from the saturation also
    # tends to propagate clockwise around the seam, so having a consistent
    # orientation of the tiles is useful.
    bottom_half = data[...,:-full_tile_height-1:-1,::-1]

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

    tiles = np.concatenate([top_tiles, bottom_tiles], axis=-3)
    return tiles


@partial(jax.jit, static_argnames=('include_overscan',))
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

    top_half = np.concatenate(top_tiles.swapaxes(-3,0), -1)
    bottom_half = np.concatenate(bottom_tiles.swapaxes(-3,0), -1)
    bottom_half = bottom_half[...,::-1,::-1]

    return np.concatenate((top_half, bottom_half), axis=-2)


@partial(np.vectorize, signature='(n,m),(k,l)->(n,m)')
def convolve2d(a, b):
    """A wrapper for jax.scipy.convolve to avoid some bugs in jax.

    This expects b to be a 2D convolution kernel, and a is an arbitrarily
    shaped array of at least 2 dimensions.
    
    Side note: I'm not finding jax very enjoyable to program with, I think
    I much prefer pytorch.

    Basically, convolutions with jax.scipy.convolve seem to fail in dimensions
    of 4 or higher, but I need to operate on 4d arrays to do batched
    computations on stacks of images in "tile format". I can get around this
    problem by calling jax.scipy.convolve on 2d arrays, and using jax's
    vectorization to properly broadcast that operation to higher dimensions.

    Note that this method will always use direct convolution, with 'same'
    boundary conditions.
    """
    return scipy.signal.convolve(a, b, mode='same')


@partial(jax.jit, static_argnames=('radius','include_wing_shadows'))
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

    mask = np.zeros(exp_tiles.shape)

    # We want to exclude the overscan here, because there are some overscan
    # pixels at the corner points of the tiles which saturate, not because
    # any nearby signal pixels are saturated. If we include those and then
    # dilate the mask, we end up masking off nearby good pixels.
    mask = mask.at[...,1:-1,1:-1].set(
        exp_tiles[...,1:-1,1:-1] >= saturation_level)

    kernel = np.ones([radius*2+1,radius*2+1])
    mask = convolve2d(mask, kernel) >= 1

    if include_wing_shadows:
        # This also masks off the regions that have the bizarre wing shadows,
        # which show up two tiles outward from saturated regions
        mask = mask.at[:46].set(mask[:46] + mask[2:48])
        mask = mask.at[50:96].set(mask[50:96] + mask[48:96-2])
        mask = mask.at[96:96+46].set(mask[96:96+46:] + mask[96+2:96+48])
        mask = mask.at[96+50:].set(mask[96+50:] + mask[96+48:-2])
        mask = mask >= 1

    return mask

@jax.jit
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
    
    background_estimate = np.minimum(tiles[...,:,0], tiles[...,:,11])

    med_width = 5 # The median will be calculated over 2*med_width+1 pixels

    # These extra zeros are a hack, because lax doesn't have a good way to
    # pad a specific axis, so I need to tell it how much to pad all the axes,
    # which means padding all the earlier axes by 0.
    pad_shape = ( ((0,0),) * (len(background_estimate.shape) - 1)
                  + ((med_width, med_width),))    
    padded_bk = np.pad(background_estimate, pad_shape, 'edge')

    # Again, here I'm just getting the right extra zeros to be used for the
    # lax.dynamic_slice function, because there's not good way to just slice
    # along a specified axis. These will all be used in the loop body.
    extra_dimensions = (0,) * (len(background_estimate.shape) - 1)
    slice_shape = background_estimate.shape[:-1] + (2*med_width+1,)
    extra_slice = ((slice(None,None,None),)
                   * (len(background_estimate.shape) - 1)) 

    def loop_body(r, s):
        relevant_slice = lax.dynamic_slice(
            padded_bk, extra_dimensions + (r,), slice_shape)
        
        s['bk'] = s['bk'].at[...,r].set(np.median(relevant_slice, axis=-1))
        return s

    init_s = dict(bk=np.empty_like(background_estimate))
    s = jax.lax.fori_loop(0, background_estimate.shape[-1], loop_body, init_s)
    background_estimate = s['bk']

    # We apply a maximum to the background estimate, because sometimes saturated
    # pixels will bloom into the overscan, causing the background estimate to
    # become unreasonably large. However, we don't apply a minimum, because
    # saturation can also cause large negative changes to the background which
    # are real, and do show up in the overscan.
    background_estimate = np.minimum(background_estimate, max_correction)

    # 0.55 is emperically the difference between the average minimum value
    # of the two outer pixels in a row, and the mean of that row when
    # in a region which is not illuminated. This arises because the minimum is
    # a biased estimator, and this accounts for that bias
    return np.maximum(tiles -  background_estimate[...,None] - 0.55, 0)


@partial(jax.jit, static_argnames=('mask','include_wing_shadows',
                                   'include_overscan'))
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

@jax.jit
def combine_exposures(frames, masks, exposure_times):
    """Combines a set of exposures of different lengths

    This expects the input to be in the form of arrays with the exposures /
    masks stacked along the first dimension. I.e., it will fail when given
    a list of frames as input.
    
    Each pixel will be a combination of data from all the exposures for which
    that pixel was not masked. The result will be normalized so the intensity
    matches that of the longest exposure.

    The output is both a synthesized frame, and a synthesized mask which
    indicates which pixels were invalid across all exposures.
    
    Any region which is masked off in all exposures will be set to the value
    from the shortest exposure. In the case of ties, the first exposure with
    the minimum time will be used.
    """
    num_trailing_dimensions = len(frames[0].shape)
    inverse_masks = 1 - masks
    exposure_weighted_masks = (inverse_masks.astype(np.float32)
                               * exposure_times.reshape(
                [len(exposure_times),] + [1,]*num_trailing_dimensions))

    total_data = np.sum((inverse_masks) * frames, axis=0)
    total_exposure = np.sum(exposure_weighted_masks, axis=0)

    synthesized_frame = (total_data / total_exposure) * np.max(exposure_times)

    synthesized_mask = np.prod(masks,axis=0)

    # This sets the fully masked off data using the shortest exposure
    inverse_synth_mask = 1 - synthesized_mask
    shortest_idx = np.argmin(exposure_times)
    factor = np.max(exposure_times) / exposure_times[shortest_idx]
    total_exposure = total_exposure.at[inverse_synth_mask].set(
        factor * frames[shortest_idx][inverse_synth_mask])

    return synthesized_frame, synthesized_mask



