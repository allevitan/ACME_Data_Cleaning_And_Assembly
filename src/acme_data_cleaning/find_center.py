"""A script to process .stxm files to .cxi files

Author: Abe Levitan, alevitan@mit.edu
"""
import sys
import argparse
import numpy as np
import torch as t
import h5py
import glob
from acme_data_cleaning import image_handling, file_handling, config_handling



def process_file(stxm_file, config, threshold, default_mask=None):
    """Processes a single stxm file to .cxi
    """
    
    # We first read the metadata and translations from the stxm file
    metadata = file_handling.read_metadata_from_stxm(stxm_file)
    
    # Next we extract the key parameters from the metadata that we need to know
    # to generate the .cxi file
    # TODO: Set up a system that can flexibly do N exposures
    if metadata['double_exposure']:
        n_exp_per_point=2
        # Note: Old files have dwell1 meaning the second exposure and
        # dwell2 meaning the first exposure
        exposure_times = np.array([metadata['dwell2'],
                                   metadata['dwell1']])
        if not config['succinct']:
            print('File uses double exposures with exposure times',
                  exposure_times)
    else:
        n_exp_per_point=1
        exposure_times = np.array([metadata['dwell1']])
        if not config['succinct']:
            print('File uses single exposures')
    
    # This gets a list of mean dark patterns, one per exposure time
    darks = file_handling.read_mean_darks_from_stxm(
        stxm_file, n_exp_per_point=n_exp_per_point, device=config['device'])
    

    # Now we instantiate a frame cleaner object and a resampling object.
    # By instantiating them here, instead of at each frame, we can do some
    # steps which are common to each frame ahead of time. E.g, transforming
    # the dark frames to the right format, or storing the sampling positions
    # for the interpolating resampler.

    frame_cleaner = image_handling.FastCCDFrameCleaner(darks)
    
    # This is used to help instantiate the resampling object
    dummy_im = frame_cleaner.process_frame(
        darks[0], 0, include_overscan=False)[0]
    

    # Now we actually open the file, and we will write to it as we
    # process the data.
    mask = default_mask.to(device=dummy_im.device)
    
    # Read the number of frames
    n_frames = file_handling.get_n_frames_from_stxm(
        stxm_file, n_exp_per_point=n_exp_per_point)

    # This just gets an iterator, it will wait to actually load the
    # images until it's iterated over
    chunked_exposures = file_handling.read_chunked_exposures_from_stxm(
        stxm_file, chunk_size=config['chunk_size'],
        n_exp_per_point=n_exp_per_point, device=config['device'])

    im = None
    norm = 0
    for idx, exps in enumerate(chunked_exposures):
        if not config['succinct']:
            print('Processing frames', idx*config['chunk_size']+1,
                  'to', min(n_frames, (idx+1)*config['chunk_size']),
                  'of', n_frames, end='\r')

        # index here is the index of the frame within the exposure, but idx
        # refers to the index of the chunk we're dealing with
        cleaned_exps, masks = zip(
            *(frame_cleaner.process_frame(
                exp, index, include_overscan=False,
                med_width=config['background_median_width'],
                max_correction=config['max_overscan_correction'],
                min_correction=config['min_overscan_correction'],
                background_offset=config['background_offset'],
                cut_zeros=config['cut_zeros'])
              for index, exp in enumerate(exps)))
            
        # Because combine_exposures works with an arbitrary number of
        # exposures, we just always use it, and avoid needing a separate
        # case for the single and double exposure processing.
        synthesized_exps, synthesized_masks = \
            image_handling.combine_exposures(
                t.stack(cleaned_exps), t.stack(masks), exposure_times)

        norm += synthesized_exps.shape[0]
        if im is None:
            im = t.sum(synthesized_exps, dim=0)
        else:
            im += t.sum(synthesized_exps, dim=0)
                        
    print('Finished processing                                          ')
    im = im / norm
    im = im * (1 - mask)
    im = im.cpu().numpy()

    probe_mask = im > threshold
    i = np.arange(probe_mask.shape[0])
    j = np.arange(probe_mask.shape[1])
    i,j = np.meshgrid(i,j, indexing='ij')
    norm = np.sum(probe_mask)

    center = [np.sum(i*probe_mask) / norm,
              np.sum(j*probe_mask) / norm]

    return center, im, probe_mask

    
def main(argv=sys.argv):

    # This argument reading setup is a bit unusual. The idea is that all the
    # arguments will have default values, which are stored in the configuration
    # file. If any are overridden by being explicitly set, then that will
    # override the values in the configuration file
    
    # We parse the input command line args here
    parser = argparse.ArgumentParser('Finds a good center point for diffraction patterns, to use in the preprocessor')
    parser.add_argument('stxm_file', type=str, help='The .stxm file to use for the calibration, ideally a short series of exposures from a blank area on the sample')
    parser.add_argument('--threshold', '-t', type=float, help='Pixels with intensity above this threshold will be assumed to belong to the zeroth order', default=500)
    parser.add_argument('--show-plots', '-p', action='store_true', help='Show plots of the calculated mean image and thresholded mask')
    args = parser.parse_args()

    # Now we get the configuration and blend it with the command line args
    config = config_handling.get_configuration()
    config = config_handling.blend_args_with_config(args, config)

    # Here we actually load the requested mask
    mask = config_handling.load_mask(config)
    with h5py.File(args.stxm_file, 'r') as stxm_file:
        center, probe, probe_mask = process_file(
            stxm_file, config,
            args.threshold, default_mask=mask)

    print('Center location at', center)

    if args.show_plots:
        from matplotlib import pyplot as plt
        plt.imshow(probe)
        plt.colorbar()
        plt.figure()
        plt.imshow(probe_mask)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    sys.exit(main())

