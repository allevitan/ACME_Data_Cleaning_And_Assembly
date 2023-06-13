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


def delete_bad_frames(frames, translations):
    sh = frames.shape
    ny, nx = sh[1] // 2, sh[2] // 2
    idx = []
    for i in range(len(frames)):
        if frames[i, ny - 5: ny + 5, 0: 100].mean() > 2. * frames[i, ny - 5: ny + 5, -100:].mean():
            pass
        else:
            idx.append(i)
    #print("Indices: ", idx)
    return frames[idx], translations[idx]

def init_probe(frames):
    dataAve = frames.cpu().mean(0)
    pMask = np.fft.fftshift((dataAve > 0.01 * dataAve.max()))
    probe = np.sqrt(np.fft.fftshift(dataAve)) * pMask
    probe = np.fft.ifftshift(np.fft.ifftn(probe))
    return probe, pMask


def process_file(stxm_file, output_filename, config, default_mask=None):
    """Processes a single stxm file to .cxi
    """
    
    # We first read the metadata and translations from the stxm file
    metadata = file_handling.read_metadata_from_stxm(stxm_file)
    translations = file_handling.read_translations_from_stxm(stxm_file)

    # TODO: Properly handle the metadata in this dictionary, like start
    # and end times, sample info, proposal, experimenters, energy.
    
    # Next we extract the key parameters from the metadata that we need to know
    # to generate the .cxi file
    # TODO: Set up a system that can flexibly do N exposures
    if metadata['double_exposure']:
        exp_type = 'double'
        n_exp_per_point=2
        # Note: Old files have dwell1 meaning the second exposure and
        # dwell2 meaning the first exposure
        exposure_times = np.array([metadata['dwell2'],
                                   metadata['dwell1']])
        if not config['succinct']:
            print('File uses double exposures with exposure times',
                  exposure_times)
    elif 'n_repeats' in metadata and metadata['n_repeats'] != 1:
        exp_type = 'repeated'
        n_exp_per_point = int(metadata['n_repeats'])
        exposure_times = np.array([metadata['dwell1']]
                                  * n_exp_per_point)
        if not config['succinct']:
            print('File uses repeated exposures,', n_exp_per_point,
                  'repeats of', metadata['dwell1'], 'ms exposures')
    else:
        exp_type = 'single'
        n_exp_per_point=1
        exposure_times = np.array([metadata['dwell1']])
        if not config['succinct']:
            print('File uses single exposures')

    if config['use_all_exposures'].lower().strip() == 'auto':
        use_all_exposures = (True if exp_type=='repeated' else False)
    else:
        use_all_exposures = config['use_all_exposures']
        
    if not config['succinct']:
        if use_all_exposures:
            print('Will include data from all non-saturated exposures, best for repeated exposures with identical dwell times.')
        else:
            print('Will only include data from the longest non-saturated exposure, best for double exposures with differing dwell times')


    if exp_type == 'repeated':
        # This gets a list of mean dark patterns, and we know that all the darks
        # share the same exposure, so we treat them the same
        darks = file_handling.read_mean_darks_from_stxm(
            stxm_file, n_exp_per_point=1, device=config['device'])
    else:
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
    
    resampler = image_handling.make_resampler(metadata, config, dummy_im)

    # Now we actually open the file, and we will write to it as we
    # process the data.
    with file_handling.create_cxi(output_filename, metadata) as cxi_file:

        # The first thing we do is resample the default mask with the resampler,
        # to get a mask that will match the output data
        if default_mask is not None:
            mask = default_mask.to(device=dummy_im.device)
            _, resampled_mask = resampler.resample(dummy_im, masks=mask)
            file_handling.add_mask(cxi_file, resampled_mask)

        # Read the number of frames
        n_frames = file_handling.get_n_frames_from_stxm(
            stxm_file, n_exp_per_point=n_exp_per_point)
        
        # This just gets an iterator, it will wait to actually load the
        # images until it's iterated over
        chunked_exposures = file_handling.read_chunked_exposures_from_stxm(
            stxm_file, chunk_size=config['chunk_size'],
            n_exp_per_point=n_exp_per_point, device=config['device'])
        
        for idx, exps in enumerate(chunked_exposures):
            if not config['succinct']:
                print('Processing frames', idx*config['chunk_size']+1,
                      'to', min(n_frames, (idx+1)*config['chunk_size']),
                      'of', n_frames, end='\r')

            # We have to treat the repeated exposure case separately
            # because all the exposures use the same darks
            if exp_type == 'repeated':
                cleaned_exps, masks = zip(
                    *(frame_cleaner.process_frame(
                        exp, 0, include_overscan=False,
                        med_width=config['background_median_width'],
                        max_correction=config['max_overscan_correction'],
                        min_correction=config['min_overscan_correction'],
                        background_offset=config['background_offset'],
                        cut_zeros=config['cut_zeros'])
                      for exp in exps))
            else:
                
                # index here is the index of the frame within the exposure, but
                # idx refers to the index of the chunk we're dealing with
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
                    t.stack(cleaned_exps), t.stack(masks), exposure_times,
                    use_all_exposures=use_all_exposures)

            # Now we resample the outputs to the requested format
            resampled_exps, resampled_masks = \
                resampler.resample(synthesized_exps,
                                   masks=synthesized_masks)

            # Here we correct the translations for shear
            chunk_translations = np.array(
                translations[idx*config['chunk_size']:
                             idx*config['chunk_size']+resampled_exps.shape[0]])
            chunk_translations[:,:2] = np.matmul(
                config['shear'],
                chunk_translations[:,:2].transpose()).transpose()

            if config["delete_bad_frames"]:
                resampled_exps, chunk_translations = delete_bad_frames(resampled_exps, chunk_translations)

            if idx == 0:
                probe, probe_mask = init_probe(resampled_exps)
                file_handling.add_probe(cxi_file, probe, probe_mask)

            # This actually adds the frames to the cxi file
            if len(resampled_exps) != 0:
                file_handling.add_frames(cxi_file,
                                        resampled_exps,
                                        chunk_translations,
                                        masks=resampled_masks,
                                        compression=config['compression'])
            
        print('Finished processing                                          ')


def main(argv=sys.argv):

    # This argument reading setup is a bit unusual. The idea is that all the
    # arguments will have default values, which are stored in the configuration
    # file. If any are overridden by being explicitly set, then that will
    # override the values in the configuration file
    
    # We parse the input command line args here
    parser = argparse.ArgumentParser()
    parser.add_argument('stxm_file', nargs='+', type=str, help='The file or files to process, allowing for unix globbing')
    config_handling.add_processing_args(parser)
    args = parser.parse_args()

    # Now we get the configuration and blend it with the command line args
    config = config_handling.get_configuration()
    config = config_handling.blend_args_with_config(args, config)

    # We print a small summary of the key config parameters
    if not config['succinct']:
        config_handling.summarize_config(config)

    # Here we actually load the requested mask
    mask = config_handling.load_mask(config)

    # Now what we do is produce a list of all the files to process, using the
    # listed files (possibly defined via unix globs)
    expanded_stxm_filenames = []
    for stxm_filename in args.stxm_file:
        filenames = glob.glob(stxm_filename)
        if len(filenames) == 0:
            print('WARNING:',stxm_filename,'did not match any files.')
        for filename in filenames:
            if filename not in expanded_stxm_filenames:
                expanded_stxm_filenames.append(filename)

    # And finally, we process all the stxm files we found
    for stxm_filename in expanded_stxm_filenames:
        output_filename = '.'.join(stxm_filename.split('.')[:-1])+'.cxi'
        if not config['succinct']:
            print('Processing',stxm_filename)
            print('Output will be saved in', output_filename)
            
        with h5py.File(stxm_filename, 'r') as stxm_file:
            process_file(stxm_file, output_filename, config,
                         default_mask=mask)


if __name__ == '__main__':
    sys.exit(main())

