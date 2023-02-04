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

# This is apparently the best-practice way to load config files from within
# the package
import importlib.resources
import json


def process_file(stxm_file, output_filename, config, default_mask=None):
    #
    # We first read the metadata
    #
    metadata = file_handling.read_metadata_from_stxm(stxm_file)
    translations = file_handling.read_translations_from_stxm(stxm_file)

    # TODO: There appears to be tons of metadata in the .stxm file which is
    # not in the metadata dictionary, such as start and end times, sample info,
    # proposal, experimenters, energy.

    # TODO: Right now, I'm not storing information about the probe guess. It
    # would be nice to be able to provide a probe calibration that could then
    # go into the .cxi file for ptychocam to use
    
    #
    # Next we get the key parameters from the metadata that we need to know
    # for the scan.
    #    
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
        # It honestly doesn't matter when it's not a double exposure
        exposure_times = np.array([metadata['dwell1']])
        if not config['succinct']:
            print('File uses single exposures')
    
                
    darks = file_handling.read_mean_darks_from_stxm(
        stxm_file, n_exp_per_point=n_exp_per_point, device=config['device'])
    
    # Instantiating the cleaner with the darks allows us to do the
    # preprocessing on the darks once, not having to repeat it at each
    # frame
    frame_cleaner = image_handling.FastCCDFrameCleaner(darks)
    
    # Instantiating a resampler at the start allows us to create the
    # various slices and info needed for the resampling functions
    # once and reuse that for the full scan
    
    # This just gets us the size of the input images
    dummy_im = frame_cleaner.process_frame(
        darks[0], 0, include_overscan=False)[0]
    
    output_shape = (config['output_pixel_count'],)*2
    
    if config['interpolate']:
        hc = 1.986446e-25 # in Joule-meters
        energy = metadata['energy'] * 1.60218e-19 # convert to Joules
        wavelength = hc / energy
        pixel_pitch = metadata['geometry']['psize']
        det_distance = metadata['geometry']['distance']
        
        # In this case, we define the binning_factor using the
        # output_pixel_size
        # Note, this will only work right when the output shape is
        # the same in both dimensions
        binning_factor = ( wavelength * det_distance / 
                           ( output_shape[0] * pixel_pitch
                             * (config['output_pixel_size'] * 1e-9)))
        
        metadata['geometry']['psize'] *= float(binning_factor)
        if 'basis_vectors' in metadata['geometry']:
            metadata['geometry']['basis_vectors'] = \
                (np.array(metadata['geometry']['basis_vectors'])
                 * float(binning_factor)).tolist()
        
        resampler = image_handling.InterpolatingResampler(
            dummy_im, config['center'], output_shape,
            binning_factor)
    else:
        # In this case, we ignore the output_pixel_size and just use the
        # manually defined binning factor

        metadata['geometry']['psize'] *= float(config['binning_factor'])
        if 'basis_vectors' in metadata['geometry']:
            metadata['geometry']['basis_vectors'] = \
                (np.array(metadata['geometry']['basis_vectors'])
                 * float(config['binning_factor'])).tolist()
                 
        resampler = image_handling.NonInterpolatingResampler(
            dummy_im, config['center'], output_shape,
            config['binning_factor'])



    # Now we actually open the file and start to write to it
    with file_handling.create_cxi(output_filename, metadata) as cxi_file:

        # The first thing we do is resample the default mask with the resampler
        if default_mask is not None:
            mask = default_mask.to(device=dummy_im.device)
            print(mask.shape)
            _, resampled_mask = resampler.resample(dummy_im, masks=mask)
            file_handling.add_mask(cxi_file, resampled_mask)
        
        # This just gets an iterator, it will wait to actually load the
        # images until it's iterated over
        chunked_exposures = file_handling.read_chunked_exposures_from_stxm(
            stxm_file, chunk_size=config['chunk_size'],
            n_exp_per_point=n_exp_per_point, device=config['device'])
        
        for idx, exps in enumerate(chunked_exposures):
            if not config['succinct']:
                print('Processing frames', idx*config['chunk_size'],
                      'to', (idx+1)*config['chunk_size']-1, end='\r')

            # index is the index of the frame within the exposure, but idx
            # refers to the index of the chunk we're dealing with
            cleaned_exps, masks = zip(
                *(frame_cleaner.process_frame(
                    exp, index, include_overscan=False)
                  for index, exp in enumerate(exps)))
            
            # Because combine_exposures works with an arbitrary number of
            # exposures, we just always use it, and avoid needing a separate
            # case for the single and double exposure processing.
            synthesized_exps, synthesized_masks = \
                image_handling.combine_exposures(
                    t.stack(cleaned_exps), t.stack(masks), exposure_times)


            # Now we resample the outputs to the requested format
            resampled_exps, resampled_masks = \
                resampler.resample(synthesized_exps,
                                   masks=synthesized_masks)
            
            chunk_translations = np.array(
                translations[idx*config['chunk_size']:
                             (idx+1)*config['chunk_size']])
            chunk_translations[:,:2] = np.matmul(
                config['shear'], chunk_translations[:,:2].transpose()).transpose()
            
            file_handling.add_frames(cxi_file,
                                     resampled_exps,
                                     chunk_translations,
                                     masks=resampled_masks,
                                     compression=config['compression'])

        print('Finished processing                                          ')


def main(argv=sys.argv):

    # This argument reading setup is a bit bizarre. The idea is that all the
    # arguments will have default values, which are stored in the configuration
    # file. If any are overridden by being explicitly set, then that will
    # override the values in the configuration file
    
    parser = argparse.ArgumentParser()

    parser.add_argument('stxm_file', nargs='+', type=str, help='The file or files to process, allowing for unix globbing')
    config_handling.add_processing_args(parser)
        
    # We parse the input command line args
    args = parser.parse_args()
    
    config = config_handling.get_configuration()

    config = config_handling.blend_args_with_config(args, config)
    

    device = config['device']
    
    
    if config['compression'].lower().strip() == 'none':
        compression = None
    else:
        compression = config['compression'].lower().strip()
    
    
    # Load the default mask from a file. This may not work for
    # zipped packages, I don't know
    with h5py.File(config['mask'], 'r') as f:
        default_mask = t.as_tensor(np.array(f['mask']))

    # TODO: We need to make the data binning and resampling code work properly
    # with the masks, and produce appropriate masks for the downsampled data!
            
    # Here we make globbing work nicely for files
    expanded_stxm_filenames = []
    for stxm_filename in args.stxm_file:
        filenames = glob.glob(stxm_filename)
        if len(filenames) == 0:
            print('WARNING:',stxm_filename,'did not match any files.')
        for filename in filenames:
            if filename not in expanded_stxm_filenames:
                expanded_stxm_filenames.append(filename)

    
    for stxm_filename in expanded_stxm_filenames:
        output_filename = '.'.join(stxm_filename.split('.')[:-1])+'.cxi'
        if not args.succinct:
            print('Processing',stxm_filename)
            print('Output will be saved in', output_filename)
            
        with h5py.File(stxm_filename, 'r') as stxm_file:
            process_file(stxm_file, output_filename, config,
                         default_mask=default_mask)


if __name__ == '__main__':
    sys.exit(main())

