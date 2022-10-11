"""A script to process .stxm files to .cxi files

Author: Abe Levitan, alevitan@mit.edu
"""
import sys
import argparse
import numpy as np
import torch as t
import h5py
import glob
from acme_data_cleaning import image_handling, file_handling

# The shear calculations are so fast, there's no point in doing them
# on the GPU
default_shear = np.array([[ 0.99961877, -0.06551266],
                          [ 0.02651655,  0.99879594]])

def process_file(stxm_file, output_filename, chunk_size=10, verbose=True,
                 compression='lzf', default_mask=None, device='cpu'):
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
    # go into the .stxm file for ptychocam to use
    
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
        if verbose:
            print('File uses double exposures with exposure times',
                  exposure_times)
    else:
        n_exp_per_point=1
        # It honestly doesn't matter when it's not a double exposure
        exposure_times = np.array([metadata['dwell1']])
        if verbose:
            print('File uses single exposures')


    with file_handling.create_cxi(output_filename, metadata) as cxi_file:
        if default_mask is not None:
            file_handling.add_mask(cxi_file, default_mask)
            
        darks = file_handling.read_mean_darks_from_stxm(
            stxm_file, n_exp_per_point=n_exp_per_point, device=device)

        # we pre-map the darks to tile format, which avoids needing to redo
        # this computation every cycle
        darks = tuple(image_handling.map_raw_to_tiles(dark) for dark in darks)

        # This just creates the generator, the actual data isn't loaded until
        # we iterate through it.
        chunked_exposures = file_handling.read_chunked_exposures_from_stxm(
            stxm_file, chunk_size=chunk_size,
            n_exp_per_point=n_exp_per_point)
        
        for idx, exps in enumerate(chunked_exposures):
            if verbose:
                print('Processing frames', idx*chunk_size,
                      'to', (idx+1)*chunk_size-1, end='\r')

            cleaned_exps, masks = zip(*(image_handling.process_frame(exp, dark,
                                                include_overscan=False)

                                        for exp, dark in zip(exps, darks)))
            # Because combine_exposures works with an arbitrary number of
            # exposures, we just always use it, and avoid needing a separate
            # case for the single and double exposure processing.
            synthesized_exps, synthesized_masks = \
                image_handling.combine_exposures(
                    t.stack(cleaned_exps), t.stack(masks), exposure_times)
            chunk_translations = np.array(translations[idx*chunk_size:(idx+1)*chunk_size])

            chunk_translations[:,:2] = np.matmul(default_shear, chunk_translations[:,:2].transpose()).transpose()
            
            file_handling.add_frames(cxi_file,
                                     synthesized_exps,
                                     chunk_translations,
                                     masks=synthesized_masks,
                                     compression=compression)

        print('Finished processing                                          ')


def main(argv=sys.argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('stxm_file', nargs='+', type=str, help='The file or files to process, allowing for unix globbing')
    parser.add_argument('--chunk_size','-c', type=int, default=10, help='The chunk size for data processing, default is 10.')
    parser.add_argument('--compression', type=str, default='lzf', help='What hdf5 compression filter to use on the output CCD data. Default is lzf.')
    parser.add_argument('--succinct', action='store_true', help='Turns off verbose output')
    
    args = parser.parse_args()
    
    stxm_filenames = args.stxm_file

    # Default mask, TODO: should be loaded from a file
    default_mask = t.zeros([960,960])
    default_mask[:480,840:] = 1
    default_mask[:480,590] = 1
    default_mask = default_mask.swapaxes(-1,-2).flip(-1,-2)

    # Here we make globbing work nicely for files
    expanded_stxm_filenames = []
    for stxm_filename in stxm_filenames:
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
            process_file(stxm_file, output_filename,
                         chunk_size=args.chunk_size,
                         verbose=not args.succinct,
                         compression=args.compression.lower().strip(),
                         default_mask=default_mask)


if __name__ == '__main__':
    sys.exit(main())

