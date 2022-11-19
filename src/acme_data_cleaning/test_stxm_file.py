"""A script to process .stxm files to .cxi files

Author: Abe Levitan, alevitan@mit.edu
"""
import sys
import argparse
import numpy as np
import torch as t
import h5py
import glob
from cdtools.tools import plotting as p
from matplotlib import pyplot as plt
from acme_data_cleaning import image_handling, file_handling
from PIL import Image
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
            
        darks = file_handling.read_darks_from_stxm(
            stxm_file, n_exp_per_point=n_exp_per_point, device=device)
        darks = list(darks)
        chunked_exposures = [(t.stack([d[0] for d in darks[10:]]),
                              t.stack([d[1] for d in darks[10:]]))]
        darks = (t.mean(t.stack([d[0] for d in darks[:10]]),dim=0),
                 t.mean(t.stack([d[1] for d in darks[:10]]),dim=0))

        darks = tuple(image_handling.map_raw_to_tiles(dark) for dark in darks)

        chunked_exposures = file_handling.read_chunked_exposures_from_stxm(
            stxm_file, chunk_size=chunk_size,
            n_exp_per_point=n_exp_per_point, device=device)

        chunked_exposures = [next(chunked_exposures) for i in range(4)]
        for idx, exps in enumerate(chunked_exposures):
            if verbose:
                print('Processing frames', idx*chunk_size,
                      'to', (idx+1)*chunk_size-1, end='\r')
            #darks = [0 * dark for dark in darks]
            cleaned_exps, masks = zip(*(image_handling.process_frame(exp, dark,
                                                                     include_overscan=True)

                                        for exp, dark in zip(exps, darks)))
            double_summed = (cleaned_exps[1][...,::2,::2] +
                             cleaned_exps[1][...,::2,1::2] +
                             cleaned_exps[1][...,1::2,::2] +
                             cleaned_exps[1][...,1::2,1::2])
            
            print(t.min(t.mean(cleaned_exps[1],dim=0)))
            #image = Image.fromarray(t.mean(cleaned_exps[1],dim=0).cpu().numpy())
            #image.save('demo_image_streaks_magnetic_300ms.tif')
            p.plot_real(cleaned_exps[1][...,200:,:])
            p.plot_real(t.mean(cleaned_exps[1][...,:,:],dim=0))
            p.plot_real(t.mean(double_summed[...,100:,:],dim=0))
            plt.figure()
            print('')
            print(np.std(cleaned_exps[1][...,200:,:].cpu().numpy().ravel()))
            plt.hist(cleaned_exps[1].cpu().numpy().ravel(), bins=np.linspace(-7,7,100))
            plt.show()
            exit()
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
    parser.add_argument('--cpu', action='store_true', help='Run everything on the cpu')
    
    args = parser.parse_args()
    if args.cpu:
        device = 'cpu'
    else:
        # TODO: offer more flexibility in the GPU choice
        device='cuda:0'

    if args.compression.lower().strip() == 'none':
        args.compression = None
    else:
        args.compression = args.compression.lower().strip()
    
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
                         compression=args.compression,
                         default_mask=default_mask, device=device)


if __name__ == '__main__':
    sys.exit(main())

