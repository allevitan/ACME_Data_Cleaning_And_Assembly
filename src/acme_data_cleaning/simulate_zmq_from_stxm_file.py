"""A script to simulate a ZMQ stream using a .stxm file

The simulated ZMQ stream is indended to be as close as is reasonable to
the stream which would be emitted live from the pystxmcontrol server during
data collection. This allows us to test the rest of the RPI/ptycho processing
chain

Author: Abe Levitan, alevitan@mit.edu
"""
import sys
import argparse
import jax.numpy as np
import h5py
from acme_data_cleaning import image_handling, file_handling


def process_file(stxm_file):
    #
    # We first read the metadata
    #
    metadata = file_handling.read_metadata_from_stxm(stxm_file)
    translations = file_handling.read_translations_from_stxm(stxm_file)

    # At this point, I should send out a "start" event"

    darks_iterator = read_darks_from_stxm(
        stxm_file, n_exp_per_point=n_exp_per_point)

    # Now I should iterate through the darks, emitting "dark" events

    exposure_iterator = file_handling.read_chunked_exposures_from_stxm(
        stxm_file, n_exp_per_point=n_exp_per_point)

    # And I should do the same, emitting "exposure" events

    # Then, I finally can emit a "stop" event
    

def main(argv=sys.argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('stxm_file', nargs='+', type=str, help='The file or files to process')
    
    
    args = parser.parse_args()
    
    stxm_filenames = args.stxm_file

    # Default mask, TODO: should be loaded from a file
    default_mask = np.zeros([960,960])
    default_mask = default_mask.at[:480,840:].set(1)

    for stxm_filename in stxm_filenames:
        print('Processing',stxm_filename)
        
        with h5py.File(stxm_filename, 'r') as stxm_file:
            process_file(stxm_file)
            
    
    
if __name__ == '__main__':
    sys.exit(main())

