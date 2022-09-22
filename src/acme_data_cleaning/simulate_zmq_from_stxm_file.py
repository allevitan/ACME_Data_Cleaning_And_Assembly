"""A script to simulate a ZMQ stream using a .stxm file

The simulated ZMQ stream is indended to be as close as is reasonable to
the stream which would be emitted live from the pystxmcontrol server during
data collection. This allows us to test the rest of the RPI/ptycho processing
chain

Author: Abe Levitan, alevitan@mit.edu
"""
import sys
import argparse
import numpy as np
import h5py
import zmq
from acme_data_cleaning import image_handling, file_handling
import time

def process_file(stxm_file, pub_socket):
    #
    # We first read the metadata
    #
    metadata = file_handling.read_metadata_from_stxm(stxm_file)
    translations = file_handling.read_translations_from_stxm(stxm_file)

    print('start event')
    start_event = {
        'event': 'start',
        'data': metadata,
    }
    
    pub_socket.send_pyobj(start_event)
    # At this point, I should send out a "start" event"

    n_exp_per_point = 2
    
    darks_iterator = file_handling.read_darks_from_stxm(
        stxm_file, n_exp_per_point=n_exp_per_point)

    # Now I should iterate through the darks, emitting "dark" events
    for idx, dark_set in enumerate(darks_iterator):
        time.sleep(0.1)
        for dark in dark_set:
            print('Frame Event (dark)', idx)
            event = {
                'event': 'frame',
                'ccd_mode': 'dark',
                'ccd_frame': dark,
            }
            pub_socket.send_pyobj(event)
    
    exposure_iterator = file_handling.read_chunked_exposures_from_stxm(
        stxm_file, n_exp_per_point=n_exp_per_point)

    for idx,(translation, exposure_set) \
        in enumerate(zip(translations, exposure_iterator)):
        time.sleep(0.1)
        for exposure in exposure_set:
            print('Frame Event (exposure)', idx)
            event = {
                'event': 'frame',
                'ccd_mode': 'exp',
                'ccd_frame': exposure,
                'xPos': translation[0],
                'yPos': translation[1],
                'index': idx
            }
            pub_socket.send_pyobj(event)

    print('Stop event')
    # Then, I finally can emit a "stop" event
    stop_event = {'event': 'stop',
                  'abort': False}
    pub_socket.send_pyobj(stop_event)

def main(argv=sys.argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('stxm_file', nargs='+', type=str, help='The file or files to process')
    
    
    args = parser.parse_args()

    # Define the socket to publish patterns on
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.bind("tcp://*:37012")
    
    stxm_filenames = args.stxm_file

    # Default mask, TODO: should be loaded from a file
    default_mask = np.zeros([960,960])
    default_mask[:480,840:] = 1

    for stxm_filename in stxm_filenames:
        print('Processing',stxm_filename)
        
        with h5py.File(stxm_filename, 'r') as stxm_file:
            process_file(stxm_file, pub)
            
    
    
if __name__ == '__main__':
    sys.exit(main())

