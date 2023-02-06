import zmq
import h5py
from contextlib import contextmanager
from acme_data_cleaning import image_handling, file_handling, config_handling
import argparse
import sys
import torch as t
import numpy as np


# How should I structure this? I'll get 4 kinds of events, and each event
# needs to:
# 1) update the state of the processor
# 2) (potentially) emit a matched event on the pub stream

# What information do we need to have in the persistent state in order to
# build up the cxi file? I need:
# 1) Any frames that are still possibly being processed
# 2) All the darks which have been collected on this scan
# 3) All the metadata for the scan



def make_new_state():
    return {
        'metadata': None,
        'dwells': None,
        'darks': None,
        'n_darks': None,
        'index': 0,
        'current_exposures': None,
        'current_masks': None,
        'position': None,
        'frame_cleaner': None,
        'resampler': None
    }


# I wrote this as a context manager because the natural pattern is for it
# to return an open .cxi file. As a context manager, we can make sure that
# those files are always properly closed. 
@contextmanager
def process_start_event(state, event, pub, config):
    """Processes a start event, getting the metadata and opening a cxi_file
    """
    
    pub.send_pyobj(event)

    print('Processing start event')

    # We load the mask, preloaded on the correct device
    mask = config_handling.load_mask(config)

    # TODO: need to figure out where this metadata actually is
    state['metadata'] = event['data']
    
    if state['metadata']['double_exposure']:
        # TODO: If this gets swapped, fix it.
        state['dwells'] = np.array([state['metadata']['dwell2'],
                                    state['metadata']['dwell1']])
        print('Start event indicates double exposures with exposure times',
              state['dwells'])
    else:
        state['dwells'] = np.array([state['metadata']['dwell1']])
        print('Start event indicates single exposures.')
    
        
    # We need to add the basis to the metadata
    psize = float(state['metadata']['geometry']['psize'])
    basis = np.array([[0,-psize,0],[-psize,0,0]])
    state['metadata']['geometry']['basis_vectors'] = basis.tolist()

    # Now we instantiate the info we need to accumulate in the state
    # dictionary. This is where we store information that we need to keep
    # around between events.
    state['current_exposures'] = {dwell: [] for dwell in state['dwells']}
    state['current_masks'] = {dwell: [] for dwell in state['dwells']}
    state['darks'] = {dwell: None for dwell in state['dwells']}
    state['n_darks'] = {dwell: 0 for dwell in state['dwells']}
    
    # Now I find the right filename to save the .cxi file in
    output_filename = make_output_filename(state)
    
    # Next, we assemble the metadata dictionary
    metadata = {}
    
    # Next, I need to create and leave open that .cxi file
    with file_handling.create_cxi(output_filename, metadata) as cxi_file:
        # We should add the mask at this point
        file_handling.add_mask(cxi_file, mask)

        yield cxi_file


def make_output_filename(state):
    # TODO: Make this work!
    # What goes into the default filename
    # There is the date, scan number, region number, and energy number
    date = '220925' # probably best to get this info from the scan data,
    # if it exists, so there's no chance of it being saved under a different
    # name from the .stxm file
    scan_no = '%03d' '006'
    region_no = str(0)
    energy_no = str(0)
    return ('NS_' + date + scan_no + '_ccdframes_'
            + region_no + '_' + energy_no + '.cxi')
    

def process_dark_event(cxi_file, state, event, pub):
    if state['metadata'] is None:
        print('Never got a start event, not processing')
        return
    print('Processing dark event')
    #'dwell' is where the dwell time is saved
    dwell = event['data']['dwell']
    dark = image_handling.map_raw_to_tiles(
        t.as_tensor(np.array(event['data']['ccd_frame'])))
    if state['darks'][dwell] is None:
        state['darks'][dwell] = dark
    else:
        n_darks = state['n_darks'][dwell]
        state['darks'][dwell] = ((state['darks'][dwell] * n_darks + dark) / 
                                 (n_darks + 1))
    state['n_darks'][dwell] += 1


def finalize_frame(cxi_file, state, event, pub):
    all_dwells = []
    all_frames = []
    all_masks = []
    for dwell, frames in state['current_exposures'].items():
        masks = state['current_masks'][dwell]
        all_dwells.extend([dwell,] * len(frames))
        all_frames.extend(frames)
        all_masks.extend(masks)


    if len(all_frames) == 0:
        return 
    all_dwells = t.as_tensor(np.array(all_dwells))
    all_frames = t.as_tensor(np.array(all_frames))
    all_masks = t.as_tensor(np.array(all_masks))

    combined_frame, combined_mask = image_handling.combine_exposures(
        all_frames, all_masks, all_dwells)

    combined_frame = np.array(combined_frame)

    basis = np.array(state['metadata']['geometry']['basis_vectors']).transpose()
    output_event = {
        'event':'frame',
        'data': combined_frame,
        'position': state['position'],
        'basis': basis
    }
    
    pub.send_pyobj(output_event)
    state['current_exposures'] = {dwell: [] for dwell in state['dwells']}
    state['current_masks'] = {dwell: [] for dwell in state['dwells']}

    file_handling.add_frame(cxi_file,
                            combined_frame,
                            synthesized_exps[sl],
                            chunk_translations,
                            masks=synthesized_masks[sl],
                            compression=compression)
    
    

def process_exp_event(cxi_file, state, event, pub, config):
    if state['metadata'] is None:
        # This is just a backup check, the upstream logic *should* prevent
        # this from getting triggered in the absence of a start event

        print('Never got a start event, not processing')
        return
    
    print('Processing exp event',event['data']['index'])
    
    state['position'] = np.array([-event['data']['xPos'],
                                   -event['data']['yPos']])
    
    # A correction for the shear in the probe positions
    state['position'] = np.matmul(config['shear'], state['position'])

    dwell = event['data']['dwell']
    dark = state['darks'][dwell]
    # TODO: There seems to b
    raw_frame = np.array(event['data']['ccd_frame'])
    frame, mask = image_handling.process_frame(raw_frame, dark)
    
    if event['data']['index'] != state['index']:
        # we've moved on, so we need to finalize the frame
        finalize_frame(cxi_file, state, event, pub)
        state['index'] = event['data']['index']

    # We always need to do this
    state['current_exposures'][dwell].append(frame)
    state['current_masks'][dwell].append(mask)


def trigger_ptycho(state, rec_trigger):
    output_filename = make_output_filename(state)
    print('TODO: trigger a ptychography reconstruction')
    

def process_stop_event(cxi_file, state, event, pub, rec_trigger):
    if state['metadata'] is None:
        # This is just a backup check, the upstream logic *should* prevent
        # this from getting triggered in the absence of a start event
        print('Never got a start event, not processing')
        return
    
    print('Processing stop event')
    finalize_frame(cxi_file, state, event, pub)
    trigger_ptycho(cxi_file, state, rec_trigger)
    pub.send_pyobj(event)


def process_emergency_stop(cxi_file, state, pub, rec_trigger):
    """This is triggered if no stop event happens, but we get a second
    start event. In this case, we just finish the scan without any of the
    info in the stop event.
    """
    print('Doing an emergency stop')
    finalize_frame(cxi_file, state, pub)

    


def run_data_accumulation_loop(cxi_file, state, event,
                               sub, pub, rec_trigger, config):
    """Accumulates darks and frame data into a cxi file, until it completes
    """
    
    while True:
        event = sub.recv_pyobj()
        
        if event['event'].lower().strip() == 'start':
            process_emergency_stop(cxi_file, state, pub, rec_trigger)
            return event # pass the event back so we can start a new loop

        if event['event'].lower().strip() == 'stop':
            process_stop_event(cxi_file, state, event, pub, rec_trigger)
            return None
                    
        elif event['event'].lower().strip() == 'frame':
            if event['data']['ccd_mode'] == 'dark':
                process_dark_event(cxi_file, state, event, pub)
            else:
                process_exp_event(cxi_file, state, event, pub, config)


def main(argv=sys.argv):

    # I'm just using this so the script has a help printout
    parser = argparse.ArgumentParser(
        prog = 'process_live_data',
        description = 'Runs a program which listens for raw data being emitted by pystxmcontrol. It assembles and preprocesses this data, saving the results inot .cxi files and passing the cleaned data on for further analysis.')
    args = parser.parse_args()
    
    # TODO maybe add the ZMQ ports as args?
    
    # Now we get the config info to read the ZMQ ports
    config = config_handling.get_configuration()

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(config['subscription_port'])
    sub.setsockopt(zmq.SUBSCRIBE, b'')
    pub = context.socket(zmq.PUB)
    pub.bind(config['broadcast_port'])
    rec_trigger = context.socket(zmq.PUB)
    rec_trigger.bind(config['trigger_reconstruction_port'])

    start_event = None
    while True:
                
        # This logic allows us to deal with cases where the 'stop' event
        # doesn't make it to the program. In that case, we need to keep the
        # 'start' event around to start the next file
        if start_event is None:
            event = sub.recv_pyobj()
        else:
            event = start_event
            start_event = None
            
        if event['event'].lower().strip() == 'start':
            # We reload the config data following every start event.
            # Not all of the updated config parameters will actually be
            # used, but this makes it easy to update things like the
            # detector center, etc. and have them propagate as soon as possible.
            config = config_handling.get_configuration()
            state = make_new_state()
            
            # Process start event returns a context manager for an open
            # cxi file with the appropriate metadata
            with process_start_event(state, event, pub, config=config) \
                 as cxi_file:
                # This loop will read in darks and exp data until it gets
                # a stop or start event, at which point it will return.
                start_event = run_data_accumulation_loop(
                    cxi_file, state, event, sub, pub, rec_trigger, config)

            # we wait to trigger the ptycho reconstruction until after saving
            # the file, to ensure that the file exists when the reconstruction
            # begins.
            trigger_ptycho(state, rec_trigger)



if __name__ == '__main__':
    sys.exit(main())
