import zmq
import h5py
from acme_data_cleaning import image_handling
from acme_data_cleaning import file_handling
import sys
from jax import numpy as np
import numpy as onp

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
        'position': None
    }

def process_start_event(cxi_file, state, event, pub):
    print('Processing start event')
    pub.send_pyobj(event)
    if state['metadata'] is not None:
        # This means that we missed a stop event but still think we are
        # taking data
        if cxi_file is not None:
            cxi_file.close()
        state['metadata'] = None
        state['darks'] = None
        state['current_exposures'] = None
        
    state['metadata'] = event['data']

    if state['metadata']['double_exposure']:
        # TODO: If this gets swapped, fix it.
        state['dwells'] = onp.array([state['metadata']['dwell2'],
                                    state['metadata']['dwell1']])

    else:
        state['dwells'] = onp.array([state['metadata']['dwell1']])
        
    psize = float(state['metadata']['geometry']['psize'])
    basis = np.array([[0,-psize,0],[-psize,0,0]])
    state['metadata']['geometry']['basis_vectors'] = basis.tolist()
            
    state['current_exposures'] = {dwell: [] for dwell in state['dwells']}
    state['current_masks'] = {dwell: [] for dwell in state['dwells']}
    state['darks'] = {dwell: None for dwell in state['dwells']}
    state['n_darks'] = {dwell: 0 for dwell in state['dwells']}

    # First, I need to find the right filename to save the .cxi file in
    # cxi_file = file

    # Next, I need to create and leave open that .cxi file

    # For now, I'm not bothering to actually save the .cxi file
    return None


def process_dark_event(cxi_file, state, event, pub):
    if state['metadata'] is None:
        print('Never got a start event, not processing')
        return
    print('Processing dark event')
    #'dwell' is where the dwell time is saved
    dwell = event['data']['dwell']
    dark = image_handling.map_raw_to_tiles(
        np.array(event['data']['ccd_frame']))
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
    all_dwells = np.array(all_dwells)
    all_frames = np.array(all_frames)
    all_masks = np.array(all_masks)

    combined_frame, combined_mask = image_handling.combine_exposures(
        all_frames, all_masks, all_dwells)

    combined_frame = onp.array(combined_frame)
    
    basis = onp.array(state['metadata']['geometry']['basis_vectors']).transpose()
    output_event = {
        'event':'frame',
        'data': combined_frame,
        'position': state['position'],
        'basis': basis
    }
    
    pub.send_pyobj(output_event)
    state['current_exposures'] = {dwell: [] for dwell in state['dwells']}
    state['current_masks'] = {dwell: [] for dwell in state['dwells']}

    

def process_exp_event(cxi_file, state, event, pub):
    if state['metadata'] is None:
        print('Never got a start event, not processing')
        return
    print('Processing exp event',event['data']['index'])
    
    state['position'] = onp.array([-event['data']['xPos'],
                                   -event['data']['yPos']])

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
    


def process_stop_event(cxi_file, state, event, pub):
    if state['metadata'] is None:
        print('Never got a start event, not processing')
        return
    print('Processing stop event')
    finalize_frame(cxi_file, state, event, pub)
    pub.send_pyobj(event)
    if cxi_file is not None:
        cxi_file.close()
    pass

def main(argv=sys.argv):
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect('tcp://localhost:37012')
    sub.setsockopt(zmq.SUBSCRIBE, b'')
    pub = context.socket(zmq.PUB)
    pub.bind("tcp://*:37013")

    state = make_new_state()
    cxi_file = None
    while True:
        event = sub.recv_pyobj()
        if event['event'].lower().strip() == 'start':
            cxi_file = process_start_event(cxi_file, state, event, pub)

        elif event['event'].lower().strip() == 'stop':
            process_stop_event(cxi_file, state, event, pub)
            # remove reference to the file so we know we're not writing
            cxi_file = None
            state = make_new_state()

        elif event['event'].lower().strip() == 'frame':
            if event['data']['ccd_mode'] == 'dark':
                process_dark_event(cxi_file, state, event, pub)
            else:
                process_exp_event(cxi_file, state, event, pub)


if __name__ == '__main__':
    sys.exit(main())
