import zmq
import h5py
from acme_data_cleaning import image_handling
from acme_data_cleaning import file_handling
import sys
import torch as t
import numpy as np

# This is apparently the best-practice way to load config files from within
# the package
import importlib.resources
import json

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
        state['dwells'] = np.array([state['metadata']['dwell2'],
                                    state['metadata']['dwell1']])

    else:
        state['dwells'] = np.array([state['metadata']['dwell1']])
        
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

    

def process_exp_event(cxi_file, state, event, pub, config):
    if state['metadata'] is None:
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

    package_root = importlib.resources.files('acme_data_cleaning')
    # This loads the default configuration first. This file is managed by
    # git and should not be edited by a user
    config = json.loads(package_root.joinpath('defaults.json').read_text())\

    # And now, if the user has installed an optional config file, we allow it
    # to override what is in defaults.json
    config_file_path = package_root.joinpath('config.json')

    # not sure if this works with zipped packages
    if config_file_path.exists():
        config.update(json.loads(config_file_path.read_text()))

    config['shear'] = np.array(config['shear'])

    # TODO: This is not currently used, because I don't want to actually
    # edit this data, I just want to includ ethe mask in the saved .cxi file.
    # currently, this doesn't actually save a .cxi file.
    # Load the default mask from a file. This may not work for
    # zipped packages, I don't know
    with h5py.File(package_root.joinpath('default_mask.h5'), 'r') as f:
        default_mask = t.as_tensor(np.array(f['mask']))
        # Crop out the correct part of the mask
        # default_mask = default_mask[sl[1:]]
    
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(config['subscription_port'])
    sub.setsockopt(zmq.SUBSCRIBE, b'')
    pub = context.socket(zmq.PUB)
    pub.bind(config['broadcast_port'])

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
                process_exp_event(cxi_file, state, event, pub, config)


if __name__ == '__main__':
    sys.exit(main())
