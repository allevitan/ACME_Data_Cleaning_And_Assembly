import zmq
import h5py
from contextlib import contextmanager
from acme_data_cleaning import image_handling, file_handling
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


# I wrote this as a context manager because the natural pattern is for it
# to return an open .cxi file. As a context manager, we can make sure that
# those files are always properly closed. 
@contextmanager
def process_start_event(cxi_file, state, event, pub, mask=None):
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
    output_filename = make_output_filename(state)
    
    # Next, we assemble the metadata dictionary
    metadata = {}
    
    # Next, I need to create and leave open that .cxi file
    with file_handling.create_cxi(output_filename, metadata) as cxi_file:
        # We should add the mask at this point
        if mask is not None:
            file_handling.add_mask(cxi_file, mask)

        yield cxi_file


def make_output_filename(state):
    # TODO: Make this work!
    # What goes into the defaul filename
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
    # Should we create a fake stop event to pass along? I think we shouldn't,
    # but if we decide to, this would be some of the code
    #pub.send_pyobj(event)
    


def run_data_accumulation_loop(cxi_file, state, event,
                               sub, pub, rec_trigger, config):
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

    parser = argparse.ArgumentParser(
        prog = 'process_live_data',
        description = 'Runs a program which listens for raw data being emitted by pystxmcontrol. It assembles and preprocesses this data, saving the results inot .cxi files and passing the cleaned data on for further analysis.')
    
    
    parser.add_argument('--mask','-m', type=str, default='', help='A custom mask file to use, if the default is not appropriate')
    parser.add_argument('--chunk_size','-c', type=int, default=10, help='The chunk size to use in the output .cxi files, default is 10.')
    parser.add_argument('--compression', type=str, default='lzf', help='What hdf5 compression filter to use on the output CCD data. Default is lzf.')
    parser.add_argument('--succinct', action='store_true', help='Turns off verbose output')
    parser.add_argument('--cpu', action='store_true', help='Run everything on the cpu')
    parser.add_argument('--center', type=int, nargs=2)
    ## TODO: Make the radius configurable
    parser.add_argument('--radius', type=int)
    # TODO: Plan for the downsampling:
    
    
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
    
    # Now we create the data slice
    if args.center is None:
        args.center = [480,480]
    if args.radius is None:
        sl = np.s_[:,:,:]
    else:
        sl = np.s_[:,args.center[0]-args.radius:args.center[0]+args.radius,
                   args.center[1]-args.radius:args.center[1]+args.radius]

    
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

    # TODO: include this mask in the saved .cxi file
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
    rec_trigger = context.socket(zmq.PUB)
    rec_trigger.bind(config['trigger_reconstruction_port'])

    state = make_new_state()
    start_event = None
    while True:
        # This logic allows us to deal with cases where the 'stop' event
        # doesn't make it to the program. In that case, we need to keep the
        # second 'start' event around after we read it to start the next file
        if start_event is None:
            event = sub.recv_pyobj()
        else:
            event = start_event
            start_event = None
            
        if event['event'].lower().strip() == 'start':
            with process_start_event(state, event, pub,
                                     mask=default_mask) as cxi_file:
                start_event = run_data_accumulation_loop(
                    cxi_file, state, event, sub, pub, rec_trigger, config)

            # we need to trigger the ptycho reconstruction after saving the
            # file, to ensure that the file exists when the reconstruction
            # begins
            trigger_ptycho(state, rec_trigger)
            state = make_new_state()



if __name__ == '__main__':
    sys.exit(main())
