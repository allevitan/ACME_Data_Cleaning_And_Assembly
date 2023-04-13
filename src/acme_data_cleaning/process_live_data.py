import time
import os
import zmq
import h5py
from contextlib import contextmanager
from acme_data_cleaning import image_handling, file_handling, config_handling, illumination_init
import argparse
import sys
import torch as t
import numpy as np
from cosmicstreams.PreprocessorStream import PreprocessorStream
from scipy import constants
import copy


# How should I structure this? I'll get 4 kinds of events, and each event
# needs to:
# 1) update the state of the processor
# 2) (potentially) emit a matched event on the pub stream

# What information do we need to have in the persistent state in order to
# build up the cxi file? I need:
# 1) Any frames that are still possibly being processed
# 2) All the darks which have been collected on this scan
# 3) All the metadata for the scan


def get_dataset_name(metadata):
    filepath = metadata['header']
    filename = os.path.basename(filepath)
    identifier = filename[:12]
    return identifier


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
        'resampler': None,
        'mask': None,
        'start_sending_frames': False
    }


# I wrote this as a context manager because the natural pattern is for it
# to return an open .cxi file. As a context manager, we can make sure that
# those files are always properly closed. 
@contextmanager
def process_start_event(state, event, pub, config):
    """Processes a start event, getting the metadata and opening a cxi_file
    """
    print('Processing start event')

    # We load the mask, preloaded on the correct device
    state['mask'] = config_handling.load_mask(config)

    # TODO: need to figure out where this metadata actually is
    state['metadata'] = event['metadata']

    if state['metadata']['double_exposure']:
        # TODO: If this gets swapped, fix it.
        state['dwells'] = np.array([state['metadata']['dwell2'], state['metadata']['dwell1']])
        print('Start event indicates double exposures with exposure times', state['dwells'])
    else:
        state['dwells'] = np.array([state['metadata']['dwell1']])
        print('Start event indicates single exposures.')

    # We need to add the basis to the metadata
    state['metadata']['geometry']['psize'] = state['metadata']['geometry']['psize'] * 1e-6
    state['metadata']['geometry']['distance'] = state['metadata']['geometry']['distance'] * 1e-3

    psize = float(state['metadata']['geometry']['psize'])
    basis = np.array([[0, -psize, 0], [-psize, 0, 0]])
    state['metadata']['geometry']['basis_vectors'] = basis.tolist()

    # Now we instantiate the info we need to accumulate in the state
    # dictionary. This is where we store information that we need to keep
    # around between events.
    state['current_exposures'] = {dwell: [] for dwell in state['dwells']}
    state['current_masks'] = {dwell: [] for dwell in state['dwells']}
    state['darks'] = {dwell: None for dwell in state['dwells']}
    state['n_darks'] = {dwell: 0 for dwell in state['dwells']}
    state['pix_translations'] = True
    # state['metadata']['pix_translations'] = True

    # state['metadata']['translations'] = (-np.array(state['metadata']['translations']) * 1e-6).tolist()
    state['metadata']['translations'] = (np.array(state['metadata']['translations']) * 1e-6).tolist()
    state['metadata'] = state['metadata']
    # state['metadata'] = copy.deepcopy(state['metadata'])
    state['metadata']['output_frame_width'] = config['output_pixel_count']
    state['metadata']['x_pixel_size'] = psize
    state['metadata']['y_pixel_size'] = psize
    state['metadata']['detector_distance'] = state['metadata']['geometry']['distance']
    state['metadata']['energy'] = state['metadata']['energy'] * constants.e
    state['metadata']['pix_translations'] = True

    wavelength = constants.h * constants.c / state['metadata']['energy']
    alpha = np.arctan(psize * state['metadata']['output_frame_width'] / (2 * state['metadata']['detector_distance']))
    px_size = wavelength / (2 * np.sin(alpha))
    state['px_size_real_space'] = px_size

    # Translations are originally given as (y, x).
    state['metadata']['translations'] = np.array(state['metadata']['translations'])
    translations_tmp = state['metadata']['translations'].copy()

    # Translations are changed to (x, y) to apply the shear.
    state['metadata']['translations'][:, 0] = translations_tmp[:, 1]
    state['metadata']['translations'][:, 1] = translations_tmp[:, 0]
    state['metadata']['translations'] = np.dot(config['shear'], state['metadata']['translations'].T).T

    state['metadata']['translations'] = np.array(state['metadata']['translations']) / px_size
    state['metadata']['translations'] = np.ceil(state['metadata']['translations'])
    state['metadata']['translations'][:, 0] -= state['metadata']['translations'][:, 0].min()
    state['metadata']['translations'][:, 1] -= state['metadata']['translations'][:, 1].min()

    state['metadata']['translations'] = state['metadata']['translations'].tolist()

    identifier = get_dataset_name(state['metadata'])
    state['metadata']['identifier'] = identifier

    metadata_cxi = copy.deepcopy(state['metadata'])
    state['metadata_cxi'] = metadata_cxi
    # pub.send_start(metadata_cxi)

    # Now I find the right filename to save the .cxi file in
    output_filename = make_output_filename(state)
    print('Saving data to', output_filename)

    # Next, I need to create and leave open that .cxi file
    # Energy has to be in eV
    state['metadata']['energy'] = state['metadata']['energy'] / constants.e
    with file_handling.create_cxi(output_filename, state['metadata']) as cxi_file:
        yield cxi_file


def make_output_filename(state):
    header = state['metadata']['header']
    header = '.'.join(header.split('.')[:-1])

    # TODO: actually find the region number and energy number
    if 'scanRegion' in state['metadata']:
        region_no = state['metadata']['scanRegion']
    else:
        region_no = 0
    if 'energyIndex' in state['metadata']:
        energy_no = state['metadata']['energyIndex']
    else:
        energy_no = 0

    return header + '_ccdframes_%d_%d.cxi' % (energy_no, region_no)


def process_dark_event(cxi_file, state, event, config):
    print('Processing dark event')

    # We add the (properly weighted) new dark image to the appropriate
    # set of darks
    dwell = event['data']['dwell']
    dark = t.as_tensor(np.array(event['data']['ccd_frame'], dtype=np.float32),
                       device=config['device'])
    if state['darks'][dwell] is None:
        state['darks'][dwell] = dark
    else:
        n_darks = state['n_darks'][dwell]
        state['darks'][dwell] = ((state['darks'][dwell] * n_darks + dark) /
                                 (n_darks + 1))
    state['n_darks'][dwell] += 1


def process_exp_event(cxi_file, state, event, pub, config):
    print('Processing exp event', event['data']['index'])

    if state['frame_cleaner'] is None:
        print('Creating the frame cleaner; all further darks will be disregarded')
        state['frame_cleaner'] = \
            image_handling.FastCCDFrameCleaner(
                [state['darks'][dwell] for dwell in state['dwells']])

    # state['position'] = np.array([-event['data']['xPos'], -event['data']['yPos']]) * 1e-6
    state['position'] = np.array([event['data']['xPos'], event['data']['yPos']]) * 1e-6 / state['px_size_real_space']

    # A correction for the shear in the probe positions
    state['position'] = np.matmul(config['shear'], state['position'])

    frame_numpy = None
    if event['data']['index'] != state['index']:
        # we've moved on, so we need to finalize the frame
        frame_numpy = finalize_frame(cxi_file, state, pub, config)
        state['index'] = event['data']['index']

    dwell = event['data']['dwell']
    dwell_idx = np.where(state['dwells'] == dwell)[0][0]

    raw_frame = t.as_tensor(
        np.array(event['data']['ccd_frame'], dtype=np.float32),
        device=config['device']
    )

    frame, mask = state['frame_cleaner'].process_frame(
        raw_frame,
        dwell_idx,
        include_overscan=False,
        med_width=config['background_median_width'],
        max_correction=config['max_overscan_correction'],
        min_correction=config['min_overscan_correction'],
        background_offset=config['background_offset'],
        cut_zeros=config['cut_zeros']
    )

    # We always need to do this
    state['current_exposures'][dwell].append(frame)
    state['current_masks'][dwell].append(mask)

    return frame_numpy


def finalize_frame(cxi_file, state, pub, config):
    # we first combine the frames

    all_dwells = []
    all_frames = []
    all_masks = []
    for dwell, frames in state['current_exposures'].items():
        masks = state['current_masks'][dwell]
        all_dwells.extend([dwell, ] * len(frames))
        all_frames.extend(frames)
        all_masks.extend(masks)

    if len(all_frames) == 0:
        return

    all_dwells = np.array(all_dwells)
    all_frames = t.stack(all_frames, dim=0)
    all_masks = t.stack(all_masks, dim=0)

    synthesized_frame, synthesized_mask = image_handling.combine_exposures(
        all_frames, all_masks, all_dwells)

    # After combining the frames, we resample them.

    if state['resampler'] is None:
        dummy_im = synthesized_frame
        state['resampler'] = image_handling.make_resampler(
            state['metadata'], config, dummy_im)

        # We should add the mask at this point
        mask = state['mask'].to(device=dummy_im.device)
        _, resampled_mask = state['resampler'].resample(dummy_im, masks=mask)
        file_handling.add_mask(cxi_file, resampled_mask)

        # And, one final correction, we need to update the pixel sizes on
        file_handling.update_cxi_metadata(cxi_file, state['metadata'])

    resampled_frame, resampled_mask = \
        state['resampler'].resample(synthesized_frame,
                                    masks=synthesized_mask)

    numpy_resampled_frame = resampled_frame.cpu().numpy()

    if state['start_sending_frames']:
        # identifier, data, index, posy, posx, metadata
        pos_x, pos_y = state['position']
        filepath = state['metadata']['header']
        filename = os.path.basename(filepath)
        identifier = filename[:12]
        index = state['index']
        pub.send_frame(
            identifier,
            numpy_resampled_frame,
            index,
            pos_y,
            pos_x
        )

    # Here we zero out the exposures and masks for the next frame
    state['current_exposures'] = {dwell: [] for dwell in state['dwells']}
    state['current_masks'] = {dwell: [] for dwell in state['dwells']}

    output_position = t.zeros(3)
    output_position[:2] = t.as_tensor(state['position'])

    file_handling.add_frame(cxi_file,
                            resampled_frame,
                            output_position,
                            mask=resampled_mask,
                            compression=config['compression'])

    return numpy_resampled_frame


def trigger_ptycho(state, config):
    if not config['use_prefect']:
        print('Not triggering ptychography, because I not using prefect')
    else:
        print('Triggering ptychography via prefect API')
        output_filename = make_output_filename(state)
        output_filename = '/'.join(output_filename.split('/')[5:])
        parameters = {
            'path': output_filename,
        }
        try:
            run_deployment(name='Reconstruct from .cxi/reconstruct-from-cxi',
                           parameters=parameters,
                           timeout=0)
        except:
            print('Failed to contact prefect')


def process_stop_event(cxi_file, state, event, pub, config):
    if state['metadata'] is None:
        # This is just a backup check, the upstream logic *should* prevent
        # this from getting triggered in the absence of a start event
        print('Never got a start event, not processing')
        return

    print('Processing stop event\n\n')
    finalize_frame(cxi_file, state, pub, config)
    pub.send_stop({})


def process_emergency_stop(cxi_file, state, pub, config):
    """This is triggered if no stop event happens, but we get a second
    start event. In this case, we just finish the scan without any of the
    info in the stop event.
    """
    print('Doing an emergency stop\n\n')
    finalize_frame(cxi_file, state, pub, config)


def run_data_accumulation_loop(
        cxi_file,
        state,
        event,
        sub,
        pub,
        config
):
    """Accumulates darks and frame data into a cxi file, until it completes
    """

    frames_for_illu_init = []
    while True:
        event = sub.recv_pyobj()

        if event['event'].lower().strip() == 'start':
            frames_for_illu_init.clear()
            process_emergency_stop(cxi_file, state, pub, config)
            return event  # pass the event back so we can start a new loop

        if event['event'].lower().strip() == 'stop':
            process_stop_event(cxi_file, state, event, pub, config)
            return None

        elif event['event'].lower().strip() == 'frame':
            if event['data']['ccd_mode'] == 'dark':
                process_dark_event(cxi_file, state, event, config)
            else:
                frame = process_exp_event(cxi_file, state, event, pub, config)

                if frame is not None:
                    frames_for_illu_init.append(frame)
                else:
                    continue

                n_frames_nominal = len(state['metadata_cxi']['translations'])
                n_frames_to_init_illu = int(n_frames_nominal * config['dp_fraction_for_illumination_init'])
                if len(frames_for_illu_init) == n_frames_to_init_illu:
                    state['start_sending_frames'] = True

                    illu, illu_mask = illumination_init.init_illumination(
                        np.array(frames_for_illu_init),
                        state['metadata_cxi']
                    )

                    # fpath = r"/homes/silvio/data/test/test_probe_mask/NS_230412297_ccdframes_0_0_offline_1withoutmask_2withmask.cxi"
                    # with h5py.File(fpath, 'r') as f:
                    #     illu = f['/entry_1/image_1/process_1/final_illumination'][()]

                    state['metadata_cxi']['illumination_real'] = illu.real.tolist()
                    state['metadata_cxi']['illumination_imag'] = illu.imag.tolist()
                    state['metadata_cxi']['illumination_mask'] = illu_mask.tolist()

                    pub.send_start(state['metadata_cxi'])

                    time.sleep(1.0)

                    # TODO: The index of the diffraction pattern is specified here as the index in the array.
                    # TODO: This is a simplification, as it assumes that the diffraction patterns are received in order,
                    # TODO: which is not necessarily true. However, left for now and can be fixed if the reconstruction
                    # TODO: looks funny.
                    for frame_idx, frame in enumerate(frames_for_illu_init):
                        pos_x, pos_y = state['metadata_cxi']['translations'][frame_idx]
                        filepath = state['metadata']['header']
                        filename = os.path.basename(filepath)
                        identifier = filename[:12]
                        index = frame_idx
                        pub.send_frame(
                            identifier,
                            frame,
                            index,
                            pos_y,
                            pos_x
                        )

                        time.sleep(0.05)

                    # TODO: write illumination and illumination mask into cxi file here.
                    cxi_file.create_dataset('entry_1/instrument_1/source_1/illumination', data=illu)
                    cxi_file.create_dataset('entry_1/instrument_1/source_1/probe_mask', data=illu_mask)


def main(argv=sys.argv):
    # I'm just using this so the script has a help printout
    parser = argparse.ArgumentParser(
        prog='process_live_data',
        description='Runs a program which listens for raw data being emitted by pystxmcontrol. It assembles and preprocesses this data, saving the results inot .cxi files and passing the cleaned data on for further analysis.')
    args = parser.parse_args()

    # TODO maybe add the ZMQ ports as args?

    # Now we get the config info to read the ZMQ ports
    config = config_handling.get_configuration()

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(config['subscription_port'])
    sub.setsockopt(zmq.SUBSCRIBE, b'')
    pub = PreprocessorStream()

    print('Listening for raw frames on', config['subscription_port'])
    print('Broadcasting processed frames on', config['broadcast_port'])

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
            config_handling.summarize_config(config)
            state = make_new_state()

            # Process start event returns a context manager for an open
            # cxi file with the appropriate metadata
            with process_start_event(state, event, pub, config=config) as cxi_file:
                # This loop will read in darks and exp data until it gets
                # a stop or start event, at which point it will return.
                start_event = run_data_accumulation_loop(cxi_file, state, event, sub, pub, config)

            # we wait to trigger the ptycho reconstruction until after saving
            # the file, to ensure that the file exists when the reconstruction
            # begins.
            trigger_ptycho(state, config)


if __name__ == '__main__':
    sys.exit(main())
