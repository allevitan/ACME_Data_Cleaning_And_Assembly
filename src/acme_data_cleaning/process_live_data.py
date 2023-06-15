import time
import asyncio
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
# from splash_flows_globus.orchestration.flows.bl7012.move import process_new_file

from prefect.client import OrionClient
from prefect.orion.schemas.core import Flow, FlowRun
from prefect.orion.schemas.states import Scheduled
from prefect.deployments import run_deployment


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
        'exp_type': None,
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
        'start_sending_frames': False,
        'frames': [],
        'illu_initialization_done': False,
        'identifier': None,
    }


def calculate_interpolated_det_px_size(distance, npx_det, wavelength, px_size_real_space):
    det_px_size = 2 * distance / npx_det
    det_px_size *= np.tan(np.arcsin(wavelength / (2 * px_size_real_space)))
    return det_px_size


def calculate_px_size_real_space(px_size_det, npx_det, distance, wavelength):
    alpha = np.arctan(px_size_det * npx_det / (2 * distance))
    px_size = wavelength / (2 * np.sin(alpha))
    return px_size


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

    state['metadata'] = event['metadata']
    state['identifier'] = make_dataset_name(state)
    if state['metadata']['double_exposure']:
        state['exp_type'] = 'double'
        state['dwells'] = np.array([state['metadata']['dwell2'], state['metadata']['dwell1']])
        print('Start event indicates double exposures with exposure times', state['dwells'])
    elif 'n_repeats' in state['metadata'] and state['metadata']['n_repeats']!=1:
        state['exp_type'] = 'repeated'
        state['dwells'] = np.array([state['metadata']['dwell1']])
        print('Start event indicates repeated exposures,', state['metadata']['n_repeats'],
              'repeats of', state['metadata']['dwell1'], 'ms exposures')

    else:
        state['exp_type'] = 'single'
        state['dwells'] = np.array([state['metadata']['dwell1']])
        print('Start event indicates single exposures.')


    if config['use_all_exposures'].lower().strip() == 'auto':
        state['use_all_exposures'] = (True if state['exp_type'] =='repeated'
                                      else False)
        print('Will include data from all non-saturated exposures, best for repeated exposures with identical dwell times.')
    else:
        state['use_all_exposures'] = config['use_all_exposures']
        print('Will only include data from the longest non-saturated exposure, best for double exposures with differing dwell times')


    # We need to add the basis to the metadata
    # psize is given in um and change to m here.
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

    state['metadata']['translations'] = (np.array(state['metadata']['translations']) * 1e-6).tolist()
    state['metadata']['output_frame_width'] = config['output_pixel_count']

    state['metadata']['detector_distance'] = state['metadata']['geometry']['distance']
    state['metadata']['energy'] = state['metadata']['energy'] * constants.e

    wavelength = constants.h * constants.c / state['metadata']['energy']
    if config['interpolate']:
        state['px_size_real_space'] = config['output_pixel_size'] * 1e-9
        distance = state['metadata']['detector_distance']
        npx = config['output_pixel_count']

        psize_interpolated = calculate_interpolated_det_px_size(
            distance,
            npx,
            wavelength,
            config['output_pixel_size'] * 1e-9
        )

        state['metadata']['x_pixel_size'] = psize_interpolated
        state['metadata']['y_pixel_size'] = psize_interpolated

    else:
        state['px_size_real_space'] = calculate_px_size_real_space(
            psize,
            state['metadata']['output_frame_width'],
            state['metadata']['detector_distance'],
            wavelength
        )

        state['metadata']['x_pixel_size'] = psize
        state['metadata']['y_pixel_size'] = psize

    # Translations are originally given as (y, x).
    state['metadata']['translations'] = np.array(state['metadata']['translations'])

    # Translations are changed to (x, y) to apply the shear.
    translations_tmp = state['metadata']['translations'].copy()
    state['metadata']['translations'][:, 0] = translations_tmp[:, 1]
    state['metadata']['translations'][:, 1] = translations_tmp[:, 0]
    state['metadata']['translations'] = np.dot(config['shear'], state['metadata']['translations'].T).T

    state['metadata']['translations'] = state['metadata']['translations'].tolist()

    # identifier = make_dataset_name(state)
    state['metadata']['identifier'] = state['identifier']

    metadata_cxi = copy.deepcopy(state['metadata'])
    state['metadata_cxi'] = metadata_cxi

    # Now I find the right filename to save the .cxi file in
    output_filename = make_output_filename(state)
    print('Saving data to', output_filename)

    # Next, I need to create and leave open that .cxi file
    # Energy has to be in eV
    state['metadata']['energy'] = state['metadata']['energy'] / constants.e
    with file_handling.create_cxi(output_filename, state['metadata']) as cxi_file:
        yield cxi_file


def make_dataset_name(state):
    # header = state['metadata']['header']
    # header = '.'.join(header.split('.')[:-1])
    basename = os.path.basename(state['metadata']['header'])
    basename = basename.replace('.stxm', '')

    # TODO: actually find the region number and energy number
    if 'scanRegion' in state['metadata']:
        region_no = state['metadata']['scanRegion']
    else:
        region_no = 0
    if 'energyIndex' in state['metadata']:
        energy_no = state['metadata']['energyIndex']
    else:
        energy_no = 0

    return basename + '_ccdframes_%d_%d' % (energy_no, region_no)


def make_output_filename(state):
    folder = os.path.dirname(state['metadata']['header'])
    dataset_name = state['identifier']

    return os.path.join(folder, f"{dataset_name}.cxi")


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
        state['frame_cleaner'] = image_handling.FastCCDFrameCleaner([state['darks'][dwell] for dwell in state['dwells']])

    # Change the position from um to m.
    state['position'] = np.array([event['data']['xPos'], event['data']['yPos']]) * 1e-6  # / state['px_size_real_space']

    # A correction for the shear in the probe positions
    state['position'] = np.matmul(config['shear'], state['position'])

    frame_numpy = None
    if event['data']['index'] != state['index']:
        # we've moved on, so we need to finalize the frame
        finalize_frame(cxi_file, state, pub, config)
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
        if 'streak_mask' in state:
            numpy_resampled_frame = numpy_resampled_frame * state['streak_mask']

        pos_x, pos_y = state['position']
        identifier = state['identifier']
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

    state['frames'].append(numpy_resampled_frame)

    n_frames_nominal = len(state['metadata_cxi']['translations'])
    n_frames_to_init_illu = int(n_frames_nominal * config['dp_fraction_for_illumination_init'])
    if len(state['frames']) == n_frames_to_init_illu:
        send_start_and_existing_frames(cxi_file, pub, state, config)


def process_stop_event(cxi_file, state, event, pub, config):
    if state['metadata'] is None:
        # This is just a backup check, the upstream logic *should* prevent
        # this from getting triggered in the absence of a start event
        print('Never got a start event, not processing')
        return

    print('Processing stop event\n\n')
    finalize_frame(cxi_file, state, pub, config)


def process_abort_event(cxi_file, state, event, pub, config):
    if state['metadata'] is None:
        # This is just a backup check, the upstream logic *should* prevent
        # this from getting triggered in the absence of a start event
        print('Never got a start event, not processing')
        return

    print('Processing abort event\n\n')
    finalize_frame(cxi_file, state, pub, config)


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

    while True:
        event = sub.recv_pyobj()

        if event['event'].lower().strip() == 'start':
            process_emergency_stop(cxi_file, state, pub, config)
            return event  # pass the event back so we can start a new loop

        if event['event'].lower().strip() == 'stop':
            process_stop_event(cxi_file, state, event, pub, config)
            return 'stop'

        if event['event'].lower().strip() == 'abort':
            process_abort_event(cxi_file, state, event, pub, config)
            return 'abort'

        elif event['event'].lower().strip() == 'frame':
            if event['data']['ccd_mode'] == 'dark':
                process_dark_event(cxi_file, state, event, config)
            else:
                process_exp_event(cxi_file, state, event, pub, config)


def send_start_and_existing_frames(cxi_file, pub, state, config):
    """
    Everything below here will init the illumination and illumination mask, writes both into the cxi file,
    then sends the metadata (including the illumination) via zmq and then sends the frames that have been
    received so far. It is really no good practice at all but since it works it is left as it is for now.
    """

    state['start_sending_frames'] = True

    if state['illu_initialization_done']:
        return

    illu, illu_mask = illumination_init.init_illumination(
        np.array(state['frames']),
        state['metadata_cxi'],
        config
    )

    if config['use_streak_mask']:
        state['streak_mask'] = illumination_init.create_streak_mask(illu_mask)

    print("Initialized illumination using {} of {} frames".format(len(state['frames']), len(state['metadata_cxi']['translations'])))
    cxi_file.create_dataset('entry_1/instrument_1/source_1/illumination', data=illu)
    cxi_file.create_dataset('entry_1/instrument_1/detector_1/probe_mask', data=illu_mask.astype(int))
    print("Wrote illumination and illumination mask into cxi file.")

    state['metadata_cxi']['illumination_real'] = illu.real.tolist()
    state['metadata_cxi']['illumination_imag'] = illu.imag.tolist()
    state['metadata_cxi']['illumination_mask'] = illu_mask.tolist()
    state['metadata_cxi']['dp_fraction_for_illumination_init'] = config['dp_fraction_for_illumination_init']

    print("Sending start event.")
    pub.send_start(state['metadata_cxi'])
    time.sleep(config['zmq_metadata_timeout'])

    # TODO: The index of the diffraction pattern is specified here as the index in the array.
    # TODO: This is a simplification, as it assumes that the diffraction patterns are received in order,
    # TODO: which is not necessarily true. However, left for now and can be fixed if the reconstruction
    # TODO: looks funny.
    print("Sending the frames that have been received so far.")
    for frame_idx, frame in enumerate(state['frames']):
        if 'streak_mask' in state:
            frame = frame * state['streak_mask']

        pos_x, pos_y = state['metadata_cxi']['translations'][frame_idx]
        identifier = state['identifier']
        index = frame_idx
        pub.send_frame(
            identifier,
            frame,
            index,
            pos_y,
            pos_x
        )

        time.sleep(config['zmq_data_timeout'])

    state['illu_initialization_done'] = True


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
                return_code = run_data_accumulation_loop(cxi_file, state, event, sub, pub, config)

            if return_code == 'stop':
                start_event = None
                pub.send_stop({'identifier': state['identifier']})
            elif return_code == 'abort':
                pub.send_abort({'identifier': state['identifier']})
                start_event = None
            else:
                start_event = return_code

            # Local cdtools reconstruction at cosmic machines, if wanted.
            if config['prefect_cdtools_local']:
                try:
                    prefect_cdtools_local(state, config)
                except Exception as e:
                    print("Prefect local cdtools reconstruction failed due to: {}".format(e))

            # Local ptychocam reconstruction at cosmic machines, if wanted.
            if config['prefect_ptychocam_local']:
                try:
                    prefect_ptychocam_local(state, config)
                except Exception as e:
                    print("Prefect local ptychocam reconstruction failed due to: {}".format(e))

            # Data transfer to NERSC
            if config["prefect_nersc_transfer"]:
                try:
                    prefect_nersc_transfer(state, config)
                except Exception as e:
                    print("NERSC data transfer failed due to: {}".format(e))


def prefect_cdtools_local(state, config):
    print('[prefect]: Initializing local cdtools reconstruction.')
    output_filename = make_output_filename(state)

    out_of_focus_distance_m = config['ref_defocus_length_um'] * 1e-6 * state['metadata']['energy'] / config['ref_defocus_energy_eV']

    parameters = {
        'path': output_filename,
        "run_split_reconstructions": config["prefect_cdtools_local_run_split_reconstructions"],
        "n_modes": config["prefect_cdtools_local_n_modes"],
        "oversampling_factor": config["prefect_cdtools_local_oversampling_factor"],
        "propagation_distance": float(out_of_focus_distance_m),
        "simulate_probe_translation": config["prefect_cdtools_local_simulate_probe_translation"],
        "n_init_rounds": config["prefect_cdtools_local_n_init_rounds"],
        "n_init_iter": config["prefect_cdtools_local_n_init_iter"],
        "n_final_iter": config["prefect_cdtools_local_n_final_iter"],
        "translation_randomization": config["prefect_cdtools_local_translation_randomization"],
        "probe_initialization": config["prefect_cdtools_local_probe_initialization"],
        "init_background": config["prefect_cdtools_local_init_background"],
        "probe_support_radius": config["prefect_cdtools_local_probe_support_radius"],
    }

    deployment = config['prefect_cdtools_local_deployment']
    run_deployment(name=deployment,
                   parameters=parameters,
                   timeout=0)


def prefect_ptychocam_local(state, config):
    print("[prefect]: Initializing local ptychocam reconstruction.")
    parameters = {
        'cxipath': make_output_filename(state),
        'n_gpus': config['prefect_ptychocam_local_n_gpus'],
        'n_iterations': config['prefect_ptychocam_local_n_iterations'],
        'period_illu_refine': config['prefect_ptychocam_local_period_illu_refine'],
        'period_bg_refine': config['prefect_ptychocam_local_period_bg_refine'],
        'use_illu_mask': config['prefect_ptychocam_local_use_illu_mask'],
    }

    run_deployment(
        name=config['prefect_ptychocam_local_deployment'],
        parameters=parameters,
        timeout=0
    )


def prefect_nersc_transfer(state, config):
    print("[prefect]: Initializing data transfer to NERSC.")
    year_2digits = state['identifier'][3:5]
    year_4digits = '20' + year_2digits
    month = state['identifier'][5:7]
    day = state['identifier'][7:9]
    filepath = f"{year_4digits}/{month}/{year_2digits}{month}{day}/{state['identifier']}.cxi"

    prefect_api_url = os.getenv('PREFECT_API_URL')
    prefect_api_key = os.getenv('PREFECT_API_KEY')
    prefect_deployment = config['prefect_nersc_transfer_deployment']

    asyncio.run(
        prefect_start_flow(
            prefect_api_url,
            prefect_deployment,
            filepath,
            api_key=prefect_api_key
        )
    )


async def prefect_start_flow(prefect_api_url, deployment_name, file_path, api_key=None):
    client = OrionClient(prefect_api_url, api_key=api_key)
    deployment = await client.read_deployment_by_name(deployment_name)
    flow_run = await client.create_flow_run_from_deployment(
        deployment.id,
        name=os.path.basename(file_path),
        parameters={"file_path": file_path},
    )
    return flow_run


if __name__ == '__main__':
    sys.exit(main())
