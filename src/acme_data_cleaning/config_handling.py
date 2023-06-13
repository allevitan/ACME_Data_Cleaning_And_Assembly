import json
import importlib.resources
import importlib_resources
import numpy as np
import torch as t
import h5py


__all__ = [
    'get_configuration',
    'add_processing_args',
    'blend_args_with_config',
    'summarize_config',
    'load_mask'
]

def get_configuration():
    """This loads a configuration dictionary from the various config files.
    """
    
    # Then we load the configuration options
    # package_root = importlib.resources.files('acme_data_cleaning')
    package_root = importlib_resources.files('acme_data_cleaning')
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

    if 'mask' not in config:
        config['mask'] = package_root.joinpath('default_mask.h5')

    # This is how h5py expects the input, so I convert it upon load
    if config['compression'].lower().strip() == 'none':
        config['compression'] = None
    else:
        config['compression'] = config['compression'].lower().strip()

    return config


def add_processing_args(parser):
    """Adds all the arguments relevant to processing data frames to an argparser
    """
    
    parser.add_argument(
        '--mask','-m', type=str,
        help='A custom mask file to use, if the default is not appropriate')
    parser.add_argument(
        '--chunk_size','-c', type=int,
        help='The chunk size to use in the output .cxi files, default is 10.')
    parser.add_argument(
        '--compression', type=str,
        help="""What hdf5 compression filter to use on the output CCD data.
        Default is lzf.""")
    parser.add_argument(
        '--succinct', action='store_true',
        help='Turns off verbose output')
    parser.add_argument(
        '--device', type=str,
        help='The torch device to run on, by default "cuda:0".')
    parser.add_argument(
        '--interpolate', action='store_true',
        help="""If set, will use interpolation to resample the data, matching
        the specified output-pixel-size and output-pixel-count but ignoring the
        binning-factor""")
    parser.add_argument(
        '--no-interpolate', dest='interpolate', action='store_false',
        help="""If set, will bin by an integer binning-factor over a specified
        output-pixel-count, ignoring the specified output-pixel-size.""")
    parser.set_defaults(interpolate=None)
    parser.add_argument(
        '--center', type=float, nargs=2,
        help='The center of the output data on the detector')
    parser.add_argument(
        '--output-pixel-size', type=int,
        help="""The designated pixel size (nm) in real space.
        Ignored if no-resample is set. Defaults to the value in the config
        file, or if not set, the value arising from the nearest detector
        edge""")
    parser.add_argument(
        '--output-pixel-count', type=int,
        help='the number of pixels across the output diffraction patterns')
    parser.add_argument(
        '--binning-factor', type=int,
        help="""The factor by which to bin the data (e.g. 1x, 2x,...).
        Ignored if resample is set.""")

    return parser


def blend_args_with_config(args, config):
    """Adds all the manually set command line args to the configuration

    This will overwrite any preset configuration values if the command line
    arg was set, but default to the values in the config file
    """

    for arg_name in args.__dict__:
        arg = args.__dict__[arg_name] 
        if arg_name not in config or arg is not None:
            config[arg_name] = arg

    # We have to redo this in case compression was explicitly set
    if config['compression'] is not None and \
       config['compression'].lower().strip() == 'none':
        config['compression'] = None
    elif config['compression'] is not None:
        config['compression'] = config['compression'].lower().strip()

    return config


def summarize_config(config):

    print('Compression to be used is:', config['compression'])
    
    if config['interpolate']:
        print('Files will be processed using the interpolating resampler.')
        print('Detector center is defined at [%0.2f, %0.2f]'
              % tuple(config['center']))
        print('Output pixel size in real space is %0.2f nm'
              % config['output_pixel_size'])
        print('Output pixel count is %d' % config['output_pixel_count'])
        
    else:
        print('Files will be processed using the non-interpolating resampler.')
        print('Detector center is defined at [%d, %d]'
              % tuple(config['center']))
        print('Binning factor is %d' % config['binning_factor'])
        print('Output pixel count is %d' % config['output_pixel_count'])


def load_mask(config):
    """Load the specified mask from a file.

    This falls back on the default mask, which is always included in the
    config dictionary if not explicitly defined. This may not work for
    zipped packages, I don't know.
    """
    with h5py.File(config['mask'], 'r') as f:
        default_mask = t.as_tensor(np.array(f['mask']), device=config['device'])

    return default_mask
