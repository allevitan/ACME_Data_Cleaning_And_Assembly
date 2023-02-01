import json
import importlib.resources
import numpy as np

__all__ = [
    'get_configuration',
    'add_processing_args',
    'blend_args_with_config',
]

def get_configuration():
    """This loads a configuration dictionary from the various config files.
    """
    
    # Then we load the configuration options
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

    if 'mask' not in config:
        config['mask'] = package_root.joinpath('default_mask.h5')

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
        '--resample', action='store_true',
        help="""If set, will resample the data to match the specified pixel size
        and output shape, ignoring the binning-factor""")
    parser.add_argument(
        '--no-resample', dest='resample', action='store_false',
        help="""If set, will bin by an integer factor over a specified output
        shape, ignoring the specified output-pixel-size.""")
    parser.set_defaults(resample=None)
    parser.add_argument(
        '--center', type=float, nargs=2,
        help='The center of the output data on the detector')
    parser.add_argument(
        '--output-pixel-size', type=int,
        help="""The design pixel size in real space.
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

    return config
