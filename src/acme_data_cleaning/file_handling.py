from contextlib import contextmanager
import h5py
import json
from jax import numpy as np
from functools import reduce


def read_metadata_from_stxm(stxm_file, add_detector_geometry=True):
    """Extracts the metadata from a .stxm file.

    The result is a python dictionary with a standard format

    The default detector geometry information is added as a temporary measure,
    until that information can be added to the .stxm files themselves.
    """
    metadata = json.loads(stxm_file["metadata/"].asstr()[()])

    # This is a hack, but this data should really be stored in the .stxm files
    # themselves. So, instead of adding a separate configuration file, I'm
    # going to leave this hack in until I can convince David to add this to the
    # .stxm files.
    if add_detector_geometry:
        if 'geometry' not in metadata:
            metadata['geometry'] = {}
        metadata['geometry']['psize'] = 30e-6
        metadata['geometry']['distance'] = 0.121

    # We also add a bit of extra metadata, if it doesn't already exist:
    if 'geometry' in metadata and 'psize' in metadata['geometry']:
        psize = float(metadata['geometry']['psize'])
        basis = np.array([[0,-psize,0],[-psize,0,0]])
        metadata['geometry']['basis_vectors'] = basis.tolist()

        # TODO: avoid hardcoding the detector shape
        # Probably, I should put this in the code that does the resampling
        # of the detector, whenever I write that. For now, it just needs
        # to happen somewhere
        distance = float(metadata['geometry']['distance'])
        corner = np.array([480*psize, 480*psize, distance])
        metadata['geometry']['corner_position'] = corner.tolist()

    return metadata


def read_translations_from_stxm(stxm_file):
    """Returns a numpy array with the translations defined in a .stxm file

    This uses the information stored in "entry0/instrument/sample<x/y>,
    rather than defined plan stored in the json metadata dictionary.
    """
    # We need to convert from um to meters
    x_pos = np.array(stxm_file['entry0/instrument/sample_x/data'],
                     dtype=np.float32) * 1e-6
    y_pos = np.array(stxm_file['entry0/instrument/sample_y/data'],
                     dtype=np.float32) * 1e-6
    return np.stack([x_pos, y_pos, np.zeros_like(x_pos)], axis=-1)


def read_exposures_from_stxm(stxm_file, n_exp_per_point=1):
    """A generator function to iterate through the exp frames in a .stxm file

    This returns an iterator which produces tuples of length n_exp_per_point,
    such that exposures taken at a common point are grouped together.    
    """
    exp_frames = stxm_file["entry0/ccd0/exp"]
    frame_indices = sorted([int(idx) for idx in list(exp_frames)])
    for idx in frame_indices[::n_exp_per_point]:
        yield tuple(np.array(exp_frames[str(idx + offset)], dtype=np.float32)
                    for offset in range(n_exp_per_point))

        
def read_chunked_exposures_from_stxm(stxm_file, chunk_size=10,
                                     n_exp_per_point=1):
    """A generator function to iterate through the exp frames in a .stxm file

    This returns an iterator which produces tuples of length n_exp_per_point,
    where each entry of the tuple is a set of chunk_size exposures, packed
    into a single array. This allows for some easy batch processing.
    """
    exp_frames = stxm_file["entry0/ccd0/exp"]
    frame_indices = sorted([int(idx) for idx in list(exp_frames)])

    # This will be a list of lists which contain the indices to process for
    # each chunk
    chunks = [frame_indices[idx:idx+chunk_size*n_exp_per_point:n_exp_per_point]
              for idx in frame_indices[::(chunk_size*n_exp_per_point)]]

    for chunk in chunks:
        # This reads in all the exposures for this chunk
        frames = [[np.array(exp_frames[str(idx + offset)], dtype=np.float32)
                   for offset in range(n_exp_per_point)]
                  for idx in chunk]

        # And this packages the exposures into 3D arrays
        yield tuple(np.stack(exposures) for exposures in zip(*frames))
        

def read_darks_from_stxm(stxm_file, n_exp_per_point=1):
    """A generator function to iterate through the dark frames in a .stxm file

    This returns an iterator which produces tuples of length n_exp_per_point,
    such that darks taken at a common point are grouped together.
    """
    dark_frames = stxm_file["entry0/ccd0/dark"]
    frame_indices = sorted([int(idx) for idx in list(dark_frames)])
    for idx in frame_indices[::n_exp_per_point]:
        yield tuple(np.array(dark_frames[str(idx + offset)], dtype=np.float32)
                    for offset in range(n_exp_per_point))


def read_mean_darks_from_stxm(stxm_file, n_exp_per_point=1):
    """Makes a tuple of average darks from the stored data in a .stxm file
    """
    darks_iterator = read_darks_from_stxm(stxm_file,
                                          n_exp_per_point=n_exp_per_point)
    # This isn't memory-efficient, but so be it. There shouldn't be more than
    # a few dozen darks, and we can rewrite it if need be
    darks = tuple(np.mean(np.stack(list(darks)), axis=0)
                  for darks in zip(*darks_iterator))
    return darks


# This dictionary (taken from Pablo Enfadaque's code) forms a shorhand for
# common locations within a .cxi file
groups = {
    "tomography": "/tomography/",
    "data": "/entry_1/data_1/",
    "geometry": "/entry_1/sample_1/geometry_1/",
    "source":  "/entry_1/instrument_1/source_1/", 
    "detector": "/entry_1/instrument_1/detector_1/",
    "process": "/entry_1/image_1/process_1/"
}

# This dictionary maps from entries that are present in the metadata json
# dictionary  of .stxm files to the locations where that metadata should
# be saved in a conformant .cxi file. This ultimately controls how the
# create_cxi function populates metadata in the new .cxi file
# Note: forward slashes in the key mean that that information is expected
# to be in a dictionary within a dictionnary
metadata_format = {
    #
    # Tomography experimental fields
    #
    "angles": [groups["tomography"] + "angles"],

    #
    # Ptychography experimental fields 
    #
    "energy": [groups["source"] + "energy"],
    "illumination" : [
        groups["source"] + "illumination", 
        groups["source"] + "probe",
        groups["detector"] + "probe",
        groups["detector"] + "data_illumination"
    ],
    "geometry/distance": [groups["detector"] + "distance"],
    #This is needed in SHARP although it is not used
    "geometry/corner_position": [groups["detector"] + "corner_position"],
    "geometry/basis_vectors": [groups["detector"] + "basis_vectors"],
    "illumination_distance": [groups["source"] + "illumination_distance"],
    "geometry/psize": [groups["detector"] + "x_pixel_size",
                       groups["detector"] + "y_pixel_size"],
    "illumination_mask" : [
        groups["source"] + "probe_mask",
        groups["detector"] + "probe_mask"
    ],
    "illumination_intensities": [
        groups["source"] + "illumination_intensities",
        groups["detector"] + "illumination_intensities"
    ],
    "near_field": [groups["detector"] + "near_field"],
    "pinhole_width": [groups["source"] + "pinhole_width"],
    "phase_curvature": [groups["source"] + "phase_curvature"],
}

@contextmanager
def create_cxi(filename, metadata):
    """Creates a .cxi file with the provided metadata, not including any data

    Accepts a filename for the new .cxi file and a metadata dictionary,
    which is expected to be in the same format used in the .stxm files. The
    values in this metadata dictionary will be saved in the appropriate
    locations within the .cxi file
    """
    # This way we piggyback on the context manager from h5py
    with h5py.File(filename, 'w') as cxi_file:
        cxi_file.create_dataset("cxi_version",data=140)

        # This ensures all the required groups are made (e.g. entry_1/data_1)
        for group, location in groups.items():
            cxi_file.create_group(location)

        # Here we create all the required groups for the metadata
        # Note that unlike Pablo's code, I do not attempt to store the
        # entries in the metadata dictionary which don't have a specified
        # location in the .cxi file. Instead, I save out the original metadata
        # dictionary in json format in it's own dataset, the same way it is
        # stored in the .stxm file. This keeps things tidy while ensuring that
        # no metadata is lost.
        cxi_file.create_dataset("metadata", data=json.dumps(metadata))
        for key, value in metadata.items():
            # I'm including this because David wants to store the detector
            # geometry information in it's own dictionary within the main
            # metadata dictionary, so I need to traverse at least one level
            # down. Sorry, future maintainers, unless you're David, in which
            # case I'm not sorry and you did this to yourself.
            if type(value) == dict:
                subkeys, subvalues = zip(*value.items())
                subkeys = [key + '/' + subkey for subkey in subkeys]
            else:
                subkeys = [key]
                subvalues = [value]
                
            for subkey, subvalue in zip(subkeys, subvalues):            
                if subkey in metadata_format:
                    for group in metadata_format[subkey]:
                        cxi_file.create_dataset(group, data = subvalue)

        # This is another hack, but unfortunately the energy is stored in
        # eV in the .stxm file, but needs to be in J for a .cxi file, so we
        # copy it over here
        if 'energy' in metadata:
            energy = metadata['energy'] * 1.60218e-19 # convert to Joules
            for group in metadata_format['energy']:
                cxi_file[group][()] = energy
        yield cxi_file


def add_frame(cxi_file, frame, translation, mask=None, intensity=None,
              compression='lzf'):
    """ Adds a single frame to an existing .cxi file

    Takes an h5py File object for the cxi file to write to, the image data
    from the frame to be added, and the probe position associated with
    the frame.
    
    Optionally, it takes an intensity value, if there was a measurement of the
    frame-to-frame incoming probe intensity.

    Additionally, it can take in a mask, which will be added to the overall
    detector mask stored in the file. This can be useful if certain shots have
    regions of bad data which should be added to the full detector mask.
    """
    masks = np.expand_dims(mask,0) if mask is not None else None
    intensities = np.array([intensity]) if intensity is not None else None
    
    return add_frames(cxi_file,
                      np.expand_dims(frame,0),
                      np.expand_dims(translation,0),
                      masks=masks,
                      intensities=intensities,
                      compression=compression)


def add_frames(cxi_file, frames, translations, masks=None, intensities=None,
               compression='lzf'):
    """ Adds a set of frames to an existing .cxi file

    Takes an h5py File object for the cxi file to write to, the image data
    from the frames to be added, and the positions of the frames.

    Optionally, it takes intensity values, if there was a measurement of the
    frame-to-frame incoming probe intensity.

    The "compression" argument will set the type of compression to use,
    by default 'lzf'. Once the first frame is added and the dataset is
    created, this no longer affects the choice of compression.

    TODO: actually implement the storing of a mask and the storing of frame
    by frame normalizations
    """

    # This is the case if no data has been saved yet, so we need to create
    # the datasets
    if groups['detector'] + 'data' not in cxi_file:
        # We start with the frame data
        chunk_shape = (1,) + frames.shape[-2:]
        max_shape = (None,) + frames.shape[-2:]
        cxi_file.create_dataset(groups['detector'] + 'data',
                                data=frames,
                                chunks=chunk_shape,
                                maxshape=max_shape,
                                compression=compression)
        # We need to also link to the data from this location
        softlink = h5py.SoftLink(groups['detector'] + 'data')
        cxi_file[groups['data'] + 'data'] = softlink

        # Next we do the translations
        cxi_file.create_dataset(groups['geometry'] + 'translation',
                                data=translations,
                                maxshape=(None,3))
        softlink = h5py.SoftLink(groups['geometry'] + 'translation')
        cxi_file[groups['data'] + 'translation'] = softlink
        cxi_file[groups['detector'] + 'translation'] = softlink

    else:
        data_group = cxi_file[groups['detector'] + 'data']
        data_group.resize((data_group.shape[0]+frames.shape[0],)
                          + data_group.shape[1:])
        data_group[-frames.shape[0]:] = frames

        translation_group = cxi_file[groups['geometry'] + 'translation']
        translation_group.resize((translation_group.shape[0]
                                  + translations.shape[0], 3))
        translation_group[-translations.shape[0]:] = translations
    
    if masks is not None:
        # Note that we only save one mask, which I find by summing all the
        # masks from individual shots.
        mask_to_save = np.sum(masks, axis=0).astype(np.bool_).astype(np.uint32)

        add_mask(cxi_file, mask_to_save)

def add_mask(cxi_file, mask):
    """Adds a mask to a .cxi file, melding it with any existing masks.
    """
    mask_to_save = mask.astype(np.uint32)
    if groups['detector'] + 'mask' not in cxi_file:
        cxi_file.create_dataset(groups['detector'] + 'mask',
                                data=mask_to_save)
    else:
        # If there's already a masked save, we include it 
        saved_mask = np.array(cxi_file[groups['detector'] + 'mask'])
        mask_to_save = np.logical_or(saved_mask, mask_to_save)
        mask_to_save = mask_to_save.astype(np.uint32)
        
        cxi_file[groups['detector'] + 'mask'][:,:] = mask_to_save
    
