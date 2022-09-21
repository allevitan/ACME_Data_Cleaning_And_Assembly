fname = '/home/abe/Dropbox (MIT)/Projects/7012RPI/20220826_Preprocessor/NS_220825066_ccdframes_0.stxm'
#output_fname = '/home/abe/Dropbox (MIT)/Projects/7012RPI/20220826_Preprocessor/NS_220825066_ccdframes_0.cxi
output_fname = 'test2.cxi'
conf_file = '/home/abe/Git/forked_cosmic2/configuration/default.json'



#import numpy as np
import jax.numpy as np
#import jax
import h5py
from matplotlib import pyplot as plt
from acme_data_cleaning import image_handling, file_handling
import json
import time

# So, the metadata is defined differently in this file, hmm...
#metadata = read_metadata_hdf5(fname)
#metadata = complete_metadata(metadata, conf_file)

with h5py.File(fname, 'r') as f:

    # currently, this doesn't include information about the detector
    # geometry!!
    metadata = file_handling.get_metadata_from_stxm(f)
    with file_handling.create_cxi(output_fname, metadata) as cxi_file:
        #print(list(cxi_file))
        #print(cxi_file['metadata'].asstr()[()])
        #print(cxi_file['entry_1/instrument_1/source_1/energy'])

    
        dark_frames = f["entry0/ccd0/dark"]
        low_darks = np.stack([np.array(dark_frames[str(i)], dtype=np.float32)
                            for i in range(0,50,2)])
        low_dark = np.mean(low_darks, axis=0)
        high_darks = np.stack([np.array(dark_frames[str(i)], dtype=np.float32)
                               for i in range(1,50,2)])
        high_dark = np.mean(high_darks, axis=0)
        
        low_dark = image_handling.map_raw_to_tiles(low_dark)
        high_dark = image_handling.map_raw_to_tiles(high_dark)

    
        exp_frames = f["entry0/ccd0/exp"]
        

        idx = 0
        while True:
            print(idx)
            t0 = time.time()
            try:
                low_raw = np.array(exp_frames[str(2*idx)], dtype=np.float32)
                high_raw = np.array(exp_frames[str(2*idx+1)], dtype=np.float32)
                idx += 1
            except KeyError:
                break
            print('load_time:', time.time() - t0)
            
            t0 = time.time()
            low, low_mask = image_handling.process_frame(low_raw, low_dark)
            high, high_mask = image_handling.process_frame(high_raw, high_dark)

            synthesized_frame, synthesized_mask = image_handling.combine_exposures(
                np.array([low,high]),
                np.array([low_mask, high_mask]),
                np.array([10,100]))

            translation = np.array([0,0,0])
            synthesized_frames = np.stack([synthesized_frame]*10)
            translations = np.stack([translation]*10)
            print('synthesis time:', time.time() - t0)
            t0 = time.time()
            file_handling.add_frame(cxi_file,
                                    synthesized_frame,
                                    translation,
                                    mask=synthesized_mask,
                                    compression='lzp')
            #file_handling.add_frames(cxi_file,
            #                         synthesized_frames,
            #                         translations,
            #                         masks=synthesized_mask)
            print('save time:', time.time() - t0)
