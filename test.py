fname = '/home/abe/Dropbox (MIT)/Projects/7012RPI/20220826_Preprocessor/NS_220825066_ccdframes_0.stxm'
output_fname = '/home/abe/Dropbox (MIT)/Projects/7012RPI/20220826_Preprocessor/NS_220825066_ccdframes_0.cxi'    
conf_file = '/home/abe/Git/forked_cosmic2/configuration/default.json'


import numpy as np
#import jax.numpy as np
#import jax
import h5py
from matplotlib import pyplot as plt
from acme_data_cleaning import image_handling
import json

# So, the metadata is defined differently in this file, hmm...
#metadata = read_metadata_hdf5(fname)
#metadata = complete_metadata(metadata, conf_file)

with h5py.File(fname, 'r') as f:

    metadata = json.loads(np.array(f['metadata'])[()])
    #print(metadata.keys())
    #print(metadata['isDoubleExp'])
    #print(metadata['double_exposure'])
    #print(metadata['dwell1'])
    #print(metadata['dwell2'])
    #exit()
    
    dark_frames = f["entry0/ccd0/dark"]
    low_dark = np.mean([np.array(dark_frames[str(i)], dtype=np.float32)
                        for i in range(0,50,2)],axis=0)
    high_dark = np.mean([np.array(dark_frames[str(i)], dtype=np.float32)
                         for i in range(1,50,2)], axis=0)

    low_dark = image_handling.map_raw_to_tiles(low_dark)
    high_dark = image_handling.map_raw_to_tiles(high_dark)
    print(np.min(image_handling.map_tiles_to_frame(low_dark)[:,:600]))
    plt.imshow(image_handling.map_tiles_to_frame(low_dark))
    plt.show()

    
    exp_frames = f["entry0/ccd0/exp"]

    def process_frame(idx, dark):
        frame = np.array(exp_frames[str(idx)], dtype=np.float32)
        return image_handling.process_frame(frame, dark)

    low, low_mask = process_frame(20, low_dark)
    high, high_mask = process_frame(21, high_dark)
    synthesized_frame, synthesized_mask = image_handling.combine_exposures(
        np.array([low,high]),
        np.array([low_mask, high_mask]),
        np.array([10,100]))

    
    low_frame = np.array(exp_frames[str(20)], dtype=np.float32)
    high_frame = np.array(exp_frames[str(21)], dtype=np.float32)

    low_frames = np.stack([low_frame]*1)
    low_darks = np.stack([low_dark]*1)
    high_frames = np.stack([high_frame]*1)
    high_darks = np.stack([high_dark]*1)
    import time
    t0 = time.time()
    for i in range(100):
        low, low_mask = image_handling.process_frame(low_frames, low_darks)
        
        high, high_mask = image_handling.process_frame(high_frames, high_darks)
        synthesized_frame, synthesized_mask = image_handling.combine_exposures(
            np.stack([low,high]),
            np.stack([low_mask, high_mask]),
            np.array([10,100]))

    low = low[0]
    low_mask = low_mask[0]
    high = high[0]
    high_mask = high_mask[0]
    synthesized_frame = synthesized_frame[0]
    synthesized_mask = synthesized_mask[0]
    print('time per frame:',( time.time() - t0)/100)



    plt.figure()
    plt.imshow(low)
    plt.figure()
    plt.imshow(low_mask)
    plt.figure()
    plt.imshow(high)
    plt.figure()
    plt.imshow(high_mask)
    plt.figure()
    plt.imshow(synthesized_frame)
    plt.figure()
    plt.imshow(synthesized_mask)
    plt.show()
    
    
