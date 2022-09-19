fname = '/home/abe/Dropbox (MIT)/Projects/7012RPI/20220826_Preprocessor/NS_220825066_ccdframes_0.stxm'
output_fname = '/home/abe/Dropbox (MIT)/Projects/7012RPI/20220826_Preprocessor/NS_220825066_ccdframes_0.cxi'    
conf_file = '/home/abe/Git/forked_cosmic2/configuration/default.json'


import numpy as np
#import jax.numpy as np
#import jax
import h5py
from matplotlib import pyplot as plt
import image_handling


# So, the metadata is defined differently in this file, hmm...
#metadata = read_metadata_hdf5(fname)
#metadata = complete_metadata(metadata, conf_file)
print('hi')
with h5py.File(fname, 'r') as f:
    
    dark_frames = f["entry0/ccd0/dark"]
    low_dark = np.mean([np.array(dark_frames[str(i)], dtype=np.float32)
                        for i in range(0,50,2)],axis=0)
    high_dark = np.mean([np.array(dark_frames[str(i)], dtype=np.float32)
                         for i in range(1,50,2)], axis=0)
    
    exp_frames = f["entry0/ccd0/exp"]

    def process_frame(idx):
        frame = np.array(exp_frames[str(idx)], dtype=np.float32) - low_dark
        tiles = image_handling.map_raw_to_tiles(frame)
        low_tiles = image_handling.apply_overscan_background_subtraction(tiles)
        final = image_handling.map_tiles_to_frame(low_tiles)
        return final

    #frames = [process_frame(idx) for idx in range(20,40,2)]
    #print(frames[0])
    #plt.imshow(np.mean(frames,axis=0))
    #plt.show()

        
    frame = np.array(exp_frames['21'], dtype=np.float32)
    
    exp_tiles = image_handling.map_raw_to_tiles(frame)
    dark_tiles = image_handling.map_raw_to_tiles(high_dark)
    mask_tiles = image_handling.make_saturation_mask(exp_tiles)
    subtracted_tiles = exp_tiles - dark_tiles
    low_tiles = image_handling.apply_overscan_background_subtraction(
        subtracted_tiles)
    final = image_handling.map_tiles_to_frame(low_tiles, True)
    mask = image_handling.map_tiles_to_frame(mask_tiles, True)

    plt.imshow(final)
    plt.figure()
    plt.imshow(mask)
    plt.show()
    
    #data = tiles[:20,2:200].reshape((-1,12))

    #plt.imshow(data)
    #plt.show()
    #data = np.mean(data, axis=-1) - np.min(data[:,::11],axis=-1)
    # So, if we pick two pixels at random and use the smallest of the two
    # as a background subtraction, we will typically find that that value
    # is about 1.6 less than the mean, confirmed for both long and short
    # xposures
    # If we only use the two background rows, the correction is 0.55
    #print(np.mean(data)) 
    #plt.hist(data, bins=20)
    #plt.show()
    #exit()

    
    #plt.imshow(np.minimum(final[:,:700],4000))
    #plt.figure()

    frame = np.array(exp_frames['21'], dtype=np.float32)
    exp_tiles = image_handling.map_raw_to_tiles(frame)
    back_tiles = image_handling.map_raw_to_tiles(high_dark)

    mask_tiles = image_handling.make_saturation_mask(exp_tiles)

    mask = image_handling.map_tiles_to_frame(mask_tiles)

    im = image_handling.map_tiles_to_frame(exp_tiles)# - back_tiles)
    plt.imshow(im)
    plt.figure()
    plt.imshow(mask)
    plt.show()

    exit()
    frame = np.stack([np.array(exp_frames[str(i)], dtype=np.float32) - high_dark
                     for i in range(21,181,2)])
    frame = np.mean(frame, axis=0)
    #frame = np.array(exp_frames['21'], dtype=np.float32) - high_dark
    
    tiles = image_handling.map_raw_to_tiles(frame)
    


    tiles = image_handling.apply_overscan_background_subtraction(tiles)

    # THis was an attempt at handling the "shadow"
    #low_tiles = tiles > 3500
    #shadow_factor=4
    #shadow_factor = 0.004
    #tiles.at[:46].set(tiles[:46] + shadow_factor * low_tiles[2:48])
    #tiles = tiles.at[50:].set(tiles[50:] + shadow_factor * low_tiles[48:-2])
    #tiles = tiles.at[96:96+46].set(tiles[96:96+46:] + shadow_factor * low_tiles[96+2:96+48])
    #tiles = tiles.at[96+50:].set(tiles[96+50:] + shadow_factor * low_tiles[96+48:-2])

    #ratio =  20
    #for idx in range(12):
    #    tiles = tiles.at[idx*16:(idx+1)*16,1:-1,1:-1].set(
    #        tiles[idx*16:(idx+1)*16,1:-1,1:-1] - ratio * 
    #        np.mean(saturation_tiles[idx*16:(idx+1)*16,1:-1,1:-1], axis=-3, keepdims=True))
        
    # It seems like the "wings" are 40 * exposure_ratio.

    a = np.mean(saturation_tiles[-48:-48+16,400:],axis=-3)
    b = np.mean(tiles[-48+8:-48+16,400:],axis=-3)
    plt.scatter(a[2:-2,2:-2].ravel(),b[2:-2,2:-2].ravel())
    plt.figure()
    plt.imshow(a)
    plt.figure()
    plt.imshow(b)
    plt.figure()
    #plt.imshow(np.mean(low_tiles[-48:-48+16,400:],axis=-3) / np.mean(tiles[-48+8:-48+16,400:],axis=-3))
    #plt.figure()
    final = image_handling.map_tiles_to_frame(tiles,True)
    
    plt.imshow(final[:,:700])
    plt.show()
    
    #plt.imshow(
    #plt.imshow(np.array(darfk_frames['2'], dtype=np.float32))
    
    #print(list(exp_frames))
