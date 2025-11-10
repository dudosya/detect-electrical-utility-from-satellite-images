import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import random

import logging
logger = logging.getLogger(__name__)


def create_patches(image_arr: np.ndarray, mask_arr: np.ndarray, patch_size: int, output_path: str | Path, original_name: str, background_fraction: float)->None:
    """ Generate and save patches from a large image and its corresponding mask
    
    This function takes a large image and mask, pads them to be perfectly divisible
    by the patch size, and then extracts smaller corresponding patches. it filters the patches,
    keeping all patches that contain labeled objects and a random sampling of patches that are 
    purely background. The resulting image and mask patches are  saved to separate subdirectories.
    

    Args:
        image_arr (np.ndarray): The source image as a NumPy array, expected in (H,W,C) format
        mask_arr (np.ndarray): The corresponding mask image as a Numpy array, expected with the same height and width as the image_arr
        patch_size (int): The side length of the square patches to generate ie 256 for 256x256 patches
        output_path (str | Path): The root directory where "img_patches" and "mask_patches" subfolders will be created and populated
        original_name (str): A unique identifier for the source image,, used as a prefix for the output patch filenames ie "image_01"
        background_fraction (float): probablity from 0.0 to 1.0 of keeping a patch if its corresponding mask is empty ie background
    """
    
    # create the output dirs if it does not exist
    output_path.mkdir(parents=True, exist_ok = True)
    img_dir = output_path / "img_patches" 
    mask_dir = output_path / "mask_patches"
    img_dir.mkdir(parents=False, exist_ok = True)
    mask_dir.mkdir(parents=False, exist_ok = True)
    
    
    logger.debug(f"img arr shape BEFORE padding: {image_arr.shape}")
    logger.debug(f"mask arr shape BEFORE padding: {mask_arr.shape}")
    # pad the imgs
    image_arr = pad_to_patch_size(image_arr,patch_size)
    mask_arr = pad_to_patch_size(mask_arr,patch_size)
    logger.debug(f"img arr shape AFTER padding: {image_arr.shape}")
    logger.debug(f"mask arr shape AFTER padding: {mask_arr.shape}")
    
    # generate the coordinates to slice
    for y_start in range(0,image_arr.shape[0], patch_size):
        for x_start in range(0,image_arr.shape[1], patch_size):
            # define the ends
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            # slice from the original images
            img_patch = image_arr[y_start:y_end, x_start:x_end , :]
            mask_patch = mask_arr[y_start:y_end,x_start:x_end]
            
            # create Pillow Image objects from np arrays
            img_patch_obj = Image.fromarray(img_patch)
            mask_patch_obj = Image.fromarray(mask_patch,mode='L')
            
            
            # define names for the objects
            img_name = f"{original_name}_img_{y_start}_{x_start}.png"
            mask_name = f"{original_name}_mask_{y_start}_{x_start}.png"
            
            # full path names
            img_fp = img_dir / img_name
            mask_fp = mask_dir/ mask_name
            
            # check if the mask patch is not all zeros 
            # or if the mask patch is empty, then randomly save some of them
            if np.any(mask_patch) or random.random() < background_fraction:
                # save the images
                img_patch_obj.save(img_fp)
                mask_patch_obj.save(mask_fp)
                
                #logger
                logger.info(f"Saved {img_fp}")
                logger.info(f"Saved {mask_fp}")
                
                
            else:
                # logger
                logger.debug(f"Discarded {img_fp}")
                logger.debug(f"Discarded {mask_fp}")


def pad_to_patch_size(img_arr: np.ndarray, patch_size: int)-> np.ndarray:
    """Pad a numpy array to be divisible by a patch size
    
    Adds constant padding (zeros) to the bottom and right edges of an array
    until its height and widths are perfectly divisible by the 
    given patch size

    Args:
        img_arr (np.ndarray): The input array to pad, expected to have shape
            of (H,W,C)
        patch_size (int): The target patch size. The output array's
            height and width will be a multiple of this value

    Returns:
        np.ndarray: A new, padded array. If not padding was needed,
            a copy of the original array is returned
    """
    # calculate the needed paddings
    pad_h = (patch_size - img_arr.shape[0] % patch_size) % patch_size
    pad_w = (patch_size - img_arr.shape[1] % patch_size) % patch_size
    
    # apply the paddings
    if img_arr.ndim == 3:
        padded_arr = np.pad(img_arr,pad_width=((0,pad_h), (0, pad_w), (0,0)), mode='constant')
    elif img_arr.ndim == 2:
        padded_arr = np.pad(img_arr,pad_width=((0,pad_h), (0, pad_w)), mode='constant')
    else:
        raise ValueError("image arr is expected to have two types of dims: (H,W,C) or (H,W)")
    
    # return the padded arr
    return padded_arr


if __name__ == "__main__":
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-cfg', required=True, help="name of the config yaml file")
    args = parser.parse_args()
    CONFIG_PATH = Path.cwd() / f"{args.config}.yaml"
    
    # open and load the config dict
    with open(CONFIG_PATH,'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # assign the config params to vars
    PATCH_SIZE = cfg_dict["preprocessing"]["patch_size"]
    BACKGROUND_FRACTION = cfg_dict["preprocessing"]["background_fraction"]
    OUTPUT_PATH = Path.cwd() / cfg_dict["paths"]["output_dir"]
    
    # empty image arr
    image_arr = np.zeros((4800,3200,3), dtype=np.uint8)
    
    # mask arr in the middle
    mask_arr = np.zeros((4800,3200), dtype=np.uint8)
    mask_arr[2000:2800,1200:2000] = 255
    mask_arr[2200:2600,1400:1800] = 128

    # it will be the real filename later on
    TEMP_OG_NAME = "some-name"
    
    # test the create patches function
    create_patches(image_arr, mask_arr, PATCH_SIZE, OUTPUT_PATH, TEMP_OG_NAME, BACKGROUND_FRACTION)
    

