import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import random


def create_patches(image_arr, mask_arr, patch_size, output_path, original_name, background_fraction):
    
    # create the output dirs if it does not exist
    output_path.mkdir(parents=True, exist_ok = True)
    img_dir = output_path / "img_patches" 
    mask_dir = output_path / "mask_patches"
    img_dir.mkdir(parents=False, exist_ok = True)
    mask_dir.mkdir(parents=False, exist_ok = True)
    
    # pad the imgs
    
    image_arr = pad_to_patch_size(image_arr,patch_size)
    mask_arr = pad_to_patch_size(mask_arr,patch_size)
    
    # generate the coordinates to slice
    for y_start in range(0,image_arr.shape[0], patch_size):
        for x_start in range(0,image_arr.shape[1], patch_size):
            # define the ends
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            # slice from the original images
            img_patch = image_arr[y_start:y_end, x_start:x_end , :]
            mask_patch = mask_arr[y_start:y_end,x_start:x_end, :]
            
            # create Pillow Image objects from np arrays
            img_patch_obj = Image.fromarray(img_patch)
            mask_patch_obj = Image.fromarray(mask_patch)
            
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
            # else discard the imgs
            
    return "Done"


def pad_to_patch_size(img_arr, patch_size):
    # calculate the needed paddings
    pad_h = (patch_size - img_arr.shape[0] % patch_size) % patch_size
    pad_w = (patch_size - img_arr.shape[1] % patch_size) % patch_size
    
    # apply the paddings
    padded_arr = np.pad(img_arr,pad_width=((0,pad_h), (0, pad_w), (0,0)), mode='constant')
    
    # return the padded arr
    return padded_arr


if __name__ == "__main__":
    image_arr = np.zeros((1024,1024,3), dtype=np.uint8)
    mask_arr = image_arr.copy()
    mask_arr[400:600,400:600, :] = [255,255,0]
    PATCH_SIZE = 256
    OUTPUT_PATH = Path.cwd() / "output_imgs" / "patches"
    TEMP_OG_NAME = "some-name"
    BACKGROUND_FRACTION = 0.1 # fraction of empty patches that we are going to save
    
    
    # test the create patches function
    print(create_patches(image_arr, mask_arr, PATCH_SIZE, OUTPUT_PATH, TEMP_OG_NAME, BACKGROUND_FRACTION))
    

