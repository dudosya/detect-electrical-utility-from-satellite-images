import torch
import torchvision
from pathlib import Path
from PIL import Image

import logging
logger = logging.getLogger(__name__)

class SatteliteImgsDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths, transforms):
        super().__init__()
        
        # validate the input
        check_img_mask_paths(img_paths,mask_paths)
        assert isinstance(transforms,torchvision.transforms.transforms.Compose), "The transform is not a subclass of torchvision.transforms.transforms.Compose"
        
        # put the stuff here
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_paths)
    
    
    def __getitem__(self, index):
        # validate arg
        assert isinstance(index,int) and (0 <= index < len(self.img_paths)), "The index is not an integer"
        assert (0 <= index < len(self.img_paths)), "The index is out of bounds"
        
        # get the imgs
        img_patch = Image.open(self.img_paths[index])
        mask_patch = Image.open(self.mask_paths[index])
        
        # apply the transforms to the IMGS, but should i apply the SAME transforms to the MASKS??? idk
        # we gotta investigate this thing
        return (self.transforms(img_patch), self.transforms(mask_patch))



def check_img_mask_paths(img_patch_paths, mask_patch_paths):
    # if the lists are not the same len, then its false
    if len(img_patch_paths) != len(mask_patch_paths):
        logger.critical(f"Img patch path list and mask patch path list have unequal number of elements. IMG: {len(img_patch_paths)}. MSK: {len(mask_patch_paths)}")
        raise ValueError(f"Img patch path list and mask patch path list have unequal number of elements. IMG: {len(img_patch_paths)}. MSK: {len(mask_patch_paths)}")
    else:
        logger.debug(f"Img patch path list and mask patch path list have equal number of elements. IMG: {len(img_patch_paths)}. MSK: {len(mask_patch_paths)}")
        
    for img_path, mask_path in zip(img_patch_paths,mask_patch_paths):
        
        diff_indexes = [i for i , (c1,c2) in enumerate(zip(img_path.name, mask_path.name)) if c1 != c2]
        
        # if there are more than 3 differences,  then something is wrong
        if len(diff_indexes) != 3:
            logger.critical(f"the difference between {img_path.name} and {mask_path.name} is not equal to 3 characters")
            raise ValueError(f"the difference between {img_path.name} and {mask_path.name} is not equal to 3 characters")
        
        # if the difference indexes are not contiguous, then something is wrong
        if diff_indexes[-1] - diff_indexes[0] != 2:
            logger.critical(f"the difference indexes are not 3 contiguous indexes between {img_path.name} and {mask_path.name}")
            raise ValueError(f"the difference indexes are not 3 contiguous indexes between {img_path.name} and {mask_path.name}")
        
        # if the diff indexes do not correspond to img or msk, then something is wrong
        if img_path.name[diff_indexes[0]:diff_indexes[-1]+1] != "img":
            logger.critical(f"the difference is not the string 'img' between {img_path.name} and {mask_path.name}. Difference is: {img_path.name[diff_indexes[0]:diff_indexes[-1]+1]} ")
            raise ValueError(f"the difference is not the string 'img' between {img_path.name} and {mask_path.name}. Difference is: {img_path.name[diff_indexes[0]:diff_indexes[-1]+1]} ")
        
        if mask_path.name[diff_indexes[0]:diff_indexes[-1]+1] != "msk":
            logger.critical(f"the difference is not the string 'msk' between {img_path.name} and {mask_path.name}. Difference is: {img_path.name[diff_indexes[0]:diff_indexes[-1]+1]} ")
            raise ValueError(f"the difference is not the string 'msk' between {img_path.name} and {mask_path.name}. Difference is: {img_path.name[diff_indexes[0]:diff_indexes[-1]+1]} ")

def dataset_tester(cfg):
    # get the patch paths
    img_patch_paths = list(sorted((cfg.paths.output_dir / "img_patches").glob("*.png")))
    mask_patch_paths = list(sorted((cfg.paths.output_dir / "mask_patches").glob("*.png")))
    
    # define transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomAdjustSharpness(0.5)
        ])
    
    # init the dataset
    myDataset = SatteliteImgsDataset(img_paths=img_patch_paths,mask_paths=mask_patch_paths,transforms=transforms)
    
    # test the dataset len
    print(len(myDataset))
    
    # test the getitem thing
    print(myDataset[0])
    
    # plot the getitem thing
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    img_arr = np.array(myDataset[0][0])
    mask_arr = np.array(myDataset[0][1])
    print(mask_arr.shape)
    
    # before plotting i gotta do this
    # C H W -> H W C
    new_img_arr = np.transpose(img_arr,axes=(1,2,0))
    new_mask_arr = np.transpose(mask_arr,axes=(1,2,0))
    # the img stays the same i guess???
    plt.imshow(new_img_arr)
    plt.imshow(new_mask_arr, alpha=0.6)
    plt.show()
    
    
    
    

        