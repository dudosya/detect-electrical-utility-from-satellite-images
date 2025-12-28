import torch
import torchvision
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision.transforms import v2

import logging
logger = logging.getLogger(__name__)

class SatteliteImgsDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, mask_paths, transforms):
        super().__init__()
        
        # put the stuff here
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        
        # validate the args
        assert isinstance(transforms,torchvision.transforms.v2.Compose), "transforms is not a subclass of torchvision.transforms.v2.Compose"
        self._check_img_mask_paths()
        
    def __len__(self):
        return len(self.img_paths)
    
    
    def __getitem__(self, index):
        # validate arg
        assert isinstance(index,int) and (0 <= index < len(self.img_paths)), "The index is not an integer"
        assert (0 <= index < len(self.img_paths)), "The index is out of bounds"
        
        # get the PIL objs
        img_pil_obj = Image.open(self.img_paths[index])
        msk_pil_obj = Image.open(self.mask_paths[index])
        
        # explicitly state what they are ie img or msk
        img = torchvision.tv_tensors.Image(img_pil_obj)
        # Convert mask to 2D long tensor to avoid channel dimension issues
        msk_array = np.array(msk_pil_obj)
        msk_tensor = torch.from_numpy(msk_array).long()
        msk = torchvision.tv_tensors.Mask(msk_tensor)
    
        
        # (img,msk) -> transforms
        return self.transforms((img,msk))


    def _check_img_mask_paths(self):
        img_patch_paths = self.img_paths
        mask_patch_paths = self.mask_paths
        
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
    transforms = v2.Compose([
        # Geometric: applied to BOTH
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        
        # Photometric: applied ONLY to imgs
        # TODO: experiment with this later to make sure we are not hurting the performance
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) ,
        v2.GaussianBlur(kernel_size=3),
        
        #Type conversions
        v2.ToDtype(dtype=torch.float32,scale=True),
        # the loss fn in torch work with torch.long dtype
        # we gotta convert the msk from uint8 to long
        # we want to put it to the very last step in transforms
        # because our mask must have the dtype of tv_tensors.Mask for it to be synced with the img
        # converting it to long before reaching the other transforms might break it
        # at least this is what LLM told me...i havent seen that with my eyes but ok i will just believe LLM
        v2.Lambda(lambda x: x.to(torch.long) if isinstance(x,torchvision.tv_tensors.Mask) else x ),
    ])
    
    
    # init the dataset
    myDataset = SatteliteImgsDataset(img_paths=img_patch_paths,mask_paths=mask_patch_paths,transforms=transforms)
    
    # test the dataset len
    # correct: 27
    # print(len(myDataset))
    
    # test the getitem thing
    # correct: tuple of two tensors
    output_tuple = myDataset[0]
    for element in output_tuple:
        print(type(element))
        print(element.dtype)
        
        
    # test the dtype
        
    # test: the max of the img and msk
    # IMG MAX: 1.0 correct. it has been rescaled to float
    # MSK MAX: 4 correct. it should stay as integer
    
    print(f"IMG MAX: {output_tuple[0].max()}")
    print(f"MSK MAX: {output_tuple[1].max()}")
    
    # test: dtype check
    assert output_tuple[0].dtype == torch.float32, "the img tensor is not of dtype torch.float32"
    assert output_tuple[1].dtype == torch.long, "the msk tensor is not of dtype torch.long "
    
    # test: ndim check
    assert output_tuple[0].ndim == 3, "the img tensor must have ndim of 3"
    assert output_tuple[1].ndim == 3, "the msk tensor must have ndim of 3"
    
    # print: shape 
    print(f"SHAPE IMG: {output_tuple[0].shape}")
    print(f"SHAPE MSK: {output_tuple[1].shape}")
    
    
    # plot the getitem thing
    import numpy as np
    import matplotlib.pyplot as plt
    
    sample_num = 3
    
    # DO NOT CALL the datasets twice. call it once and get the tuple out
    img_tensor, msk_tensor = myDataset[sample_num]
    
    
    # convert to the right dtypes
    img_arr = np.array(img_tensor, dtype=np.float32)
    msk_arr = np.array(msk_tensor,dtype=np.long)
    

    # the shapes of arrs are (C,H,W). we need (H,W,C)
    img_arr = np.permute_dims(img_arr,axes=(1,2,0))
    msk_arr = np.permute_dims(msk_arr, axes=(1,2,0))
    
    
    # # the img stays the same i guess???
    plt.imshow(img_arr)
    plt.imshow(msk_arr, alpha=0.5)
    plt.show()
