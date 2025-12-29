import logging
import typing
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import v2

logger = logging.getLogger(__name__)


class SatteliteImgsDataset(torch.utils.data.Dataset):
    """Dataset for satellite images and corresponding segmentation masks.

    This dataset loads pairs of satellite images and their corresponding
    segmentation masks, applying the same geometric transformations to both
    while applying photometric transformations only to images.

    Args:
        img_paths: List of paths to image files
        mask_paths: List of paths to mask files (must correspond to img_paths)
        transforms: torchvision transforms to apply to (image, mask) pairs
    """

    def __init__(
        self,
        img_paths: list[Path],
        mask_paths: list[Path],
        transforms: torchvision.transforms.v2.Compose,
    ) -> None:
        super().__init__()

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

        # Validate the arguments
        if not isinstance(transforms, torchvision.transforms.v2.Compose):
            raise TypeError(
                f"transforms must be a torchvision.transforms.v2.Compose, "
                f"got {type(transforms)}",
            )

        self._check_img_mask_paths()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Tuple of (image_tensor, mask_tensor) where:
            - image_tensor: Float tensor of shape (C, H, W) in range [0, 1]
            - mask_tensor: Long tensor of shape (1, H, W) with class indices
        """
        # Validate index
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer, got {type(index)}")
        if not (0 <= index < len(self.img_paths)):
            raise IndexError(
                f"index {index} out of bounds for dataset of size {len(self.img_paths)}",
            )

        # Load PIL images
        img_pil = Image.open(self.img_paths[index])
        mask_pil = Image.open(self.mask_paths[index])

        # Convert to torchvision tensors
        img = torchvision.tv_tensors.Image(img_pil)

        # Convert mask to 2D long tensor to avoid channel dimension issues
        mask_array = np.array(mask_pil)
        mask_tensor = torch.from_numpy(mask_array).long()
        mask = torchvision.tv_tensors.Mask(mask_tensor)

        # Apply transforms to (image, mask) pair
        return self.transforms((img, mask))

    def _check_img_mask_paths(self):
        img_patch_paths = self.img_paths
        mask_patch_paths = self.mask_paths

        # if the lists are not the same len, then its false
        if len(img_patch_paths) != len(mask_patch_paths):
            logger.critical(
                f"Img patch path list and mask patch path list have unequal number of elements. IMG: {len(img_patch_paths)}. MSK: {len(mask_patch_paths)}",
            )
            raise ValueError(
                f"Img patch path list and mask patch path list have unequal number of elements. IMG: {len(img_patch_paths)}. MSK: {len(mask_patch_paths)}",
            )
        logger.debug(
            f"Img patch path list and mask patch path list have equal number of elements. IMG: {len(img_patch_paths)}. MSK: {len(mask_patch_paths)}",
        )

        for img_path, mask_path in zip(img_patch_paths, mask_patch_paths):
            diff_indexes = [
                i
                for i, (c1, c2) in enumerate(zip(img_path.name, mask_path.name))
                if c1 != c2
            ]

            # if there are more than 3 differences,  then something is wrong
            if len(diff_indexes) != 3:
                logger.critical(
                    f"the difference between {img_path.name} and {mask_path.name} is not equal to 3 characters",
                )
                raise ValueError(
                    f"the difference between {img_path.name} and {mask_path.name} is not equal to 3 characters",
                )

            # if the difference indexes are not contiguous, then something is wrong
            if diff_indexes[-1] - diff_indexes[0] != 2:
                logger.critical(
                    f"the difference indexes are not 3 contiguous indexes between {img_path.name} and {mask_path.name}",
                )
                raise ValueError(
                    f"the difference indexes are not 3 contiguous indexes between {img_path.name} and {mask_path.name}",
                )

            # if the diff indexes do not correspond to img or msk, then something is wrong
            if img_path.name[diff_indexes[0] : diff_indexes[-1] + 1] != "img":
                logger.critical(
                    f"the difference is not the string 'img' between {img_path.name} and {mask_path.name}. Difference is: {img_path.name[diff_indexes[0] : diff_indexes[-1] + 1]} ",
                )
                raise ValueError(
                    f"the difference is not the string 'img' between {img_path.name} and {mask_path.name}. Difference is: {img_path.name[diff_indexes[0] : diff_indexes[-1] + 1]} ",
                )

            if mask_path.name[diff_indexes[0] : diff_indexes[-1] + 1] != "msk":
                logger.critical(
                    f"the difference is not the string 'msk' between {img_path.name} and {mask_path.name}. Difference is: {img_path.name[diff_indexes[0] : diff_indexes[-1] + 1]} ",
                )
                raise ValueError(
                    f"the difference is not the string 'msk' between {img_path.name} and {mask_path.name}. Difference is: {img_path.name[diff_indexes[0] : diff_indexes[-1] + 1]} ",
                )


def dataset_tester(cfg):
    # get the patch paths
    img_patch_paths = sorted((cfg.paths.output_dir / "img_patches").glob("*.png"))
    mask_patch_paths = sorted((cfg.paths.output_dir / "mask_patches").glob("*.png"))

    # define transforms
    transforms = v2.Compose(
        [
            # Geometric: applied to BOTH
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            # Photometric: applied ONLY to imgs
            # TODO: experiment with this later to make sure we are not hurting the performance
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.GaussianBlur(kernel_size=3),
            # Type conversions
            v2.ToDtype(dtype=torch.float32, scale=True),
            # the loss fn in torch work with torch.long dtype
            # we gotta convert the msk from uint8 to long
            # we want to put it to the very last step in transforms
            # because our mask must have the dtype of tv_tensors.Mask for it to be synced with the img
            # converting it to long before reaching the other transforms might break it
            # at least this is what LLM told me...i havent seen that with my eyes but ok i will just believe LLM
            v2.Lambda(
                lambda x: x.to(torch.long)
                if isinstance(x, torchvision.tv_tensors.Mask)
                else x,
            ),
        ],
    )

    # init the dataset
    myDataset = SatteliteImgsDataset(
        img_paths=img_patch_paths, mask_paths=mask_patch_paths, transforms=transforms,
    )

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
    assert output_tuple[0].dtype == torch.float32, (
        "the img tensor is not of dtype torch.float32"
    )
    assert output_tuple[1].dtype == torch.long, (
        "the msk tensor is not of dtype torch.long "
    )

    # test: ndim check
    assert output_tuple[0].ndim == 3, "the img tensor must have ndim of 3"
    # Masks can be 2D (H, W) or 3D (1, H, W) - both are valid
    assert output_tuple[1].ndim in [2, 3], f"the msk tensor must have ndim of 2 or 3, got {output_tuple[1].ndim}"

    # print: shape
    print(f"SHAPE IMG: {output_tuple[0].shape}")
    print(f"SHAPE MSK: {output_tuple[1].shape}")

    # plot the getitem thing
    import matplotlib.pyplot as plt
    import numpy as np

    sample_num = 3

    # DO NOT CALL the datasets twice. call it once and get the tuple out
    img_tensor, msk_tensor = myDataset[sample_num]

    # convert to the right dtypes
    img_arr = np.array(img_tensor, dtype=np.float32)
    msk_arr = np.array(msk_tensor, dtype=np.long)

    # Handle different mask dimensions
    # Image is always 3D (C, H, W) -> convert to (H, W, C)
    if img_arr.ndim == 3:
        img_arr = np.transpose(img_arr, (1, 2, 0))
    
    # Mask can be 2D (H, W) or 3D (1, H, W)
    if msk_arr.ndim == 3:
        # If 3D, squeeze channel dimension if it's 1
        if msk_arr.shape[0] == 1:
            msk_arr = msk_arr.squeeze(0)
        else:
            # If multiple channels, take first channel for display
            msk_arr = msk_arr[0]
    # If 2D, keep as is
    
    # # the img stays the same i guess???
    plt.imshow(img_arr)
    plt.imshow(msk_arr, alpha=0.5)
    plt.show()
