import argparse
import yaml
from pathlib import Path
from preprocess import create_patches
from utils.logging_config import setup_logger
from utils.file_utils import drop_jpg_paths_with_no_npz_pair
import logging
import numpy as np
from PIL import Image
from config import AppConfig
import pydantic
import sys

def main():
    
    # set up parser
    parser = argparse.ArgumentParser()
    
    # add config path arg
    parser.add_argument("-cfg", "--config", required=True, help="relative path to the config file. no need to write the extension ie .yaml")
    
    # parse the args
    args = parser.parse_args()
    
    # get the relative path
    CFG_PATH = Path.cwd() / f"{args.config}.yaml"
    
    #load the config dict
    with open(CFG_PATH,'r') as f:
        cfg_dict = yaml.safe_load(f)
        
    #get the configs
    try:
        cfg = AppConfig(**cfg_dict)
    except pydantic.ValidationError as e:
        print(f"FATAL. CONFIG FAILED: {e}")
        sys.exit(1)

    # logger setup
    setup_logger(cfg.paths.logging_dir_name, cfg.logging.logger_lvl, cfg.logging.console_handler_lvl, cfg.logging.file_handler_lvl)
    
    # get logger
    logger = logging.getLogger(__name__)
    
    # now we can JUST use the logger like this 
        # logger.debug("lvl 1 whaat")
        # logger.warning("there is a warning dude")
    
    # get all the jpg paths
    jpg_paths = sorted(cfg.paths.data_dir.glob("*/*.jpg"))
    
    # drop paths that dont have npz pair
    jpg_paths = drop_jpg_paths_with_no_npz_pair(jpg_paths)
    
    for jpg_path in jpg_paths:
        # get filename
        filename = jpg_path.name
        
        # load img arr
        PIL_obj = Image.open(jpg_path)
        img_arr = np.array(PIL_obj)
        
        # assuming that mask files are all .npz files
        npz_path = jpg_path.with_suffix(".npz")
        
        # load the mask arr
        npz_obj = np.load(npz_path)
        mask_arr = npz_obj[npz_obj.files[0]]
        
        logger.info(f"{filename} is going to be sliced to patches")
        
        # call create_patch func for a single pair
        create_patches(img_arr,mask_arr,cfg.preprocessing.patch_size,cfg.paths.output_dir,filename,cfg.preprocessing.background_fraction)


if __name__ == "__main__":
    
    main()
    

    