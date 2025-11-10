import argparse
import yaml
from pathlib import Path
from preprocess import create_patches
from utils.logging_config import setup_logger
from utils.file_utils import drop_jpg_paths_with_no_npz_pair
import logging
import numpy as np
from PIL import Image

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
    PATCH_SIZE = cfg_dict["preprocessing"]["patch_size"]
    BACKGROUND_FRACTION = cfg_dict["preprocessing"]["background_fraction"]
    
    OUTPUT_DIR = Path.cwd() / cfg_dict["paths"]["output_dir"]
    DATA_DIR = Path.cwd() / cfg_dict["paths"]["data_dir"]
    
    LOG_DIR = Path.cwd() / "logs" / cfg_dict["logging"]["logging_dir_name"]
    LOGGER_LVL = cfg_dict["logging"]["logger_lvl"]
    CONSOLE_HANDLER_LVL = cfg_dict["logging"]["console_handler_lvl"]
    FILE_HANDER_LVL = cfg_dict["logging"]["file_handler_lvl"]
    
    # logger setup
    setup_logger(LOG_DIR, LOGGER_LVL, CONSOLE_HANDLER_LVL, FILE_HANDER_LVL)
    
    # get logger
    logger = logging.getLogger(__name__)
    
    # now we can JUST use the logger like this 
        # logger.debug("lvl 1 whaat")
        # logger.warning("there is a warning dude")
    
    # get all the jpg paths
    jpg_paths = sorted(DATA_DIR.glob("*/*.jpg"))
    
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
        create_patches(img_arr,mask_arr,PATCH_SIZE,OUTPUT_DIR,filename,BACKGROUND_FRACTION)


if __name__ == "__main__":
    
    main()
    

    