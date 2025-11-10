if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    from preprocess import create_patches
    from utils.logging_config import setup_logger
    import logging
    import numpy as np
    from PIL import Image
    
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
    
    LOG_DIR = Path.cwd() / "utils" / "logs" / cfg_dict["logging"]["logging_dir_name"]
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
    
    
    # TODO: move this to utils folder later
    def drop_jpg_paths_with_no_npz_pair(jpg_paths: list[Path])-> list[Path]:
        new_list = []
        for jpg_path in jpg_paths:
            if jpg_path.with_suffix(".npz").exists():
                logger.debug(f"npz file for {jpg_path.name} exists")
                new_list.append(jpg_path)
            else:
                logger.warning(f"npz file for {jpg_path} does not exist. it is ignored")
        return new_list
    
    jpg_paths = drop_jpg_paths_with_no_npz_pair(jpg_paths)
    
    # convert the imgs and npzs to np arrays
    
    # TODO: move this thing to utils later maybe
    def get_img_npz_arr_list(jpg_paths: list[Path]) -> list:
        img_npz_arr_list = []
        for jpg_path in jpg_paths:
            filename = jpg_path.name
            PIL_obj = Image.open(jpg_path)
            img_arr = np.array(PIL_obj)
            npz_path = jpg_path.with_suffix(".npz")
            npz_file = np.load(npz_path)
            npz_arr = npz_file[npz_file.files[0]]
            img_npz_arr_list.append((img_arr,npz_arr,filename))
        return img_npz_arr_list
    
    img_npz_arr_list = get_img_npz_arr_list(jpg_paths)

    # call the create patches func
    for toople in img_npz_arr_list:
        img_arr, npz_arr, filename = toople
        logger.info(f"{filename} is going to be sliced to patches")
        create_patches(img_arr,npz_arr,PATCH_SIZE,OUTPUT_DIR,filename,BACKGROUND_FRACTION)
    
    