import time
# this is here to see how much time it takes for things to run
start_import = time.perf_counter()


import argparse
import yaml
from pathlib import Path
from detect_electrical_utility_from_satellite_images.utils.logging_config import setup_logger
from detect_electrical_utility_from_satellite_images.preprocess import jpg_paths_to_patches
import logging
import numpy as np
from PIL import Image
from detect_electrical_utility_from_satellite_images.config import AppConfig
import pydantic
import sys
from detect_electrical_utility_from_satellite_images.dataset import dataset_tester
import re


end_import = time.perf_counter()


def get_cfg():
    # set up parser
    parser = argparse.ArgumentParser()
    
    # add config path arg
    parser.add_argument("-cfg", "--config", required=True, help="relative path to the config file. no need to write the extension ie .yaml")
    
    # parse the args
    args = parser.parse_args()
    
    # get the relative path
    CFG_PATH = Path.cwd() / "src" / re.sub(r"-","_",Path.cwd().name) / f"{args.config}.yaml"
    
    #load the config dict
    with open(CFG_PATH,'r') as f:
        cfg_dict = yaml.safe_load(f)
        
    #get the configs
    try:
        cfg = AppConfig(**cfg_dict)
    except pydantic.ValidationError as e:
        print(f"FATAL. CONFIG FAILED: {e}")
        sys.exit(1)
        
    return cfg

def main():
    
    # timer setup
    start_time = time.perf_counter()

    # get config from CLI and config.yaml
    cfg = get_cfg()

    # logger setup
    setup_logger(cfg.paths.logging_dir_name, cfg.logging.logger_lvl, cfg.logging.console_handler_lvl, cfg.logging.file_handler_lvl)
    
    # get logger
    logger = logging.getLogger(__name__)
    
    # jpg_paths_to_patches
    # it will effectively create patches
    #jpg_paths_to_patches(cfg)
    
    # dataset tester program
    dataset_tester(cfg)
    
    # end timer
    end_time = time.perf_counter()
    
    # final log
    logger.info(f"DONE.")
    logger.info(f"Import Time: {end_import-start_import} sec")
    logger.info(f"Running Time: {end_time-start_time} sec")

if __name__ == "__main__":
    main()