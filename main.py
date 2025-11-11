import argparse
import yaml
from pathlib import Path
from utils.logging_config import setup_logger
from preprocess import jpg_paths_to_patches
import logging
import numpy as np
from PIL import Image
from config import AppConfig
import pydantic
import sys
from dataset import dataset_tester



def get_cfg():
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
        
    return cfg

def main():

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

    logger.info("THE PROGRAM IS DONE RUNNING")

if __name__ == "__main__":
    main()