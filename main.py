if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    from preprocess import create_patches
    
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
    OUTPUT_DIR = cfg_dict["paths"]["output_dir"]
    DATA_DIR = cfg_dict["paths"]["data_dir"]
    
    # TODO: learn how to use logging 
    # TODO: preprocess the sample photo in the data folder
    # REMINDER: the paths are all RELATIVE. dont forget to do Path.cwd() / PATH on them
    # STOPPED AT GUIDE preprocessing, patching
    