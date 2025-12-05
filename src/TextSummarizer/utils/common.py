from box.exceptions import BoxValueError
import os
import yaml
from src.TextSummarizer.logging import logger

from box.config_box import ConfigBox
from pathlib import Path
from typing import Any
from beartype import beartype
#-----------------------------------------

@beartype
def read_yaml(path_to_yaml:Path)->ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content=yaml.safe_load(yaml_file)
            logger.info(f"yaml file {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@beartype
def create_directories(path_to_dirs:list , verbose=True):
    for path in path_to_dirs:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at{path}")