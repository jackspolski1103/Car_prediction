import argparse
import pickle
import logging
import datetime
import os, re
from os.path import join, split
from importlib.machinery import SourceFileLoader
from types import ModuleType
from tqdm import tqdm
from glob import glob

logger = logging.getLogger(__name__)

def read_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    configs = {}
    for config_type in vars(args):
        path = getattr(args, config_type)
        if not path:
            raise ValueError(f"Missing {config_type} config path")
        config_name = split(path)[1]
        #path es /configs/model/config_name.py
        path = join('configs',config_type,path)
        #agregar el .py y la barra al principio
        path = path if path.endswith('.py') else path + '.py'        
        config = import_configs_objs(path)
        config.config_name = config_name
        configs[config_type]=config
    return [configs[cfg_type] for cfg_type in ['model','data']]

def import_configs_objs(config_file):
    """Dynamicaly loads the configuration file"""
    if config_file is None:
        raise ValueError("No config path")
    loader = SourceFileLoader('config', config_file)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)
    for var in ["__name__", "__doc__", "__package__", "__loader__", "__spec__", "__builtins__"]:
        delattr(mod, var)
    return mod

def create_result_folder(data_name,model_name):
    
    # se fija si esta la carpeta data_name dentro de results
    if not os.path.exists(join('results',data_name)):
        os.makedirs(join('results',data_name))

    # se fija si esta la carpeta model_name dentro de results/data_name
    if not os.path.exists(join('results',data_name,model_name)):
        os.makedirs(join('results',data_name,model_name))

    
