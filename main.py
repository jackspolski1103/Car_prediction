import logging
import logging.config
from os.path import join
from datetime import datetime

from src.pipeline import run_experiment
from src.utils.files import read_configs
from src.utils.logger import set_logger_level

seed = 16 # set seed for reproducibility

if __name__ == '__main__':
    model, data = read_configs()
    #logfilename = '_'.join([cfg.config_name for cfg in [model, data]])
    #logging.config.fileConfig(join('configs', 'logging.conf'), defaults={'logfilename': join('logs',logfilename)})    
    #logger = logging.getLogger('src')
    #set_logger_level(logger,'DEBUG')
    #logger.info(f'Start experiment. Logfile name: {logfilename}')
    run_experiment(model, data)