import configparser
import os

config = configparser.ConfigParser()
config.read(f"{os.environ['PYTHONPATH']}/config.ini")
PROJ_PATH = config['data.path']['proj']
PROJ_SORTED_PATH = config['data.path']['proj_sorted']
