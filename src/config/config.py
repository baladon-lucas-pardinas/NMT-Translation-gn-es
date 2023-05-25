import os
import json

MODEL_NAME = 'MODEL_NAME'
COMMAND_NAME = 'COMMAND_NAME'
BASE_DIR_SCRIPTS = 'BASE_DIR_SCRIPTS'
BASE_DIR_ARTIFACTS = 'BASE_DIR_ARTIFACTS'

CONFIG_VARIABLES = [
    MODEL_NAME,
    COMMAND_NAME,
    BASE_DIR_SCRIPTS,
    BASE_DIR_ARTIFACTS,
]

def load_config_variables():
    # type: () -> dict
    root_abs_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    config_variables_path = os.path.join(root_abs_dir, 'config.json')

    config_variables = {}
    with open(config_variables_path, 'r') as config_variables_file:
        config_variables = json.load(config_variables_file)

    variable_names = CONFIG_VARIABLES
    config_variables = {
        variable_name: config_variables[variable_name.lower()] 
        for variable_name in variable_names
    }

    return config_variables