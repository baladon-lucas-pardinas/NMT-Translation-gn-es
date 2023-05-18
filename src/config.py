import os
import dotenv
from src.utils.command_handler import CommandConfig

COMMAND_NAME = 'COMMAND_NAME'
BASE_DIR_OUTPUT = 'BASE_DIR_OUTPUT'
BASE_DIR_CORPUS = 'BASE_DIR_CORPUS'
BASE_DIR_SCRIPTS = 'BASE_DIR_SCRIPTS'
BASE_DIR_VOCAB = 'BASE_DIR_VOCAB'
BASE_DIR_ARTIFACTS = 'BASE_DIR_ARTIFACTS'

ENVIRONMENT_VARIABLES = [
    COMMAND_NAME,
    BASE_DIR_OUTPUT,
    BASE_DIR_CORPUS,
    BASE_DIR_SCRIPTS,
    BASE_DIR_VOCAB,
    BASE_DIR_ARTIFACTS,
]

def load_env(absolute_path=False):
    # type: (bool) -> dict
    # (Assumes env variables are paths)
    root_abs_dir = os.path.join(os.path.dirname(__file__), '..')
    env_abs_dir = os.path.join(root_abs_dir, '.env')
    dotenv.load_dotenv(env_abs_dir)

    # Loads env variables
    variable_names = ENVIRONMENT_VARIABLES
    environment_variables = {
        variable_name.lower(): os.getenv(variable_name)
        for variable_name in variable_names
    }

    # Adds ../ to paths
    environment_variables = {
        variable_name: os.path.join(root_abs_dir, variable_value) if absolute_path else variable_value
        for variable_name, variable_value in environment_variables.items()
        if variable_value is not None
    }

    return environment_variables

class ProjectConfig:
    def __init__(self, **kwargs):
        self.base_dir_output = kwargs[BASE_DIR_OUTPUT]
        self.base_dir_corpus = kwargs[BASE_DIR_CORPUS]
        self.base_dir_scripts = kwargs[BASE_DIR_SCRIPTS]
        self.base_dir_vocab = kwargs[BASE_DIR_VOCAB]
        self.base_dir_artifacts = kwargs[BASE_DIR_ARTIFACTS]

    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return str(self)

def get_project_config ():
    # type: () -> ProjectConfig
    return ProjectConfig(**load_env())

def get_command_config(**kwargs):
    # type: (dict) -> CommandConfig
    return CommandConfig(**load_env(), **kwargs)