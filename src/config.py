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
    def __init__(
        self, 
        base_dir_output,
        base_dir_corpus, 
        base_dir_scripts, 
        base_dir_vocab, 
        base_dir_artifacts
    ):
        # type: (str, str, str, str, str) -> None
        self.base_dir_output = base_dir_output
        self.base_dir_corpus = base_dir_corpus
        self.base_dir_scripts = base_dir_scripts
        self.base_dir_vocab = base_dir_vocab
        self.base_dir_artifacts = base_dir_artifacts

    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return str(self)

def get_project_config ():
    # type: () -> ProjectConfig
    env_variables = load_env()
    return ProjectConfig(
        base_dir_output=env_variables[BASE_DIR_OUTPUT],
        base_dir_corpus=env_variables[BASE_DIR_CORPUS],
        base_dir_scripts=env_variables[BASE_DIR_SCRIPTS],
        base_dir_vocab=env_variables[BASE_DIR_VOCAB],
        base_dir_artifacts=env_variables[BASE_DIR_ARTIFACTS],
    )

def get_command_config(command_path, flags):
    # type: (str, dict) -> CommandConfig
    env_variables = load_env()
    return CommandConfig(
        command_name=env_variables[COMMAND_NAME],
        command_path=command_path,
        flags=flags,
    )