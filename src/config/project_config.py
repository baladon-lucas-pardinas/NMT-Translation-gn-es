from src.config.config import load_config_variables, \
    BASE_DIR_SCRIPTS, BASE_DIR_ARTIFACTS

class ProjectConfig:
    def __init__(
        self, 
        base_dir_scripts, 
        base_dir_artifacts
    ):
        # type: (str, str) -> None
        self.base_dir_scripts = base_dir_scripts
        self.base_dir_artifacts = base_dir_artifacts

    def __str__(self):
        return str(self.__dict__)
    
    def __repr__(self):
        return str(self)

def get_project_config ():
    # type: () -> ProjectConfig
    config_variables = load_config_variables()
    return ProjectConfig(
        base_dir_scripts=config_variables[BASE_DIR_SCRIPTS],
        base_dir_artifacts=config_variables[BASE_DIR_ARTIFACTS],
    )