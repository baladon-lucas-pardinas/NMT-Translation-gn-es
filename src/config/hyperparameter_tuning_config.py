from src.config.config import load_config_variables, \
    COMMAND_NAME, \
    FLAG_SEPARATOR, \
    BASE_DIR_EVALUATION

class HyperparameterTuningConfig:
    def __init__(
        self, 
        tuning_grid_files=None,
        tuning_params_files=None,
        search_method='grid',
        seed=None,
    ):
        # type: (list[str], list[str], str, int) -> None
        self.tuning_grid_files = tuning_grid_files
        self.tuning_params_files = tuning_params_files
        self.search_method = search_method
        self.seed = seed

    def copy(self, deep=False):
        # type: (bool) -> HyperparameterTuningConfig
        return HyperparameterTuningConfig(
            tuning_grid_files=self.tuning_grid_files.copy() if (deep and self.tuning_grid_files) else self.tuning_grid_files,
            tuning_params_files=self.tuning_params_files.copy() if (deep and self.tuning_params_files) else self.tuning_params_files,
            search_method=self.search_method,
            seed=self.seed,
        )
    
    def __str__(self):
        # type: () -> str
        return "HyperparameterTuningConfig(tuning_grid_files={}, tuning_params_files={}, search_method={}, seed={})".format(
            self.tuning_grid_files,
            self.tuning_params_files,
            self.search_method,
            self.seed,
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_hyperparameter_tuning_config(
    tuning_grid_files=None,
    tuning_params_files=None,
    search_method='grid',
    seed=None,
):
    # type: (list[str], list[str], str, int) -> HyperparameterTuningConfig
    config_variables = load_config_variables()
    return HyperparameterTuningConfig(
        tuning_grid_files=tuning_grid_files,
        tuning_params_files=tuning_params_files,
        search_method=search_method,
        seed=seed,
    )