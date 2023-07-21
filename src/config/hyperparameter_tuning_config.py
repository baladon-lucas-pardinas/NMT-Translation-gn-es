class HyperparameterTuningConfig:
    def __init__(
        self, 
        run_id=None,
        tuning_grid_files=None,
        tuning_params_files=None,
        search_method='grid',
        from_flags=None,
        to_flags=None,
    ):
        # type: (str, list[str], list[str], str, int, int) -> None
        self.run_id = run_id
        self.tuning_grid_files = tuning_grid_files
        self.tuning_params_files = tuning_params_files
        self.search_method = search_method
        self.from_flags = from_flags
        self.to_flags = to_flags

    def copy(self, deep=False):
        # type: (bool) -> HyperparameterTuningConfig
        return HyperparameterTuningConfig(
            run_id=self.run_id,
            tuning_grid_files=self.tuning_grid_files.copy() if (deep and self.tuning_grid_files) else self.tuning_grid_files,
            tuning_params_files=self.tuning_params_files.copy() if (deep and self.tuning_params_files) else self.tuning_params_files,
            search_method=self.search_method,
            from_flags=self.from_flags,
            to_flags=self.to_flags,
        )
    
    def __str__(self):
        # type: () -> str
        return "HyperparameterTuningConfig(run_id={}, tuning_grid_files={}, tuning_params_files={}, search_method={}, from_flags={}, to_flags={})".format(
            self.run_id,
            self.tuning_grid_files,
            self.tuning_params_files,
            self.search_method,
            self.from_flags,
            self.to_flags,
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_hyperparameter_tuning_config(
    run_id=None,
    tuning_grid_files=None,
    tuning_params_files=None,
    search_method='grid',
    from_flags=None,
    to_flags=None,
):
    # type: (str, list[str], list[str], str, int, int) -> HyperparameterTuningConfig
    return HyperparameterTuningConfig(
        run_id=run_id,
        tuning_grid_files=tuning_grid_files,
        tuning_params_files=tuning_params_files,
        search_method=search_method,
        from_flags=from_flags,
        to_flags=to_flags,   
    )