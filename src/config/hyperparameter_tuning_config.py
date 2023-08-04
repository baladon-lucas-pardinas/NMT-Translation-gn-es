class HyperparameterTuningConfig:
    def __init__(
        self, 
        run_id=None,
        tuning_grid_files=None,
        tuning_params_files=None,
        tuning_strategy='grid',
        from_flags=None,
        to_flags=None,
        seed=None,
        max_iters=None,
    ):
        # type: (str, list[str], list[str], str, int, int, int, int) -> None
        self.run_id = run_id
        self.tuning_grid_files = tuning_grid_files
        self.tuning_params_files = tuning_params_files
        self.tuning_strategy = tuning_strategy
        self.from_flags = from_flags
        self.to_flags = to_flags
        self.seed = seed
        self.max_iters = max_iters

    def copy(self, deep=False):
        # type: (bool) -> HyperparameterTuningConfig
        return HyperparameterTuningConfig(
            run_id=self.run_id,
            tuning_grid_files=self.tuning_grid_files.copy() if (deep and self.tuning_grid_files) else self.tuning_grid_files,
            tuning_params_files=self.tuning_params_files.copy() if (deep and self.tuning_params_files) else self.tuning_params_files,
            tuning_strategy=self.tuning_strategy,
            from_flags=self.from_flags,
            to_flags=self.to_flags,
            seed=self.seed,
            max_iters=self.max_iters,
        )
    
    def __str__(self):
        # type: () -> str
        return "HyperparameterTuningConfig(run_id={}, tuning_grid_files={}, tuning_params_files={}, tuning_strategy={}, from_flags={}, to_flags={}, seed={}, max_iters={})".format(
            self.run_id,
            self.tuning_grid_files,
            self.tuning_params_files,
            self.tuning_strategy,
            self.from_flags,
            self.to_flags,
            self.seed,
            self.max_iters,
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_hyperparameter_tuning_config(
    run_id=None,
    tuning_grid_files=None,
    tuning_params_files=None,
    tuning_strategy='gridsearch',
    from_flags=None,
    to_flags=None,
    seed=None,
    max_iters=None,
):
    # type: (str, list[str], list[str], str, int, int, int, int) -> HyperparameterTuningConfig
    return HyperparameterTuningConfig(
        run_id=run_id,
        tuning_grid_files=tuning_grid_files,
        tuning_params_files=tuning_params_files,
        tuning_strategy=tuning_strategy,
        from_flags=from_flags,
        to_flags=to_flags,   
        seed=seed,
        max_iters=max_iters,
    )