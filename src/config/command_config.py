from src.config.config import load_config_variables, \
    COMMAND_NAME, \
    FLAG_SEPARATOR, \
    BASE_DIR_EVALUATION

class CommandConfig:
    def __init__(self, command_name, command_path, flags, flag_separator, validate_each_epochs=None, validation_metrics=None, save_checkpoints=False, results_dir=None, base_dir_evaluation=None):
        # type: (str, str, dict, str, int, list[str], bool, str, str) -> None
        self.command_name = command_name
        self.command_path = command_path
        self.flags = flags
        self.flag_separator = flag_separator
        self.validate_each_epochs = validate_each_epochs
        self.validation_metrics = validation_metrics
        self.save_checkpoints = save_checkpoints
        self.results_dir = results_dir
        self.base_dir_evaluation = base_dir_evaluation

    def copy(self, deep=False):
        # type: (bool) -> CommandConfig
        return CommandConfig(
            command_name=self.command_name,
            command_path=self.command_path,
            flags=self.flags.copy() if deep else self.flags,
            flag_separator=self.flag_separator,
            validate_each_epochs=self.validate_each_epochs,
            validation_metrics=self.validation_metrics.copy() if deep else self.validation_metrics,
            save_checkpoints=self.save_checkpoints,
            results_dir=self.results_dir,
            base_dir_evaluation=self.base_dir_evaluation,
        )
    
    def __str__(self):
        # type: () -> str
        return "CommandConfig(command_name={}, command_path={}, flags={}, flag_separator={}, validate_each_epochs={}, validation_metrics={}, save_checkpoints={}, results_dir={}, base_dir_evaluation={})".format(
            self.command_name,
            self.command_path,
            self.flags,
            self.flag_separator,
            self.validate_each_epochs,
            self.validation_metrics,
            self.save_checkpoints,
            self.results_dir,
            self.base_dir_evaluation,
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_command_config(command_path, flags, validate_each_epochs=None, validation_metrics=None, save_checkpoints=False, results_dir=None):
    # type: (str, dict, int, list[str], bool, str) -> CommandConfig
    config_variables = load_config_variables()
    return CommandConfig(
        command_name=config_variables[COMMAND_NAME],
        command_path=command_path,
        flags=flags,
        flag_separator=config_variables[FLAG_SEPARATOR],
        validate_each_epochs=validate_each_epochs,
        validation_metrics=validation_metrics,
        save_checkpoints=save_checkpoints,
        results_dir=results_dir,
        base_dir_evaluation=config_variables[BASE_DIR_EVALUATION],
    )