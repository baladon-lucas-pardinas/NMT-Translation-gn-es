from src.config.config import load_config_variables, \
    COMMAND_NAME, \
    FLAG_SEPARATOR

class CommandConfig:
    def __init__(self, command_name, command_path, flags, flag_separator, validate_each_epochs=None, validation_metrics=None, save_checkpoints=False):
        # type: (str, str, dict, str, int, list[str], bool) -> None
        self.command_name = command_name
        self.command_path = command_path
        self.flags = flags
        self.flag_separator = flag_separator
        self.validate_each_epochs = validate_each_epochs
        self.validation_metrics = validation_metrics
        self.save_checkpoints = save_checkpoints

    def copy(self):
        # type: () -> CommandConfig
        return CommandConfig(
            command_name=self.command_name,
            command_path=self.command_path,
            flags=self.flags,
            flag_separator=self.flag_separator,
            validate_each_epochs=self.validate_each_epochs,
            save_checkpoints=self.save_checkpoints,
        )
    
    def __str__(self):
        # type: () -> str
        return "CommandConfig(command_name={}, command_path={}, flags={}, flag_separator={}, validate_each_epochs={}, save_checkpoints={})".format(
            self.command_name,
            self.command_path,
            self.flags,
            self.flag_separator,
            self.validate_each_epochs,
            self.save_checkpoints,
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_command_config(command_path, flags, validate_each_epochs=None, validation_metrics=None, save_checkpoints=False):
    # type: (str, dict, int, list[str], bool) -> CommandConfig
    config_variables = load_config_variables()
    return CommandConfig(
        command_name=config_variables[COMMAND_NAME],
        command_path=command_path,
        flags=flags,
        flag_separator=config_variables[FLAG_SEPARATOR],
        validate_each_epochs=validate_each_epochs,
        validation_metrics=validation_metrics,
        save_checkpoints=save_checkpoints,
    )