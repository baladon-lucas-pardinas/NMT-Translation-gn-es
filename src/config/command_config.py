from src.config.config import load_config_variables, \
    COMMAND_NAME, \
    FLAG_SEPARATOR

class CommandConfig:
    def __init__(self, command_name, command_path, flags, flag_separator, save_each_epochs=None):
        # type: (str, str, dict, str, int) -> None
        self.command_name = command_name
        self.command_path = command_path
        self.flags = flags
        self.flag_separator = flag_separator
        self.save_each_epochs = save_each_epochs

    def copy(self):
        # type: () -> CommandConfig
        return CommandConfig(
            command_name=self.command_name,
            command_path=self.command_path,
            flags=self.flags,
            flag_separator=self.flag_separator,
            save_each_epochs=self.save_each_epochs,
        )
    
    def __str__(self):
        # type: () -> str
        return "CommandConfig(command_name={}, command_path={}, flags={}, flag_separator={}, save_each_epochs={})".format(
            self.command_name,
            self.command_path,
            self.flags,
            self.flag_separator,
            self.save_each_epochs,
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_command_config(command_path, flags, save_each_epochs=None):
    # type: (str, dict, int) -> CommandConfig
    config_variables = load_config_variables()
    return CommandConfig(
        command_name=config_variables[COMMAND_NAME],
        command_path=command_path,
        flags=flags,
        flag_separator=config_variables[FLAG_SEPARATOR],
        save_each_epochs=save_each_epochs,
    )