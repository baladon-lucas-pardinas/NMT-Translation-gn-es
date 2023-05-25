from src.config.config import load_config_variables, \
    COMMAND_NAME

FLAG_SEPARATOR = '--'

class CommandConfig:
    def __init__(self, command_name, command_path, flags, flag_separator=FLAG_SEPARATOR):
        # type: (str, str, dict, str) -> None
        self.command_name = command_name
        self.command_path = command_path
        self.flags = flags
        self.flag_separator = flag_separator

    def copy(self):
        # type: () -> CommandConfig
        return CommandConfig(
            command_name=self.command_name,
            command_path=self.command_path,
            flags=self.flags,
        )
    
    def __str__(self):
        # type: () -> str
        return "CommandConfig(command_name={}, command_path={}, flags={})".format(
            self.command_name,
            self.command_path,
            self.flags
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_command_config(command_path, flags):
    # type: (str, dict) -> CommandConfig
    config_variables = load_config_variables()
    return CommandConfig(
        command_name=config_variables[COMMAND_NAME],
        command_path=command_path,
        flags=flags,
    )