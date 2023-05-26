import os

from src.config.command_config import CommandConfig, \
    FLAG_SEPARATOR

# Flags are a string of the form --flag1 value1 ... valueN --flag2 value1 ... valueM
# The output dict should be like {flag1: [value1, ..., valueN], flag2: [value1, ..., valueM]}
def parse_flags(flags, flag_separator=FLAG_SEPARATOR):
    # type: (str, str) -> dict
    flag_dict = {}
    flags = flags.split(flag_separator)
    print(flags)
    for flag in flags:
        flag = flag.strip()
        if flag == "":
            continue
        flag = flag.split(' ')
        flag_name = flag[0]
        flag_values = flag[1:]
        flag_dict[flag_name] = flag_values
    return flag_dict

# Flags are a list of tuples (flag, values: list)
# The output string should be like --flag1 value1 ... valueN --flag2 value1 ... valueM
def create_command_flags(flags):
    # type: (dict) -> str
    output = ""
    flags = list(flags.items())
    for flag, values in flags:
        output += " --" + flag
        for value in values:
            output += " " + value
        output += " "
    return output

def create_command(config):
    # type: (CommandConfig) -> str
    command = ""
    command_dir = os.path.join(config.command_path, config.command_name)
    
    command += command_dir.format(
        command_path=config.command_path, 
        command_name=config.command_name
    )
    command += create_command_flags(config.flags)

    return command