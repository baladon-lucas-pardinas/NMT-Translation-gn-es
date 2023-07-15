import json
import itertools

from src.utils.parsing import deep_copy_flags

# If the flag value is bool: 
#     If its value is True, the flag value should be an empty list (e.g. '--overwrite' has no value, unlike '--model model.npz')
#     If its value is False, the flag should be deleted (because False is the default value)
def handle_boolean_flags(flags):
    # type: (dict[str, list]) -> dict[str, list]
    temp_flags = deep_copy_flags(flags) # Dicts shouldn't be changed during iteration
    for flag_name, flag_value in flags.items():
        if flag_value == [True]:
            temp_flags[flag_name] = []
        elif flag_value == [False]:
            del temp_flags[flag_name]
    return temp_flags

def rename_model_file(model_name, flags):
    # type: (str, dict[str, list]) -> str
    model_name_without_extension = '.'.join(model_name.split('.')[:-1])
    model_name_extension = model_name.split('.')[-1]
    param_names = list(flags.keys())
    param_values = list(map(lambda v: str(v[0]), flags.values()))
    param_names_and_values = [param_name + '_' + param_value for param_name, param_value in zip(param_names, param_values)]
    model_name = model_name_without_extension + '_' + '_'.join(param_names_and_values) + '.' + model_name_extension
    return model_name

def get_grid_flags(default_flags, grid_file):
    # type: (dict[str, list], str) -> list[dict[str, list[str]]]
    grid_flags = []
    grid = {}

    with open(grid_file, 'r') as f:
        grid = json.load(f)

    grid_cartesian_product = itertools.product(*grid.values())
    for product in grid_cartesian_product:
        product = list(map(lambda p: [str(p)], product)) # Convert all values to lists of strings
        product_and_param_names = dict(zip(grid.keys(), product))
        default_model_name = default_flags.get('model')[0]
        current_default_flags = deep_copy_flags(default_flags)
        current_default_flags['model'] = [rename_model_file(default_model_name, product_and_param_names)]
        current_default_flags = {**current_default_flags, **product_and_param_names}
        current_default_flags = handle_boolean_flags(current_default_flags)
        grid_flags.append(current_default_flags)

    return grid_flags

def get_custom_config_flags(default_flags, grid_file):
    # type: (dict[str, list], str) -> dict[str, list[str]]
    with open(grid_file, 'r') as f:
        grid = json.load(f)

    # Convert all values to lists
    for flag_name in grid.keys():
        grid[flag_name] = [str(grid[flag_name])]

    default_flags = deep_copy_flags(default_flags)
    default_model_name = default_flags.get('model')[0]
    default_flags['model'] = [rename_model_file(default_model_name, grid)]
    custom_config_flags = {**default_flags, **grid}
    custom_config_flags = handle_boolean_flags(custom_config_flags)

    return custom_config_flags