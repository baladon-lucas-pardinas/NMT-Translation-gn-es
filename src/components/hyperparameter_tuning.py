import json
import itertools

from src.utils.command_handler import deep_copy_flags
# TODO: Change model names on each config...

def rename_model_file(model_name, flags):
    # type: (str, dict[str, list]) -> str
    model_name_without_extension = ''.join(model_name.split('.')[:-1])
    model_name_extension = model_name.split('.')[-1]
    param_names_and_values = [str(flags.get(key)) for key in flags.keys()]
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
        product_and_param_names = dict(zip(product, grid.keys()))
        current_default_flags = deep_copy_flags(default_flags)
        default_model_name = default_flags.get('model')[0]
        current_default_flags['model'] = rename_model_file(default_model_name, product_and_param_names)
        grid_flags.append({**current_default_flags, **product_and_param_names})

    return grid_flags

def get_custom_config_flags(default_flags, grid_file):
    # type: (dict[str, list], str) -> dict[str, list[str]]
    with open(grid_file, 'r') as f:
        grid = json.load(f)
    default_flags = deep_copy_flags(default_flags)
    default_model_name = default_flags.get('model')[0]
    default_flags['model'] = rename_model_file(default_model_name, grid)
    return {**default_flags, **grid}