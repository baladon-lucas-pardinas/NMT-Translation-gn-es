import json
import itertools
import random

import scipy.stats as stats

from src.utils.parsing import deep_copy_flags, \
    handle_boolean_flags, \
    handle_sentencepiece_flags, \
    rename_model_file

def get_hyperparameters_flags(default_flags, hyperparameters_file, search_method, max_iters=None, seed=None):
    # type: (dict[str, list], str, str, int, int) -> list[dict[str, list[str]]]
    methods = {'gridsearch': get_grid_flags, 'randomsearch': get_random_flags}
    extra_parameters = {'seed': seed, 'max_iters': max_iters} if search_method == 'randomsearch' else {}
    return methods[search_method](default_flags, hyperparameters_file, **extra_parameters)

def get_random_flags(default_flags, hyperparameters_file, max_iters, seed=None):
    # type: (dict[str, list], str, int, int) -> list[dict[str, list[str]]]
    random_flags = []
    random_instance = random.Random(); random_instance.seed(seed) # random.seed is not enough to ensure reproducibility
    distribution_functions = {
        "int_truncnorm": lambda *args, **kwargs: stats.truncnorm.rvs(*args, **kwargs).round().astype(int), 
        "loguniform": stats.loguniform.rvs, 
        "multinoulli": random_instance.choices,
        "int_uniform": stats.randint.rvs,
        "uniform": stats.uniform.rvs,
    }

    with open(hyperparameters_file, 'r') as f:
        hyperparameters = json.load(f)

    hyperparameters_values = {}
    for hyperparameter_name, hyperparameter_info in hyperparameters.items():
        hyperparameter_distribution = hyperparameter_info.get('distribution')
        hyperparameter_distribution_args = hyperparameter_info.get('args')
        hyperparameter_distribution_kwargs = hyperparameter_info.get('kwargs', {})
        shares_value_with = hyperparameter_info.get('shares_value_with', None)

        distribution_function = distribution_functions[hyperparameter_distribution]
        distribution_extra_params = {'size': max_iters, 'random_state': seed} if hyperparameter_distribution != 'multinoulli' else {'k': max_iters}
        distribution_extra_params = {**distribution_extra_params, **hyperparameter_distribution_kwargs}
        random_values = distribution_function(*hyperparameter_distribution_args, **distribution_extra_params)
        random_values = list(map(lambda p: [str(p)], random_values)) # Convert all values to lists of strings

        if shares_value_with is not None:
            hyperparameters_values[shares_value_with] = random_values
        hyperparameters_values[hyperparameter_name] = random_values

    for i in range(max_iters):
        current_flags = {}
        for hyperparameter_name, hyperparameter_values in hyperparameters_values.items():
            current_flags[hyperparameter_name] = hyperparameter_values[i]
        current_flags = deep_copy_flags(current_flags)
        default_model_name = default_flags.get('model')[0]
        current_flags['model'] = [rename_model_file(default_model_name, current_flags)]
        current_flags = {**default_flags, **current_flags}
        current_flags = handle_boolean_flags(current_flags)
        current_flags = handle_sentencepiece_flags(current_flags)

        random_flags.append(current_flags)

    return random_flags

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
        current_default_flags = handle_sentencepiece_flags(current_default_flags)
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
    custom_config_flags = handle_sentencepiece_flags(custom_config_flags)

    return custom_config_flags