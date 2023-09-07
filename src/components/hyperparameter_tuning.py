import json
import itertools
import random
import os

import scipy.stats as stats

from src.config.finetuning_config import FinetuningConfig
from src.utils.parsing import deep_copy_flags
from src.utils import file_manager

PRETRAINING_EPOCHS = 'finetuning-epochs'

def get_cached_model_dir(cache_dir_template, pretraining_epochs):
    # type: (str, str) -> (str, int)
    pretraining_epochs = int(pretraining_epochs)

    for i in reversed(range(pretraining_epochs)): # The more trained the model the better
        cached_model_path_i = cache_dir_template.format(str(i))

        if os.path.isdir(cached_model_path_i):
            # file_manager.save_copy(cached_model_path_i, new_model_dir)
            # for new_dir_file in os.listdir(new_model_dir):
            #     if new_dir_file.endswith('.npz') \
            #        and not new_dir_file.endswith('optimizer.npz'):
            #         new_cached_model_path = \
            #             os.path.join(new_model_dir, new_dir_file)
            #         renamed_new_cached_model_path = \
            #             os.path.join(new_model_dir, new_model_filename)
            #         os.rename(new_cached_model_path, 
            #                   renamed_new_cached_model_path)
            return cached_model_path_i, i

    return None, None

def handle_finetuning_flags(finetuning_config, flags):
    # type: (FinetuningConfig, dict) -> (FinetuningConfig, dict)
    pretraining_epochs = flags.pop(PRETRAINING_EPOCHS, [None])[0]
    if pretraining_epochs is not None:
        finetuning_config.epochs = pretraining_epochs
    return finetuning_config, flags

# If the flag value is bool: 
#     If its value is True, the flag value should be an empty list (e.g. '--overwrite' has no value, unlike '--model model.npz')
#     If its value is False, the flag should be deleted (because False is the default value)
def handle_boolean_flags(flags):
    # type: (dict[str, list]) -> dict[str, list]
    temp_flags = deep_copy_flags(flags) # Dicts shouldn't be changed during iteration
    for flag_name, flag_value in flags.items():
        flag_value_id = str(flag_value)
        if flag_value_id == "[True]":
            temp_flags[flag_name] = []
        elif flag_value_id == "[False]":
            del temp_flags[flag_name]
    return temp_flags

# If the model uses sentencepiece, each vocabulary configuration must be in a different file.
def handle_sentencepiece_flags(flags):
    # type: (dict[str, list]) -> dict[str, list]
    src_vocab, trg_vocab = flags.get('vocabs', ['', ''])
    if src_vocab.endswith('.spm') or trg_vocab.endswith('.spm'):
        assert src_vocab.endswith('.spm') and trg_vocab.endswith('.spm'), 'Both vocabularies must use sentencepiece if one of them does.'
        dim_vocab = flags.get('dim-vocabs', [None])[0]
        dim_vocab = dim_vocab.replace(' ', '_') # In case dim-vocabs is passed as a string instead of a list of ints
        src_new_name = src_vocab.replace('.spm', 'V{dim_vocab}.spm'.format(dim_vocab=dim_vocab))
        trg_new_name = trg_vocab.replace('.spm', 'V{dim_vocab}.spm'.format(dim_vocab=dim_vocab))
        flags['vocabs'] = [src_new_name, trg_new_name]
    return flags

def rename_model_file(model_name, flags):
    # type: (str, dict[str, list]) -> str
    model_name_without_extension = '.'.join(model_name.split('.')[:-1])
    model_name_extension = model_name.split('.')[-1]
    param_names = list(flags.keys())
    param_values = list(map(lambda v: str(v[0]), flags.values()))
    param_names_and_values = [param_name + '_' + param_value for param_name, param_value in zip(param_names, param_values)]
    model_name = model_name_without_extension + '_' + '_'.join(param_names_and_values) + '.' + model_name_extension
    model_name = model_name.replace(' ', '')
    return model_name

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