import os

from src.logger import logging
from src.model import model
from src.components import data_ingestion
from src.config.ingestion_config import DataIngestionConfig
from src.config.command_config import CommandConfig
from src.config.data_transformation_config import DataTransformationConfig
from src.config.hyperparameter_tuning_config import HyperparameterTuningConfig
from src.components import hyperparameter_tuning

def get_hyperparameter_flags(default_flags, hyperparameter_space, hyperparameter_configs, search_method, seed=None, max_iters=None):
    # type: (list[str], list[list[dict]], list[dict], str, int, int) -> list[dict]
    hyperparameter_flags = []
    for hyperparameters in hyperparameter_space:
        hyperparameter_flags.extend(hyperparameter_tuning.get_hyperparameters_flags(default_flags, hyperparameters, search_method, seed=seed, max_iters=max_iters))
    hyperparameter_configs_flags = [hyperparameter_tuning.get_custom_config_flags(default_flags, hyperparamter_config) for hyperparamter_config in hyperparameter_configs]
    trained_flags = [*hyperparameter_configs_flags, *hyperparameter_flags]
    return trained_flags

def save_checkpoint(temp_file, checkpoint):
    with open(temp_file, 'w') as f:
        f.write(checkpoint)

def load_checkpoint(temp_file):
    if not os.path.isfile(temp_file):
        return 0
    
    with open(temp_file, 'r') as f:
        return int(f.read())

def delete_checkpoint(temp_file):
    os.remove(temp_file)

def create_checkpoint_temp_dir_name(id):
    return 'temp_{}.txt'.format(id)

# from_flag and to_flag can be integers in {0, 1, .., |flag_combinations|} or floats in [0, 1]
def get_to_and_from_flags_indices(from_flag, to_flag, flag_combinations):
    # type: (int, int, list[dict]) -> tuple[int, int]
    flag_combinations_n = len(flag_combinations) 
    from_flag = from_flag if from_flag is not None else 0
    to_flag = to_flag if to_flag is not None else flag_combinations_n

    if '.' in str(from_flag) and (0 <= float(from_flag) <= 1):
        from_flag = int(from_flag*flag_combinations_n)
    if '.' in str(to_flag) and (0 <= float(to_flag) <= 1):
        to_flag = int(to_flag*flag_combinations_n)

    return from_flag, to_flag     

def train(
    data_ingestion_config,
    data_transformation_config,
    command_config,
    hyperparameter_tuning_config,
):
    # type: (DataIngestionConfig, DataTransformationConfig, CommandConfig, HyperparameterTuningConfig) -> None
    default_flags, run_id = (command_config.flags, command_config.run_id) if command_config else ({}, None)
    trained_flags = [default_flags]
    from_flag, to_flag = 2*[None]

    if hyperparameter_tuning_config is not None:
        hyperparamter_grids = hyperparameter_tuning_config.tuning_grid_files
        hyperparameter_configs = hyperparameter_tuning_config.tuning_params_files
        tuning_strategy = hyperparameter_tuning_config.tuning_strategy
        from_flag = hyperparameter_tuning_config.from_flags
        to_flag = hyperparameter_tuning_config.to_flags
        seed = hyperparameter_tuning_config.seed
        max_iters = hyperparameter_tuning_config.max_iters
        trained_flags = get_hyperparameter_flags(default_flags, hyperparamter_grids, hyperparameter_configs, tuning_strategy, seed, max_iters)
    logging.info('Starting training with {} flag combinations'.format(len(trained_flags)))

    temp_dir = create_checkpoint_temp_dir_name(run_id)

    if from_flag is not None:
        logging.info('Starting training from flag combination {}'.format(from_flag))
    else:
        from_flag = load_checkpoint(temp_dir)
        logging.info('Loaded training checkpoint: {}'.format(from_flag))

    from_flag, to_flag = get_to_and_from_flags_indices(from_flag, to_flag, trained_flags)
    idx = from_flag # Enumerate shouldn't be used in loop as idx would start in 0 after the checkpoint is loaded
    for current_flags in trained_flags[from_flag:to_flag]:
        save_checkpoint(temp_dir, str(idx))
        logging.info('Training checkpoint: {} saved correctly'.format(idx))

        if data_ingestion_config:
            data_ingestion.ingest_data(data_ingestion_config)

        if data_transformation_config:
            # Data transformation not implemented
            pass

        if command_config:
            command_config.flags = current_flags
            model.train(command_config)

        idx += 1

    delete_checkpoint(temp_dir)
    logging.info('Temp checkpoint file {} deleted'.format(temp_dir))