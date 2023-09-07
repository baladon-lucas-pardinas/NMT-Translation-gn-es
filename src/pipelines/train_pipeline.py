import os

from src.logger import logging
from src.components import model_trainer, data_ingestion, hyperparameter_tuning, finetuning
from src.config.ingestion_config import DataIngestionConfig
from src.config.command_config import CommandConfig
from src.config.hyperparameter_tuning_config import HyperparameterTuningConfig
from src.config.finetuning_config import FinetuningConfig
from src.utils import file_manager, parsing

def get_hyperparameter_flags(default_flags, hyperparameter_space, hyperparameter_configs, search_method, seed=None, max_iters=None):
    # type: (list[str], list[list[dict]], list[dict], str, int, int) -> list[dict]
    hyperparameter_flags = []
    for hyperparameters in hyperparameter_space:
        hyperparameter_file_flags = hyperparameter_tuning.get_hyperparameters_flags(
                                                                default_flags, 
                                                                hyperparameters, 
                                                                search_method, 
                                                                seed=seed, 
                                                                max_iters=max_iters)
        hyperparameter_flags.extend(hyperparameter_file_flags)
    hyperparameter_configs_flags = [
        hyperparameter_tuning.get_custom_config_flags(default_flags, hyperparamter_config) \
            for hyperparamter_config in hyperparameter_configs]

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

def has_sentencepiece_vocabulary(command_config):
    # type: (CommandConfig) -> bool
    first_vocab_file = command_config.flags.get('vocabs', [''])[0]
    return first_vocab_file.endswith('.spm')

def already_exists_vocabulary(command_config):
    # type: (CommandConfig) -> bool
    first_vocab_file = command_config.flags.get('vocabs', [''])[0]
    return os.path.isfile(first_vocab_file)

def handle_finetuning(command_config, finetuning_config):
    # type: (CommandConfig, FinetuningConfig) -> CommandConfig
    cached_model_dir = None
    new_model_path = None
    finetuning_epochs = int(finetuning_config.epochs)
    full_sets = finetuning_config.full_sets
    augmented_sets = finetuning_config.augmented_sets
    cache_dir_template = finetuning_config.cache_dir_template
    finetuned_model_path = command_config.flags.get('model')[0]
    finetuned_model_dir = os.path.dirname(finetuned_model_path)

    if finetuning_epochs == 0:
        return command_config

    finetuning_config, command_config.flags = \
        parsing.handle_finetuning_flags(finetuning_config, command_config.flags)

    if has_sentencepiece_vocabulary(command_config) \
        and not already_exists_vocabulary(command_config):
        finetuning_vocabulary_command_config = \
            finetuning.create_finetuning_vocabulary_train_config(
                                                command_config, 
                                                full_sets)
        
        logging.info("Creating finetuning vocabulary...")
        model_trainer.train(finetuning_vocabulary_command_config)

    if cache_dir_template is not None:
        cached_model_dir, epoch = finetuning.get_cached_pretrained_model_dir(
                                                        cache_dir_template,
                                                        finetuning_epochs)
        if cached_model_dir is not None:
            file_manager.save_copy(cached_model_dir, finetuned_model_dir)

    if cached_model_dir is None or str(epoch) != str(finetuning_epochs):
        finetuning_command_config = \
            finetuning.create_finetuning_train_config(command_config, 
                                                      augmented_sets, 
                                                      finetuning_epochs,)
        
        # Save copy of pretrained model for future cache use
        if cache_dir_template is not None:
            new_cache_dir = cache_dir_template.format(finetuning_epochs)
            if not os.path.exists(new_cache_dir):
                file_manager.save_copy(finetuned_model_dir, new_cache_dir)

        logging.info("Training pretrained model...")
        model_trainer.train(finetuning_command_config)

    command_config = finetuning.adapt_train_config(
                                command_config, 
                                finetuning_epochs, 
                                new_model_path)
    return command_config

def train(
    data_ingestion_config,
    command_config,
    hyperparameter_tuning_config,
    finetuning_config,
):
    # type: (DataIngestionConfig, CommandConfig, HyperparameterTuningConfig, FinetuningConfig) -> None
    default_flags, run_id = (command_config.flags, command_config.run_id) \
                                        if command_config else ({}, None)
    temp_dir = create_checkpoint_temp_dir_name(run_id)
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
        trained_flags = get_hyperparameter_flags(default_flags, 
                                                 hyperparamter_grids, 
                                                 hyperparameter_configs, 
                                                 tuning_strategy, 
                                                 seed,
                                                 max_iters)
        
    logging.info('Starting training with {} flag combinations'.format(len(trained_flags)))

    if from_flag is not None:
        logging.info('Starting training from flag combination {}'.format(from_flag))
    else:
        from_flag = load_checkpoint(temp_dir)
        logging.info('Loaded training checkpoint: {}'.format(from_flag))

    from_flag, to_flag = get_to_and_from_flags_indices(from_flag, to_flag, trained_flags)
    idx = from_flag # Enumerate shouldn't be used in loop as idx would start in 0 after the checkpoint is loaded

    # Training loop
    for current_flags in trained_flags[from_flag:to_flag]:
        save_checkpoint(temp_dir, str(idx))
        logging.info('Training checkpoint: {} saved correctly'.format(idx))

        if data_ingestion_config:
            data_ingestion.ingest_data(data_ingestion_config)

        if command_config:
            command_config.flags = current_flags

            if finetuning_config:
                command_config = handle_finetuning(command_config, finetuning_config)

            model_trainer.train(command_config)

        idx += 1

    delete_checkpoint(temp_dir)
    logging.info('Temp checkpoint file {} deleted'.format(temp_dir))