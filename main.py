import argparse

from src.pipelines import train_pipeline
from src.utils import command_handler
from src.logger import logging
from src.config import command_config as command, ingestion_config as ingestion, data_transformation_config, hyperparameter_tuning_config
from src.config.config import load_config_variables, FLAG_SEPARATOR

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--flags', type=str, required=True)

    # Training
    parser.add_argument('--command-path', type=str, required=True, help='Path to the model executable')
    parser.add_argument('--validate-each-epochs', type=int, required=False, default=None, help='Number of epochs between validations')
    parser.add_argument('--validation_metrics', type=str, required=False, default=None, help='Whitespace separated list of metrics to use for validation')
    parser.add_argument('--save-checkpoints', action='store_true', required=False, default=False, help='Save a copy of the model after each validation')
    parser.add_argument('--not-delete-model-after', action='store_true', required=False, default=False, help='Do not delete the model after training')

    # Pipeline
    parser.add_argument('--ingest', action='store_true', required=False, default=False)
    parser.add_argument('--train', action='store_true', required=False, default=False)
    parser.add_argument('--transform', action='store_true', required=False, default=False)

    # Tuning
    parser.add_argument('--hyperparameter-tuning', action='store_true', required=False, default=False, help='Run hyperparameter tuning')
    parser.add_argument('--tuning-grid-files', type=str, required=False, default=None, help='Whitespace separated list of files with a grid of configurations')
    parser.add_argument('--tuning-params_files', type=str, required=False, default=None, help='Whitespace separated list of files with only one configuration')
    parser.add_argument('--search-method', type=str, required=False, default='grid', help='Search method for hyperparameter tuning (grid or random)')
    parser.add_argument('--seed', type=int, required=False, default=None, help='Seed for random search')

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    flags                  = args.get('flags')
    command_path           = args.get('command_path')
    validate_each_epochs   = args.get('validate_each_epochs')
    validation_metrics     = args.get('validation_metrics')
    save_checkpoints       = args.get('save_checkpoints')
    not_delete_model_after = args.get('not_delete_model_after')
    ingest                 = args.get('ingest')
    transform              = args.get('transform')
    train                  = args.get('train')
    hyperparameter_tuning  = args.get('hyperparameter_tuning')
    tuning_grid_files      = args.get('tuning_grid_files')
    tuning_params_files    = args.get('tuning_params_files')
    
    config_variables = load_config_variables()
    flag_separator   = config_variables.get(FLAG_SEPARATOR, ' ')
    command_config, ingestion_config, transformation_config, tuning_config = 4*[None]

    flags = command_handler.parse_flags(flags, flag_separator=flag_separator)
    train_dirs = flags.get('train-sets', [])
    val_dirs   = flags.get('valid-sets', [])
    logging.info('Running with flags {}'.format(flags))

    if ingest:
        vocab_dirs = flags.get('vocabs', [])
        ingestion_config = ingestion.get_data_ingestion_config(config_variables, train_dirs, val_dirs, vocab_dirs, persist_each=1000)
        logging.info('Ingesting data with config {}'.format(ingestion_config))
    if transform:
        transformation_config = data_transformation_config.get_data_transformation_config()
        logging.info('Transforming data with config {}'.format(transformation_config))
    if train:
        validation_metrics = validation_metrics.split(' ') if validation_metrics else None
        tuning_grid_files = tuning_grid_files.split(' ') if tuning_grid_files else []
        tuning_params_files = tuning_params_files.split(' ') if tuning_params_files else []
        command_config = command.get_command_config(command_path, flags, validate_each_epochs=validate_each_epochs, 
            validation_metrics=validation_metrics, save_checkpoints=save_checkpoints, not_delete_model_after=not_delete_model_after)
        logging.info('Training model with config {}'.format(command_config))

        if hyperparameter_tuning:
            search_method = args.get('search_method')
            seed = args.get('seed')
            tuning_config = hyperparameter_tuning_config.get_hyperparameter_tuning_config(
                tuning_grid_files=tuning_grid_files, tuning_params_files=tuning_params_files, 
                search_method=search_method, seed=seed)
            logging.info('Hyperparameter tuning with config {}'.format(tuning_config))

    try:
        train_pipeline.train(
            command_config=command_config, 
            data_ingestion_config=ingestion_config,
            data_transformation_config=transformation_config,
            hyperparameter_tuning_config=tuning_config
        )
    except Exception as e:
        logging.error('Error while training with config {} and {}'.format(command_config, ingestion_config))
        raise e