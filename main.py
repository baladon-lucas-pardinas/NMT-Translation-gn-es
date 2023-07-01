import argparse

from src.pipelines import train_pipeline
from src.utils import command_handler
from src.logger import logging
from src.config import command_config as command, ingestion_config as ingestion, data_transformation_config
from src.config.config import load_config_variables, FLAG_SEPARATOR

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--flags', type=str, required=True)

    # Training
    parser.add_argument('--command-path', type=str, required=True)
    parser.add_argument('--validate-each-epochs', type=int, required=False, default=None)
    parser.add_argument('--validation_metrics', type=str, required=False, default=None)
    parser.add_argument('--save-checkpoints', action='store_true', required=False, default=False)
    parser.add_argument('--not-delete-model-after', action='store_true', required=False, default=False)

    # Pipeline
    parser.add_argument('--ingest', action='store_true', required=False, default=False)
    parser.add_argument('--train', action='store_true', required=False, default=False)
    parser.add_argument('--transform', action='store_true', required=False, default=False)
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
    
    config_variables = load_config_variables()
    flag_separator   = config_variables.get(FLAG_SEPARATOR, ' ')
    command_config, ingestion_config, transformation_config = [None] * 3

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
        command_config = command.get_command_config(command_path, flags, validate_each_epochs=validate_each_epochs, validation_metrics=validation_metrics, 
                                                    save_checkpoints=save_checkpoints, not_delete_model_after=not_delete_model_after)
        logging.info('Training model with config {}'.format(command_config))

    try:
        train_pipeline.train(
            command_config=command_config, 
            data_ingestion_config=ingestion_config,
            data_transformation_config=transformation_config,
        )
    except Exception as e:
        logging.error('Error while training with config {} and {}'.format(command_config, ingestion_config))
        raise e