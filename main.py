import argparse
import os

from src.pipelines import train_pipeline
from src.utils import command_handler
from src.logger import logging
from src.config import command_config as command, ingestion_config as ingestion, data_transformation_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command-path', type=str, required=True)
    parser.add_argument('--save-each-epochs', type=int, required=False)
    parser.add_argument('--ingest', action='store_true', required=False, default=False)
    parser.add_argument('--train', action='store_true', required=False, default=False)
    parser.add_argument('--transform', action='store_true', required=False, default=False)
    parser.add_argument('--flags', type=str, required=True)
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    command_path     = args.get('command_path')
    save_each_epochs = args.get('save_each_epochs')
    ingest           = args.get('ingest')
    transform        = args.get('transform')
    train            = args.get('train')
    flags            = args.get('flags')
    
    command_config = None
    ingestion_config = None

    flags = command_handler.parse_flags(flags)
    if ingest:
        ingestion_config = ingestion.get_data_ingestion_config(persist_each=1000)
        logging.info('Ingesting data with config {}'.format(ingestion_config))
    if transform:
        transformation_config = data_transformation_config.get_data_transformation_config()
        logging.info('Transforming data with config {}'.format(data_transformation_config))
    if train:
        command_config = command.get_command_config(command_path, flags, save_each_epochs=save_each_epochs)
        logging.info('Training model with config {}'.format(command_config))

    train_dir  = flags.get('train-sets', [])
    val_dir    = flags.get('valid-sets', [])
    test_dirs  = [os.path.join(ingestion_config.test_data_dir, 'test_gn.txt'), os.path.join(ingestion_config.test_data_dir, 'test_es.txt')] #TODO: this should be a flag
    vocab_dirs = flags.get('vocabs', [])

    ingestion_config.train_data_dir      = train_dir
    ingestion_config.validation_data_dir = val_dir
    ingestion_config.test_data_dir       = test_dirs
    ingestion_config.vocabulary_dir      = vocab_dirs

    try:
        train_pipeline.train(
            command_config=command_config, 
            data_ingestion_config=ingestion_config,
            data_transformation_config=transformation_config
        )
    except Exception as e:
        logging.error('Error while training with config {} and {}'.format(command_config, ingestion_config))
        raise e