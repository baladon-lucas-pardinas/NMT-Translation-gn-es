import argparse
import os
import sys
from src.pipelines import train_pipeline
from src.utils import command_handler
from src.logger import logging
from src.config import command_config as command, ingestion_config as ingestion

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command-path', type=str, required=True)
    parser.add_argument('--save-each-epochs', type=int, required=False)
    parser.add_argument('--ingest', action='store_true', required=False, default=False)
    parser.add_argument('--train', action='store_true', required=False, default=False)
    parser.add_argument('--flags', type=str, required=True)
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    command_path = args.get('command_path')
    save_each_epochs = args.get('save_each_epochs')
    ingest = args.get('ingest')
    train = args.get('train')
    flags = args.get('flags')
    
    flags = command_handler.parse_flags(flags)
    command_config = command.get_command_config(command_path=command_path, flags=flags) #TODO: configs should be centralized in a single module.
    ingestion_config = ingestion.get_data_ingestion_config()
    train_dir = flags.get('train-sets', [])
    val_dir = flags.get('valid-sets', [])
    test_dirs = [os.path.join(ingestion_config.test_data_dir, 'test_gn.txt'), os.path.join(ingestion_config.test_data_dir, 'test_es.txt')]
    vocab_dirs = flags.get('vocabs', [])

    logging.info('Training model with config {}'.format(command_config))
    try:
        train_pipeline.train(
            model_name=command_config.command_name,
            config=command_config,
            data_ingestion_config=ingestion_config,
            train_dirs=train_dir,
            validation_dirs=val_dir,
            test_dirs=test_dirs,
            vocab_dirs=vocab_dirs,
            persist_each=1000,
            save_each_epochs=save_each_epochs,
            will_train=train,
            will_ingest=ingest,
        )
    except Exception as e:
        logging.error('Error while training with config {}'.format(command_config))
        raise e