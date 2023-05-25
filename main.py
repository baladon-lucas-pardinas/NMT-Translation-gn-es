import os
import sys
from src.pipelines import train_pipeline
from src.utils import command_handler
from src.logger import logging
from src.config import command_config as command, ingestion_config as ingestion

if __name__ == '__main__':
    command_path = sys.argv[1]
    save_each_epochs = int(sys.argv[2])
    flags = sys.argv[3:]
    
    flags = ' '.join(flags)
    flags = command_handler.parse_flags(flags)
    command_config = command.get_command_config(command_path=command_path, flags=flags)
    ingestion_config = ingestion.get_data_ingestion_config()
    train_dir = flags.get('train-sets', [])
    val_dir = flags.get('valid-sets', [])
    test_dirs = [
        os.path.join(ingestion_config.test_data_dir, 'test_gn.txt'),
        os.path.join(ingestion_config.test_data_dir, 'test_es.txt')
    ]
    logging.info('Training model with config {}'.format(command_config))

    try:
        train_pipeline.train(
            model_name=command_config.command_name,
            config=command_config,
            data_ingestion_config=ingestion_config,
            train_dirs=train_dir,
            validation_dirs=val_dir,
            test_dirs=test_dirs,
            persist_each=1000,
            save_each_epochs=save_each_epochs
        )
    except Exception as e:
        logging.error('Error while training with config {}'.format(command_config))
        raise e