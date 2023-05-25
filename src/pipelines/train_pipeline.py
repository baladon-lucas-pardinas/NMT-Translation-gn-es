from src.model import model
from src.components import data_ingestion
from src.config.ingestion_config import DataIngestionConfig
from src.config.command_config import CommandConfig

def train(
        model_name, 
        config, 
        data_ingestion_config,
        train_dirs,
        validation_dirs,
        test_dirs,
        persist_each = 10000,
        save_each_epochs=None,
    ):
    # type: (str, CommandConfig, DataIngestionConfig, list, list, list, int, int) -> None
    columns_to_clean = data_ingestion_config.raw_data_columns_to_clean
    for train_dir, validation_dir, test_dir, column_to_clean in zip(train_dirs, validation_dirs, test_dirs, columns_to_clean):
        data_ingestion.ingest_data(data_ingestion_config, column_to_clean, train_dir, validation_dir, test_dir, persist_each)
    model.train(model_name, config, save_each_epochs)