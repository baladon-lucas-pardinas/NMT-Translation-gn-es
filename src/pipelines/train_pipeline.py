from src.model import model
from src.components import data_ingestion
from src.config.ingestion_config import DataIngestionConfig
from src.config.command_config import CommandConfig

# TODO: Replace will_x with a config object
def train(
        model_name, 
        config, 
        data_ingestion_config,
        train_dirs,
        validation_dirs,
        test_dirs,
        vocab_dirs,
        persist_each = 10000,
        save_each_epochs=None,
        will_ingest=False,
        will_train=False,
    ):
    # type: (str, CommandConfig, DataIngestionConfig, list, list, list, str, int, int, bool, bool) -> None
    if will_ingest:
        data_ingestion.ingest_data(data_ingestion_config, train_dirs, validation_dirs, test_dirs, vocab_dirs, persist_each)

    if will_train:
        model.train(model_name, config, save_each_epochs)