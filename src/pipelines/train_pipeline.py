from src.model import model
from src.components import data_ingestion
from src.config.ingestion_config import DataIngestionConfig
from src.config.command_config import CommandConfig

def train(
        command_config,
        data_ingestion_config,
    ):
    # type: (DataIngestionConfig, CommandConfig) -> None

    # this should be a loop over each config (maybe config should be a list of configs)
    if data_ingestion_config:
        data_ingestion.ingest_data(data_ingestion_config)

    if command_config:
        model.train(command_config) 

    #optimization: the same config can be used for multiple epochs

    # for each config
    # ingest
    # train on train data
    # evaluate on validation data
    # save results (model, metrics, etc.)
    # if the model is better than the previous one:
    #   save the model
    #   plot results
