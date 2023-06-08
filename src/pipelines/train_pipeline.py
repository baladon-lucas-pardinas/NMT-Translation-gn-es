from src.model import model
from src.components import data_ingestion
from src.config.ingestion_config import DataIngestionConfig
from src.config.command_config import CommandConfig
from src.config.data_transformation_config import DataTransformationConfig

def train(
        data_ingestion_config,
        data_transformation_config,
        command_config,
    ):
    # type: (DataIngestionConfig, CommandConfig, DataTransformationConfig) -> None

    # this should be a loop over each config (maybe config should be a list of configs)
    if data_ingestion_config:
        data_ingestion.ingest_data(data_ingestion_config)

    if data_transformation_config:
        pass

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
