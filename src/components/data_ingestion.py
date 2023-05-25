import csv
import os

from src.config.ingestion_config import DataIngestionConfig
from src.logger import logging

def __persist_split_data(splits, split):
        splits[split]['file'].write('\n'.join(splits[split]['data']))
        splits[split]['data'] = []

# TODO: Check that length of src and target are equal.
def ingest_data(data_ingestion_config, column_to_clean, train_output, validation_output, test_output, persist_each=10000):
    # type: (DataIngestionConfig, str, str, str, str, int) -> None
    
    logging.info("Ingesting data from {}...".format(data_ingestion_config.raw_data_dir))
    train_output_path = train_output + '.' + column_to_clean
    validation_output_path = validation_output + '.' + column_to_clean
    test_output_path = test_output + '.' + column_to_clean

    logging.info("Writing train data to {}...".format(train_output_path))
    logging.info("Writing validation data to {}...".format(validation_output_path))
    logging.info("Writing test data to {}...".format(test_output_path))

    with open(data_ingestion_config.raw_data_file_path, 'r', encoding='utf-8') as raw_f, \
            open(train_output_path, 'w', encoding='utf-8') as train_f, \
            open(validation_output_path, 'w', encoding='utf-8') as validation_f, \
            open(test_output_path, 'w', encoding='utf-8') as test_f:
        
        train_column = data_ingestion_config.raw_data_train_column
        validation_column = data_ingestion_config.raw_data_validation_column
        test_column = data_ingestion_config.raw_data_test_column
        splits = {
            train_column:      {'data': [], 'file': train_f,      'count': 0},
            validation_column: {'data': [], 'file': validation_f, 'count': 0},
            test_column:       {'data': [], 'file': test_f,       'count': 0},
        }

        reader = csv.reader(raw_f)
        columns = next(reader)
        column_to_clean_index = columns.index(column_to_clean)
        split_column_index = columns.index(data_ingestion_config.raw_data_split_column)

        for row in reader:
            split = row[split_column_index]
            splits[split]['count'] += 1

            text = row[column_to_clean_index]
            splits[split]['data'].append(text)

            if len(splits[split]['data']) >= persist_each:
                __persist_split_data(splits, split)

        for split in splits:
            __persist_split_data(splits, split)
                    
        logging.info("Train data count: {}".format(splits[train_column]['count']))
        logging.info("Validation data count: {}".format(splits[validation_column]['count']))
        logging.info("Test data count: {}".format(splits[test_column]['count']))
    logging.info("Ingestion complete.")




