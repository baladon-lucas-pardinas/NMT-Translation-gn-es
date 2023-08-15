import csv

from src.config.ingestion_config import DataIngestionConfig
from src.logger import logging
from src.components.processing import tokenization

def __persist_split_data(splits, split):
        splits[split]['file'].write('\n'.join(splits[split]['data']))
        splits[split]['data'] = []

def split_dataset(
    raw_data_file_path, 
    raw_data_columns, 
    raw_data_split_column, 
    column_to_clean, 
    train_output, 
    validation_output, 
    test_output, 
    persist_each=10000
):
    # type: (str, list, str, str, str, str, str, int) -> None
    logging.info("Splitting data from {}...".format(raw_data_file_path))
    train_output_path = train_output + '.' + column_to_clean
    validation_output_path = validation_output + '.' + column_to_clean
    test_output_path = test_output + '.' + column_to_clean

    logging.info("Writing train data to {}...".format(train_output_path))
    logging.info("Writing validation data to {}...".format(validation_output_path))
    logging.info("Writing test data to {}...".format(test_output_path))

    with open(raw_data_file_path, 'r', encoding='utf-8') as raw_f, \
         open(train_output_path, 'w', encoding='utf-8') as train_f, \
         open(validation_output_path, 'w', encoding='utf-8') as validation_f, \
         open(test_output_path, 'w', encoding='utf-8') as test_f:
        
        train_column, validation_column, test_column = raw_data_columns
        splits = {
            train_column:      {'data': [], 'file': train_f,      'count': 0},
            validation_column: {'data': [], 'file': validation_f, 'count': 0},
            test_column:       {'data': [], 'file': test_f,       'count': 0},
        }

        reader = csv.reader(raw_f)
        columns = next(reader)
        column_to_clean_index = columns.index(column_to_clean)
        split_column_index = columns.index(raw_data_split_column)

        for row in reader:
            split = row[split_column_index]
            splits[split]['count'] += 1

            text = row[column_to_clean_index]
            splits[split]['data'].append(text)

            if len(splits[split]['data']) % persist_each == 0:
                __persist_split_data(splits, split)

        for split in splits:
            __persist_split_data(splits, split)
                    
        logging.info("Train data count: {}".format(splits[train_column]['count']))
        logging.info("Validation data count: {}".format(splits[validation_column]['count']))
        logging.info("Test data count: {}".format(splits[test_column]['count']))
    logging.info("Ingestion complete.")

def create_vocabularies(
    raw_data_file_path,
    raw_data_train_column,
    raw_data_split_column,
    column_to_ingest, 
    train_vocab_output, 
    default_vocabulary=[]
):
    # type: (str, str, str, str, str, list) -> None
    logging.info("Creating vocabulary from {}...".format(raw_data_file_path))
    train_vocab_output_path = train_vocab_output + '.' + column_to_ingest
    logging.info("Writing train vocabulary to {}...".format(train_vocab_output_path))

    tokenizer = tokenization.get_tokenizer()

    with open(raw_data_file_path, 'r', encoding='utf-8') as raw_f, \
            open(train_vocab_output_path, 'w', encoding='utf-8') as train_vocab_f:
        
        train_column = raw_data_train_column
        splits = {
            train_column: {'data': set(), 'file': train_vocab_f, 'count': 0},
        }

        reader = csv.reader(raw_f)
        columns = next(reader)
        column_to_clean_index = columns.index(column_to_ingest)
        split_column_index = columns.index(raw_data_split_column)

        # Default vocabulary should be written first.
        for word in default_vocabulary:
            splits[train_column]['file'].write(word + '\n')

        for row in reader:
            split = row[split_column_index]
            if split == train_column:
                text = row[column_to_clean_index]
                tokenized_text = tokenizer.tokenize(text)
                splits[split]['count'] += len(tokenized_text)
                splits[split]['data'].update(tokenized_text)

        for split in splits:
            __persist_split_data(splits, split)
                    
        logging.info("Train vocabulary count: {}".format(splits[train_column]['count']))
    logging.info("Vocabulary creation complete.")

def split_augmented_data(
    raw_augmented_data_file_path,
    augmented_data_output_path,
    language_extensions,
    persist_each,
    separated_by=','
):
    # type: (str, str, list, int, str) -> None
    logging.info("Creating augmented vocabulary from {}...".format(raw_augmented_data_file_path))
    raw_augmented_data_file_paths = [augmented_data_output_path + '.' + extension for extension in language_extensions[::-1]]
    logging.info("Writing train vocabulary to {}...".format(raw_augmented_data_file_paths))

    with open(raw_augmented_data_file_path, 'r', encoding='utf-8') as raw_f, \
         open(raw_augmented_data_file_paths[0], 'w', encoding='utf-8') as augmented_data1_f, \
         open(raw_augmented_data_file_paths[1], 'w', encoding='utf-8') as augmented_data2_f:
        
        splits = {
            extension: {'data': [], 'file': file, 'count': 0} \
                for extension, file in zip(language_extensions, [augmented_data1_f, augmented_data2_f])
        }

        for line in raw_f:
            splitted_line = line.split(separated_by)
            for ext, data in zip(language_extensions, splitted_line):
                splits[ext]['count'] += 1
                splits[ext]['data'].append(data.replace('\n', ''))
            
                if splits[ext]['count'] % persist_each == 0:
                    __persist_split_data(splits, ext)
                    logging.info("Vocabulary count for {}: {}".format(ext, splits[ext]['count']))
    logging.info("Vocabulary creation complete.")
    
    return None

# TODO: Check that length of src and target are equal.
def ingest_data(data_ingestion_config):
        # type: (DataIngestionConfig,) -> None
        train_split_outputs           = [data_ingestion_config.train_data_src_dir, 
                                         data_ingestion_config.train_data_tgt_dir]
        validation_split_outputs      = [data_ingestion_config.validation_data_src_dir, 
                                         data_ingestion_config.validation_data_tgt_dir]
        test_split_outputs            = [data_ingestion_config.test_data_src_dir, 
                                         data_ingestion_config.test_data_tgt_dir]
        vocab_outputs                 = [data_ingestion_config.vocab_src_dir, 
                                         data_ingestion_config.vocab_tgt_dir]
        raw_data_columns              = [data_ingestion_config.raw_data_train_column, 
                                         data_ingestion_config.raw_data_validation_column, 
                                         data_ingestion_config.raw_data_test_column]
        columns_to_ingest             = data_ingestion_config.raw_data_columns_to_clean
        ingest_augmented_data         = data_ingestion_config.ingest_augmented_data
        raw_augmented_data_file_path  = data_ingestion_config.raw_augmented_data_file_path
        augmented_data_output_path    = data_ingestion_config.augmented_data_output_path
        persist_each                  = data_ingestion_config.persist_each

        for train_dir, validation_dir, test_dir, column_to_ingest in \
              zip(train_split_outputs, validation_split_outputs, test_split_outputs, columns_to_ingest):
            split_dataset(
                data_ingestion_config.raw_data_file_path,
                raw_data_columns,
                data_ingestion_config.raw_data_split_column,
                column_to_ingest,
                train_dir,
                validation_dir,
                test_dir,
                persist_each=persist_each
            )

        for column_to_ingest, vocab_output in zip(columns_to_ingest, vocab_outputs):
            create_vocabularies(
                data_ingestion_config.raw_data_file_path,
                data_ingestion_config.raw_data_train_column,
                data_ingestion_config.raw_data_split_column,
                column_to_ingest,
                vocab_output,
                default_vocabulary=data_ingestion_config.default_vocabulary
            )

        if ingest_augmented_data:
            split_augmented_data(raw_augmented_data_file_path, augmented_data_output_path, columns_to_ingest, persist_each=persist_each)