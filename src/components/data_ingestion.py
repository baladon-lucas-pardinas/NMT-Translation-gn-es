import csv
import os

from src.config.ingestion_config import DataIngestionConfig
from src.logger import logging
from src.domain.processing import tokenization

def __persist_split_data(splits, split):
        splits[split]['file'].write('\n'.join(splits[split]['data']) + '\n')
        splits[split]['data'] = []

def rename_file(filename, language):
    # type: (str, str) -> str
    return filename + '.' + language

def split_dataset(raw_data_file_path,
                  raw_data_columns,
                  raw_data_split_column,
                  column_to_clean,
                  train_output,
                  validation_output,
                  test_output,
                  persist_each=10000):
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

def split_augmented_data(raw_augmented_data_file_path,
                         augmented_data_output_path,
                         language_extensions,
                         persist_each,
                         separated_by='\t'):
    # type: (str, str, list, int, str) -> None
    logging.info("Creating augmented set from {}...".format(raw_augmented_data_file_path))
    raw_augmented_data_file_paths = [augmented_data_output_path + '.' + extension 
                                     for extension in language_extensions[::-1]]
    logging.info("Writing train set to {}...".format(raw_augmented_data_file_paths))

    with open(raw_augmented_data_file_path, 'r', encoding='utf-8') as raw_f, \
         open(raw_augmented_data_file_paths[0], 'w', encoding='utf-8') as augmented_data1_f, \
         open(raw_augmented_data_file_paths[1], 'w', encoding='utf-8') as augmented_data2_f:
        
        splits = {extension: {'data': [], 'file': file, 'count': 0} \
                  for extension, file in 
                  zip(language_extensions, [augmented_data1_f, augmented_data2_f])}

        for line in raw_f:
            splitted_line = line.split(separated_by)
            for ext, data in zip(language_extensions, splitted_line):
                splits[ext]['count'] += 1
                splits[ext]['data'].append(data.replace('\n', ''))
            
                if splits[ext]['count'] % persist_each == 0:
                    __persist_split_data(splits, ext)
                    logging.info("Vocabulary count for {}: {}"
                                 .format(ext, splits[ext]['count']))

    logging.info("Vocabulary creation complete.")

def create_vocabulary(input_path, 
                      output_path, 
                      tokenizer_type='spacy', 
                      default_vocabulary=[]):
    # type: (str, str, str, list) -> None
    sentences = list()

    with open(input_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    tokenizer = tokenization.get_tokenizer(tokenizer=tokenizer_type)
    tokens = [token for sentence in sentences 
                    for token in tokenizer.tokenize(sentence)]
    tokens = list(set(tokens))

    with open(output_path, 'w', encoding='utf-8') as f:
        if len(default_vocabulary) > 0:
            f.write('\n'.join(default_vocabulary))
            f.write('\n')
        f.write('\n'.join(tokens))
    
def append_augmented_data(augmented_filename,
                         train_files,
                         language_extensions,
                         output_filename,
                         persist_each):
    # type: (str, list, list, str, int) -> None
    augmented_files = [augmented_filename + '.' + extension 
                       for extension in language_extensions[::-1]]
    train_files     = [train_file + '.' + extension 
                       for train_file, extension in 
                       zip(train_files, language_extensions)][::-1]
    output_files    = [output_filename + '.' + extension 
                       for extension in language_extensions[::-1]]

    logging.info("Writing augmented and train data to {}...".format(output_filename))
    with open(augmented_files[0], 'r', encoding='utf-8') as augmented_data1_f, \
         open(augmented_files[1], 'r', encoding='utf-8') as augmented_data2_f, \
         open(train_files[0], 'r', encoding='utf-8') as train_data1_f, \
         open(train_files[1], 'r', encoding='utf-8') as train_data2_f, \
         open(output_files[0], 'w', encoding='utf-8') as full_data1_f, \
         open(output_files[1], 'w', encoding='utf-8') as full_data2_f:
        
        splits = {extension: {'data': [], 'file': file, 'count': 0} \
                  for extension, file in 
                  zip(language_extensions, [full_data1_f, full_data2_f])}

        for ext, train_f, aug_f in zip(language_extensions, 
                                       [train_data1_f, train_data2_f], 
                                       [augmented_data1_f, augmented_data2_f]):
            for file in [train_f, aug_f]:
                for line in file:
                    splits[ext]['count'] += 1
                    splits[ext]['data'].append(line.replace('\n', ''))
        
                    if splits[ext]['count'] % persist_each == 0:
                        __persist_split_data(splits, ext)
            logging.info("Vocabulary count for {}: {}".format(ext, splits[ext]['count']))

    return

# TODO: Check that length of src and target are equal.
def ingest_data(data_ingestion_config):
        # type: (DataIngestionConfig,) -> None
        columns_to_ingest = data_ingestion_config.raw_data_columns_to_clean
        ingest_augmented_data = data_ingestion_config.ingest_augmented_data
        raw_augmented_data_file_path = data_ingestion_config.raw_augmented_data_file_path
        augmented_data_output_path = data_ingestion_config.augmented_data_output_path
        full_augmented_data_output_path = data_ingestion_config.full_augmented_data_output_path
        persist_each = data_ingestion_config.persist_each
        
        train_split_outputs = [data_ingestion_config.train_data_src_dir, 
                               data_ingestion_config.train_data_tgt_dir]
        validation_split_outputs = [data_ingestion_config.validation_data_src_dir, 
                                    data_ingestion_config.validation_data_tgt_dir]
        test_split_outputs = [data_ingestion_config.test_data_src_dir, 
                              data_ingestion_config.test_data_tgt_dir]
        vocab_outputs = [data_ingestion_config.vocab_src_dir, 
                         data_ingestion_config.vocab_tgt_dir]
        raw_data_columns = [data_ingestion_config.raw_data_train_column, 
                            data_ingestion_config.raw_data_validation_column, 
                            data_ingestion_config.raw_data_test_column]

        zipped_splits = zip(train_split_outputs, validation_split_outputs, 
                            test_split_outputs, columns_to_ingest)
        for train_dir, validation_dir, test_dir, column_to_ingest in zipped_splits:
            split_dataset(data_ingestion_config.raw_data_file_path,
                          raw_data_columns,
                          data_ingestion_config.raw_data_split_column,
                          column_to_ingest,
                          train_dir, validation_dir, test_dir,
                          persist_each=persist_each)

        for train_dir, vocab_output, language in zip(train_split_outputs, 
                                                     vocab_outputs, 
                                                     columns_to_ingest):
            train_file = rename_file(train_dir, language)
            vocab_file = rename_file(vocab_output, language)
            create_vocabulary(train_file, vocab_file, 
                default_vocabulary=data_ingestion_config.default_vocabulary)

        if ingest_augmented_data:
            split_augmented_data(raw_augmented_data_file_path, 
                                 augmented_data_output_path, 
                                 columns_to_ingest,
                                 persist_each=persist_each)
            append_augmented_data(augmented_data_output_path, 
                                  train_split_outputs, 
                                  columns_to_ingest, 
                                  full_augmented_data_output_path, 
                                  persist_each=persist_each)

            for vocab_path, language in zip(vocab_outputs, columns_to_ingest):
                vocabulary_dir = os.path.dirname(vocab_path)
                full_augmented_vocab_path = os.path.join(vocabulary_dir, 
                                                         'full_augmented_vocab')
                input_corpus_path = rename_file(full_augmented_data_output_path, 
                                                language)
                vocab_output_path = rename_file(full_augmented_vocab_path, 
                                                language)
                create_vocabulary(input_corpus_path, vocab_output_path, 
                    default_vocabulary=data_ingestion_config.default_vocabulary)