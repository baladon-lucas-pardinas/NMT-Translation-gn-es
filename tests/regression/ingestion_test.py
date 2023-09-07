import unittest
import os

from src.config.config import load_config_variables, \
    BASE_DIR_ARTIFACTS, \
    RAW_DATA_FILENAME, \
    RAW_DATA_COLUMNS_TO_CLEAN, \
    RAW_DATA_SPLIT_COLUMN, \
    RAW_DATA_TRAIN_COLUMN, \
    RAW_DATA_VALIDATION_COLUMN, \
    RAW_DATA_TEST_COLUMN

from src.components import data_ingestion
from src.domain.processing.search_duplicates import search_duplicates

# TODO: Test if there are empty lines or the default vocabulary is not at the beginning of the file
class TestIngestion(unittest.TestCase):
    def setUp(self) -> None:
        self.config_variables = load_config_variables()
        self.current_dir               = os.path.dirname(os.path.realpath(__file__))
        self.test_data_dir             = os.path.join(self.current_dir, '..', 'data')
        self.train_output_dir          = os.path.join(self.test_data_dir, 'train.ingestion')
        self.validation_output_dir     = os.path.join(self.test_data_dir, 'validation.ingestion')
        self.test_output_dir           = os.path.join(self.test_data_dir, 'test.ingestion')
        self.vocab_output_dir          = os.path.join(self.test_data_dir, 'vocab.ingestion')
        self.raw_data_filepath_dataset = os.path.join(self.config_variables[BASE_DIR_ARTIFACTS], 'data', 'raw', self.config_variables[RAW_DATA_FILENAME])

        # Split dataset
        self.raw_data_columns = [self.config_variables[RAW_DATA_TRAIN_COLUMN], self.config_variables[RAW_DATA_VALIDATION_COLUMN], self.config_variables[RAW_DATA_TEST_COLUMN]]
        self.raw_data_split_column = self.config_variables[RAW_DATA_SPLIT_COLUMN]
        self.column_to_ingest = self.config_variables[RAW_DATA_COLUMNS_TO_CLEAN][0]

        # Create vocabulary
        self.raw_data_train_column = self.config_variables[RAW_DATA_TRAIN_COLUMN]
        self.default_vocabulary = ['<PAD>', '<UNK>', '<S>', '</S>']
        pass

    def test_ingestion(self):
        try:
            data_ingestion.split_dataset(
                raw_data_file_path=self.raw_data_filepath_dataset,
                raw_data_columns=self.raw_data_columns,
                raw_data_split_column=self.raw_data_split_column,
                column_to_clean=self.column_to_ingest,
                train_output=self.train_output_dir,
                validation_output=self.validation_output_dir,
                test_output=self.test_output_dir,
                persist_each=1000
            )
            data_ingestion.create_vocabulary(
                input_path=self.train_output_dir+'.gn',
                output_path=self.vocab_output_dir,
                default_vocabulary=self.default_vocabulary
            )

        except Exception as e:
            self.fail("Ingestion failed with exception {}".format(e))

        should_exist_paths = [self.train_output_dir, self.validation_output_dir, self.test_output_dir, self.vocab_output_dir]
        vocabulary_indices = [should_exist_paths.index(self.vocab_output_dir)]
        should_exist_paths = [path + '.' + self.column_to_ingest for path in should_exist_paths]
        
        for path in should_exist_paths:
            if not os.path.exists(path):
                self.fail("Ingestion failed to create file {}".format(path))
            else:
                for index in vocabulary_indices:
                    vocabulary_path = should_exist_paths[index]
                    _, duplicate_indexes = search_duplicates(vocabulary_path, verbose=False)
                    if len(duplicate_indexes.keys()) > 0:
                        self.fail("Ingestion created duplicate word in file {}".format(path))
                os.remove(path)

def main():
    unittest.main()

if __name__ == '__main__':
    main()