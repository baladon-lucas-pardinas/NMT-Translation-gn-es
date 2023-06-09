import os

from src.config.config import \
    BASE_DIR_ARTIFACTS, \
    RAW_DATA_FILENAME, \
    RAW_DATA_COLUMNS_TO_CLEAN, \
    RAW_DATA_SPLIT_COLUMN, \
    RAW_DATA_TRAIN_COLUMN, \
    RAW_DATA_VALIDATION_COLUMN, \
    RAW_DATA_TEST_COLUMN, \
    TEST_SRC_CORPUS_FILENAME, \
    TEST_DST_CORPUS_FILENAME

class DataIngestionConfig:
    def __init__(
        self, 
        artifacts_dir,
        train_output_src_dir,
        train_output_dst_dir,
        validation_output_src_dir,
        validation_output_dst_dir,
        test_src_output_filename, 
        test_dst_output_filename, 
        vocab_src_output_filename,
        vocab_tgt_output_filename,
        raw_data_filename,
        columns_to_clean,
        split_column,
        train_column,
        validation_column,
        test_column,
        default_vocabulary=[],
        persist_each=None
    ):
        # type: (str, str, str, str, str, str, str, str, str, str, list, str, str, str, str, str, list, int) -> None
        self.default_vocabulary = default_vocabulary
        self.persist_each = persist_each
        self.artifacts_dir = artifacts_dir

        data_dir                      = os.path.join(artifacts_dir, 'data')
        self.data_dir                 = data_dir
        self.raw_data_dir             = os.path.join(data_dir, 'raw')
        self.corpora_dir              = os.path.join(data_dir, 'corpora')
        self.vocabulary_dir           = os.path.join(data_dir, 'vocabulary')
        self.train_data_dir           = os.path.join(data_dir, 'train')
        self.validation_data_dir      = os.path.join(data_dir, 'validation')
        self.test_data_dir            = os.path.join(data_dir, 'test')
        self.train_data_src_dir       = os.path.join(self.train_data_dir, train_output_src_dir)
        self.train_data_tgt_dir       = os.path.join(self.train_data_dir, train_output_dst_dir)
        self.validation_data_src_dir  = os.path.join(self.validation_data_dir, validation_output_src_dir)
        self.validation_data_tgt_dir  = os.path.join(self.validation_data_dir, validation_output_dst_dir)
        self.test_data_src_dir        = os.path.join(self.test_data_dir, test_src_output_filename)
        self.test_data_tgt_dir        = os.path.join(self.test_data_dir, test_dst_output_filename)
        self.vocab_src_dir            = os.path.join(self.vocabulary_dir, vocab_src_output_filename)
        self.vocab_tgt_dir            = os.path.join(self.vocabulary_dir, vocab_tgt_output_filename)
        self.raw_data_file_path       = os.path.join(self.raw_data_dir, raw_data_filename)

        self.raw_data_columns_to_clean  = columns_to_clean
        self.raw_data_split_column      = split_column
        self.raw_data_train_column      = train_column
        self.raw_data_validation_column = validation_column
        self.raw_data_test_column       = test_column


    def __str__(self):
        # type: () -> str
        return "DataIngestionConfig(persist_each={}, artifacts_dir={}, data_dir={}, raw_data_dir={}, corpora_dir={}, vocabulary_dir={}, train_data_dir={}, validation_data_dir={}, test_data_dir={}, test_data_src_dir={}, test_data_tgt_dir={}, raw_data_file_path={}, raw_data_columns_to_clean={}, raw_data_split_column={}, raw_data_train_column={}, raw_data_validation_column={}, raw_data_test_column={})".format(
            self.persist_each,
            self.artifacts_dir,
            self.data_dir,
            self.raw_data_dir,
            self.corpora_dir,
            self.vocabulary_dir,
            self.train_data_dir,
            self.validation_data_dir,
            self.test_data_dir,
            self.test_data_src_dir,
            self.test_data_tgt_dir,
            self.raw_data_file_path,
            self.raw_data_columns_to_clean,
            self.raw_data_split_column,
            self.raw_data_train_column,
            self.raw_data_validation_column,
            self.raw_data_test_column,
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_data_ingestion_config(
        config_variables,
        train_output_dirs,
        validation_output_dirs,
        vocab_dirs,
        persist_each=None
    ):
    # type: (dict, list, list, list, int) -> DataIngestionConfig
    train_output_src_dir, train_output_dst_dir = train_output_dirs
    validation_output_src_dir, validation_output_dst_dir = validation_output_dirs
    vocab_src_output_filename, vocab_tgt_output_filename = vocab_dirs
    
    return DataIngestionConfig(
        artifacts_dir            =config_variables[BASE_DIR_ARTIFACTS],
        train_output_src_dir     =train_output_src_dir,
        train_output_dst_dir     =train_output_dst_dir,
        validation_output_src_dir=validation_output_src_dir,
        validation_output_dst_dir=validation_output_dst_dir,
        test_src_output_filename =config_variables[TEST_SRC_CORPUS_FILENAME],
        test_dst_output_filename =config_variables[TEST_DST_CORPUS_FILENAME],
        vocab_src_output_filename=vocab_src_output_filename,
        vocab_tgt_output_filename=vocab_tgt_output_filename,
        raw_data_filename        =config_variables[RAW_DATA_FILENAME],
        columns_to_clean         =config_variables[RAW_DATA_COLUMNS_TO_CLEAN],
        split_column             =config_variables[RAW_DATA_SPLIT_COLUMN],
        train_column             =config_variables[RAW_DATA_TRAIN_COLUMN],
        validation_column        =config_variables[RAW_DATA_VALIDATION_COLUMN],
        test_column              =config_variables[RAW_DATA_TEST_COLUMN],
        persist_each             =persist_each,
    )