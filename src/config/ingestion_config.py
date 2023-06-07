import os

from src.config.project_config import get_project_config

class DataIngestionConfig:
    def __init__(self, artifacts_dir, persist_each=None):
        # type: (str, int) -> None
        self.persist_each = persist_each

        self.artifacts_dir = artifacts_dir
        data_dir                 = os.path.join(artifacts_dir, 'data')
        self.data_dir            = data_dir
        self.raw_data_dir        = os.path.join(data_dir, 'raw')
        self.corpora_dir         = os.path.join(data_dir, 'corpora')
        self.vocabulary_dir      = os.path.join(data_dir, 'vocabulary')
        self.train_data_dir      = os.path.join(data_dir, 'train')
        self.validation_data_dir = os.path.join(data_dir, 'validation')
        self.test_data_dir       = os.path.join(data_dir, 'test')
        self.raw_data_file_path  = os.path.join(self.raw_data_dir, 'jojajovai_all.csv')

        self.raw_data_columns_to_clean  = ['gn', 'es']
        self.raw_data_split_column      = 'split'
        self.raw_data_train_column      = 'train'
        self.raw_data_validation_column = 'dev'
        self.raw_data_test_column       = 'test'


    def __str__(self):
        # type: () -> str
        return "DataIngestionConfig(artifacts_dir={})".format(
            self.artifacts_dir
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_data_ingestion_config(persist_each=None):
    # type: (int) -> DataIngestionConfig
    project_config = get_project_config()
    
    return DataIngestionConfig(
        artifacts_dir=project_config.base_dir_artifacts,
        persist_each=persist_each,
    )