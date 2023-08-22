import unittest
import os
import itertools
import csv
import json

from src.config.config import load_config_variables, \
    FLAG_SEPARATOR
from src.config import command_config as command, hyperparameter_tuning_config as hyperparameter_tuning
from src.model import model
from src.components.evaluation import metrics
from src.pipelines import train_pipeline
from src.utils import file_manager, parsing

class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.config_variables = load_config_variables()
        self.current_dir             = os.path.dirname(os.path.realpath(__file__))
        self.test_data_dir           = os.path.join(self.current_dir, '..', 'data')
        self.test_valid_data_dir_src = os.path.join(self.test_data_dir, 'valid_gn.txt.gn')
        self.test_valid_data_dir_tgt = os.path.join(self.test_data_dir, 'valid_gn.txt.gn')

        ############# Train Test #############
        # Command
        self.command_path = ''
        self.command_name = 'echo'
        self.command_dir = os.path.join(self.command_path, self.command_name)
        self.command_output_dir = os.path.join(self.test_data_dir, 'output.txt')

        # Checkpoints
        self.validate_each_epochs = '5'
        self.after_epochs = '10'
        self.save_checkpoints = True

        # Translation output
        self.model_dir = os.path.join(self.test_data_dir, 'model.npz')
        self.valid_translation_output = os.path.join(self.test_data_dir, 'test_valid_translation_output{E}.txt')
        self.valid_sets = ''
        self.validation_metrics = ['sacrebleu_corpus_bleu', 'sacrebleu_corpus_chrf']
        self.csv_file_name = metrics.get_results_filename(os.path.join(self.test_data_dir, self.command_name))

        # Config
        self.flags = {
            'valid-sets': [self.test_valid_data_dir_src, self.test_valid_data_dir_tgt],
            'model': [self.model_dir],
            'after-epochs': [self.after_epochs],
            'valid-translation-output': [self.valid_translation_output],
            'valid-metrics': ['translation'], # The model needs this to validate
            'echo-injection': [f' > {self.command_output_dir}'],
        }

        self.command_config = command.CommandConfig(
            command_name=self.command_name,
            command_path=self.command_path,
            flags=self.flags,
            flag_separator=self.config_variables[FLAG_SEPARATOR],
            validate_each_epochs=self.validate_each_epochs,
            validation_metrics=self.validation_metrics,
            save_checkpoints=self.save_checkpoints,
            base_dir_evaluation=self.test_data_dir,
            not_delete_model_after=True,
        )

        command_length = len(self.command_dir) + 1
        injected_length = len(self.flags['echo-injection'][0])
        self.expected_output = parsing.create_command(self.command_config)
        self.expected_output = self.expected_output[command_length:-injected_length].strip()

        ############# Hyperparameter tuning Test (Grid Search) #############
        self.log_metric_count = 10
        self.valid_log_dir = os.path.join(self.test_data_dir, 'test_log.log')
        self.hyperparameter_validation_output_dir = os.path.join(self.test_data_dir, 'hyperparameter_validation_output.txt')
        self.grid1 = {"k11": ['v11', 'v12'],"k12": ['v13', 'v14']}
        self.grid2 = {"k21": ['v22'],"k21": ['v22']}
        self.params1 = {"k31": 'v31',"k32": 'v32'}
        self.params2 = {"k41": 'v41',"k42": 'v42'}
        self.grid1_filename = os.path.join(self.test_data_dir, 'grid1.json')
        self.grid2_filename = os.path.join(self.test_data_dir, 'grid2.json')
        self.params1_filename = os.path.join(self.test_data_dir, 'params1.json')
        self.params2_filename = os.path.join(self.test_data_dir, 'params2.json')
        self.tuning_config = hyperparameter_tuning.HyperparameterTuningConfig(
            run_id='test',
            tuning_grid_files=[self.grid1_filename, self.grid2_filename],
            tuning_params_files=[self.params1_filename, self.params2_filename],
            tuning_strategy='gridsearch',
        )
        self.tuning_flags = parsing.deep_copy_flags(self.flags)
        self.tuning_flags['valid-translation-output'] = [self.hyperparameter_validation_output_dir]
        self.tuning_flags['valid-log'] = [self.valid_log_dir]
        self.tuning_command_config = command.CommandConfig(
            command_name=self.command_name,
            command_path=self.command_path,
            flags=self.tuning_flags,
            flag_separator=self.config_variables[FLAG_SEPARATOR],
            validation_metrics=self.validation_metrics,
            base_dir_evaluation=self.test_data_dir,
            not_delete_model_after=True,
        )
        pass

        ############# Hyperparameter tuning Test (Random Search) #############
        self.random_search_config = self.tuning_config.copy(deep=True)
        self.random_search_max_iters = 10
        self.random_search_seed = 42
        self.random_search_config.__setattr__('tuning_strategy', 'randomsearch')
        self.random_search_config.__setattr__('seed', self.random_search_seed)
        self.random_search_config.__setattr__('max_iters', self.random_search_max_iters)
        self.random_config_file = os.path.join(self.test_data_dir, 'random_config.json')
        self.random_search_config.__setattr__('tuning_grid_files', [self.random_config_file])
        self.random_search_config.__setattr__('tuning_params_files', [])
        self.random_search_command_config = self.tuning_command_config.copy(deep=True)
        self.random_search_command_config.__setattr__('validate_each_epochs', self.validate_each_epochs)

        ############# Early stopping Test #############
        self.early_stopping_config = self.command_config.copy(deep=True)
        self.early_stopping_config.flags['early-stopping'] = ['2']

    def clean_files(self, files):
        # type: (list[str]) -> None
        for file in files:
            if os.path.exists(file):
                os.remove(file)

    def test_train_marian(self):
        # Create mock translation output
        first_output_filename = model.parse_output_filename(self.valid_translation_output, epoch=self.validate_each_epochs)
        second_output_filename = model.parse_output_filename(self.valid_translation_output, epoch=int(self.validate_each_epochs) * 2)
        file_manager.save_copy(self.test_valid_data_dir_tgt, first_output_filename)
        file_manager.save_copy(self.test_valid_data_dir_tgt, second_output_filename)
        first_checkpoint_filename = model.rename_checkpoint(self.model_dir, self.validate_each_epochs)
        second_checkpoint_filename = model.rename_checkpoint(self.model_dir, int(self.validate_each_epochs) * 2)
        files_to_clean = [self.csv_file_name, self.valid_translation_output, first_output_filename, 
                          second_output_filename, first_checkpoint_filename, second_checkpoint_filename, 
                          self.command_output_dir, self.model_dir]

        # Create mock checkpoint
        with open(self.model_dir, 'w') as f:
            f.write('')

        try:
            config = self.command_config.copy(deep=True)
            model.train(config)

            # Should run command (should echo something)
            with open(self.command_output_dir, 'r') as f:
                output = f.read().strip()
            self.assertEqual(output, self.expected_output)

            # Should exist metric csv
            self.assertTrue(os.path.exists(self.csv_file_name))

            # Metric csv score should be 100, as files are equal
            with open(self.csv_file_name, 'r') as f:
                reader = csv.DictReader(f)
                has_model = False
                for row in reader:
                    if row['model_name'] != 'model.npz':
                        continue
                    has_model = True
                    self.assertEqual(row['score'].split('.')[0], '100')
                self.assertTrue(has_model)

            # Should exist checkpoint
            self.assertTrue(os.path.exists(first_checkpoint_filename))
            self.assertTrue(os.path.exists(second_checkpoint_filename))

            self.clean_files(files_to_clean)
        except AssertionError as e:
            self.clean_files(files_to_clean)
            self.fail("Failed with assertion {}".format(e.with_traceback()))

    def test_grid_search(self):
        files_to_clean = [self.csv_file_name, self.grid1_filename, self.grid2_filename, self.params1_filename, self.params2_filename, self.hyperparameter_validation_output_dir]

        try:
            # Add hyperparameter tuning values
            command_config = self.tuning_command_config
            hyperparameter_tuning_config = self.tuning_config

            # Create translation output mock file
            file_manager.save_copy(self.test_valid_data_dir_tgt, self.hyperparameter_validation_output_dir)

            # Create mock grid files
            filenames = [self.grid1_filename, self.grid2_filename, self.params1_filename, self.params2_filename]
            grids = [self.grid1, self.grid2, self.params1, self.params2]
            for filename, grid in zip(filenames, grids):
                with open(filename, 'w') as f:
                    json.dump(grid, f)
                
            # Train with hyperparameter tuning
            train_pipeline.train(
                command_config=command_config,
                data_ingestion_config=None,
                data_transformation_config=None,
                hyperparameter_tuning_config=hyperparameter_tuning_config,
                finetuning_config=None,
            )

            # Metric csv should have n rows
            n_combinations = self.log_metric_count * \
                            (len(list(itertools.product(*self.grid1.values()))) + \
                             len(list(itertools.product(*self.grid2.values()))) + \
                             len(hyperparameter_tuning_config.tuning_params_files))
            
            with open(self.csv_file_name, 'r') as f:
                reader = csv.DictReader(f)
                self.assertEqual(len(list(reader)), n_combinations)

            self.clean_files(files_to_clean)
        except AssertionError as e:
            self.clean_files(files_to_clean)
            self.fail("Failed with assertion {}".format(e.with_traceback()))

    def test_random_search(self):
        try:
            command_config = self.random_search_command_config
            hyperparameter_tuning_config = self.random_search_config
            files_to_clean = [self.hyperparameter_validation_output_dir, self.csv_file_name]

            # Create translation output mock file
            file_manager.save_copy(self.test_valid_data_dir_tgt, self.hyperparameter_validation_output_dir)

            experiments_n = 2
            hyperparameter_configs = [hyperparameter_tuning_config.copy(deep=True) for _ in range(experiments_n)]
            command_configs = [command_config.copy(deep=True) for _ in range(experiments_n)]
            for i in range(experiments_n):                    
                hyperparameter_tuning_config = hyperparameter_configs[i]
                command_config = command_configs[i]
                train_pipeline.train(
                    data_ingestion_config=None,
                    data_transformation_config=None,
                    hyperparameter_tuning_config=hyperparameter_tuning_config,
                    command_config=command_config,
                    finetuning_config=None
                )

                # Metric csv should have number_of_rows = max_iters * metrics_n * epochs_n * iterations_i+1
                n = self.random_search_max_iters * len(self.validation_metrics) * (int(self.after_epochs) / int(self.validate_each_epochs)) * (i+1)
                with open(self.csv_file_name, 'r') as f:
                    reader = csv.DictReader(f)
                    reader_list = list(reader)
                    metric_results = [row['parameters'] for row in reader_list]
                    self.assertEqual(len(reader_list), n)

            #Check reproductibility of both results
            metric_results_middle = len(str(metric_results)) // 2
            metrics1_str = str(metric_results)[:metric_results_middle]
            metrics2_str = str(metric_results)[metric_results_middle:]

            # Remove list brackets
            metrics1_str = metrics1_str[1:-1]
            metrics2_str = metrics2_str[1:-1]

            # Test that both result are equal (reproductibility)
            self.assertEqual(metrics1_str, metrics2_str)

            self.clean_files(files_to_clean)
        except AssertionError as e:
            self.clean_files(files_to_clean)
            self.fail("Failed with assertion {}".format(e.with_traceback()))

    def test_early_stopping(self):
        try:
            # Add early stopping values
            command_config = self.early_stopping_config

            # Create mock translation output
            first_output_filename = model.parse_output_filename(self.valid_translation_output, epoch=self.validate_each_epochs)
            second_output_filename = model.parse_output_filename(self.valid_translation_output, epoch=int(self.validate_each_epochs) * 2)
            third_output_filename = model.parse_output_filename(self.valid_translation_output, epoch=int(self.validate_each_epochs) * 3)
            file_manager.save_copy(self.test_valid_data_dir_tgt, first_output_filename)
            file_manager.save_copy(self.test_valid_data_dir_tgt, second_output_filename)
            file_manager.save_copy(self.test_valid_data_dir_tgt, third_output_filename)
            files_to_clean = [self.csv_file_name, self.valid_translation_output, first_output_filename, second_output_filename, third_output_filename]

            # Create mock checkpoint
            with open(self.model_dir, 'w') as f:
                f.write('')

            # Train with early stopping
            model.train(command_config)

            # Should exist only len(metrics)*early_stopping rows in csv
            with open(self.csv_file_name, 'r') as f:
                rows = list(csv.reader(f))
                rows = rows[1:] # Columns
                rows = [row for row in rows if row] # Remove empty rows
                n_rows = len(rows)
                n_metrics = len(self.early_stopping_config.validation_metrics)
                early_stopping = int(self.early_stopping_config.flags['early-stopping'][0])
                self.assertEqual(n_rows, n_metrics * early_stopping)

            self.clean_files(files_to_clean)
        except AssertionError as e:
            self.clean_files(files_to_clean)
            self.fail("Failed with assertion {}".format(e.with_traceback()))

def main():
    unittest.main()

if __name__ == '__main__':
    main()