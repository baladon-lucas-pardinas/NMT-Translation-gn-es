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
from src.utils import file_manager, command_handler

class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.config_variables = load_config_variables()
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.test_data_dir = os.path.join(self.current_dir, '..', 'data')
        self.test_valid_data_dir_src = os.path.join(self.test_data_dir, 'valid_gn.txt.gn')
        self.test_valid_data_dir_tgt = os.path.join(self.test_data_dir, 'valid_gn.txt.gn')

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
        self.expected_output = command_handler.create_command(self.command_config)
        self.expected_output = self.expected_output[command_length:-injected_length].strip()

        # Hyperparameter tuning
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
            tuning_grid_files=[self.grid1_filename, self.grid2_filename],
            tuning_params_files=[self.params1_filename, self.params2_filename],
            search_method='grid',
        )
        self.tuning_flags = command_handler.deep_copy_flags(self.flags)
        self.tuning_flags['valid-translation-output'] = [self.hyperparameter_validation_output_dir]
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

    def test_train_marian(self):
        # Create mock translation output
        first_output_filename = model.parse_output_filename(self.valid_translation_output, epoch=self.validate_each_epochs)
        second_output_filename = model.parse_output_filename(self.valid_translation_output, epoch=int(self.validate_each_epochs) * 2)
        file_manager.save_copy(self.test_valid_data_dir_tgt, first_output_filename)
        file_manager.save_copy(self.test_valid_data_dir_tgt, second_output_filename)

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
                for row in reader:
                    self.assertEqual(row['score'].split('.')[0], '100')

            # Should exist checkpoint
            first_checkpoint_filename = model.rename_checkpoint(self.model_dir, self.validate_each_epochs)
            second_checkpoint_filename = model.rename_checkpoint(self.model_dir, int(self.validate_each_epochs) * 2)
            self.assertTrue(os.path.exists(first_checkpoint_filename))
            self.assertTrue(os.path.exists(second_checkpoint_filename))

        except AssertionError as e:
            self.fail("Failed with assertion {}".format(e.with_traceback()))
        finally:
            os.remove(self.model_dir)
            os.remove(self.command_output_dir)
            os.remove(self.csv_file_name)
            os.remove(first_output_filename)
            os.remove(second_output_filename)
            os.remove(first_checkpoint_filename)
            os.remove(second_checkpoint_filename)

    def test_hyperparameter_tuning(self):
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
            )

            # Metric csv should have n rows
            n_combinations = (len(list(itertools.product(*self.grid1.values()))) + \
                                len(list(itertools.product(*self.grid2.values()))) + \
                                len(hyperparameter_tuning_config.tuning_params_files)) * \
                                len(self.validation_metrics)
            
            with open(self.csv_file_name, 'r') as f:
                reader = csv.DictReader(f)
                self.assertEqual(len(list(reader)), n_combinations)

        except AssertionError as e:
            self.fail("Failed with assertion {}".format(e.with_traceback()))
        finally:
            print()
            os.remove(self.grid1_filename)
            os.remove(self.grid2_filename)
            os.remove(self.params1_filename)
            os.remove(self.params2_filename)
            os.remove(self.hyperparameter_validation_output_dir)

def main():
    unittest.main()

if __name__ == '__main__':
    main()