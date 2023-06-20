import unittest
import os
import csv

from src.config.config import load_config_variables, \
    FLAG_SEPARATOR
from src.config import command_config as command
from src.utils import command_handler
from src.model import model
from src.components.evaluation import metrics
from src.utils import file_manager

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
        )

        command_length = len(self.command_dir) + 1
        injected_length = len(self.flags['echo-injection'][0])
        self.expected_output = command_handler.create_command(self.command_config)
        self.expected_output = self.expected_output[command_length:-injected_length].strip()
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
            config = self.command_config    
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

def main():
    unittest.main()

if __name__ == '__main__':
    main()