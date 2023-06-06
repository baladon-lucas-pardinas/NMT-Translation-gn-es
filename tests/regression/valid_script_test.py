import unittest
import os
import subprocess

SCRIPT_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    'scripts',
    'validate',
)

class TestValidationScript(unittest.TestCase):
    def setUp(self) -> None:
        self.example_reference = ['This is an example reference sentence.', '', 'Hello']
        self.example_translation = ['This is an example translation sentence.', 'My name is Marian, but people call me MarianMT', '']
        pass

    def test_validation_script(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(current_dir, '..', 'data')
        reference_file = os.path.join(test_data_dir, 'reference.txt')
        translation_file = os.path.join(test_data_dir, 'translation.txt')

        with open(reference_file, 'w', encoding='utf-8') as f:
            for line in self.example_reference:
                f.write(line + '\n')

        with open(translation_file, 'w', encoding='utf-8') as f:
            for line in self.example_translation:
                f.write(line + '\n')

        process_result = subprocess.run([
            'python', os.path.join(SCRIPT_DIR, 'score.py'),
            '--reference_file', reference_file,
            '--translation_file', translation_file,
            '--score', 'sacrebleu',
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        os.remove(reference_file)
        os.remove(translation_file)

        output = process_result.stdout.decode('utf-8')
        error = process_result.stderr.decode('utf-8')

        if process_result.stderr != b'':
            print(output)
            print(error)
            self.fail('Error while running the validation script.')

        self.assertNotEqual(process_result.stdout, b'')
        try:
            float(output)
        except ValueError:
            print(output)
            self.fail('Validation script did not return a number.')
        self.assertEqual(process_result.returncode, 0)

def main():
    unittest.main()

if __name__ == '__main__':
    main()