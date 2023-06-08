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
        self.example_translation = ['This is an example translation sentence.', 'My name is Marian, but people call me MarianMT', '', "Ko'ágã ojehecha hikuái karu guasu 14:00 a la Samaniego oúvape pe dato 42 pasajero elecciones de transporte oúva temimbo’e rupi ha upéi oñemotenonde 1 de abril oúvape pero oiko jave 1 de prensa oikóva oúva ha upéicha ojapóvo ha oñemotenonde upe 1 de prensa"]
        self.example_reference = ['This is an example reference sentence.', '', 'Hello', "Ojehechava'erã orrenunisava'erã umi concejal Junta Municipal-pe, oñemoherakuãgui ganador elecciones ágã 30 de mayo, ikatu haguã oñepyrû hikuái 1 de julio."]

        # Expected scores calculated using https://huggingface.co/spaces/evaluate-metric/sacrebleu
        self.expected_scores = {
            'sacrebleu': [48.89230, 0.0, 0.0, 1.99869780],
            'bleu': [48.89230, 0.0, 0.0, 1.99869780],
            'chrf': [0.0, 0.0, 0.0, 0.0],
        }
        self.expected_mean_score = {
            metric: sum(self.expected_scores[metric]) / len(self.expected_scores[metric]) \
                for metric in self.expected_scores
        }
        self.precision_epsilon = 1e-4
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

        for metric in self.expected_scores:
            process_result = subprocess.run([
                'python', os.path.join(SCRIPT_DIR, 'score.py'),
                '--reference_file', reference_file,
                '--translation_file', translation_file,
                '--score', metric,
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            os.remove(reference_file)
            os.remove(translation_file)

            output = process_result.stdout.decode('utf-8')
            error = process_result.stderr.decode('utf-8')
            if process_result.stderr != b'':
                print('Metric:', metric)
                print('Output:', output)
                print('Error:', error)
                self.fail('Error while running the validation script.')

            self.assertNotEqual(process_result.stdout, b'')

            try:
                float(output)
            except ValueError:
                print('Metric:', metric)
                print('Output:', output)
                self.fail('Validation script did not return a number.')

            self.assertEqual(process_result.returncode, 0)
            self.assertAlmostEqual(float(output), self.expected_mean_score[metric], delta=self.precision_epsilon)

def main():
    unittest.main()

if __name__ == '__main__':
    main()