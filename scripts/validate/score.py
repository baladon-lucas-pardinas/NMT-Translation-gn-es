import argparse
import sys
import os

# This is a trick (not the most clean) to import from the root folder
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(current_dir, '..', '..'))

from src.components.evaluation import metrics
from src.utils import wrappers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_file', type=str, required=True)
    parser.add_argument('--translation_file', type=str, required=True)
    parser.add_argument('--score', type=str, default='sacrebleu_corpus_bleu')
    args = parser.parse_args()
    return vars(args)

@wrappers.warning_filter(logger='spacy') # This is done to avoid warnings from spacy
def main(reference_file, translation_file, metric='sacrebleu_corpus_bleu'):

    with open(reference_file, 'r', encoding='utf-8') as f:
        reference_lines = f.readlines()
    with open(translation_file, 'r', encoding='utf-8') as f:
        translation_lines = f.readlines()

    reference_lines = [s.strip() for s in reference_lines]
    translation_lines = [s.strip() for s in translation_lines]

    score = metrics.calculate_metric(reference_lines, translation_lines, metric)
    rounded_result = '{:g}'.format(float('{:.6g}'.format(score)))
    print(rounded_result)

if __name__ == '__main__':
    args = parse_args()
    reference_file   = args['reference_file']
    translation_file = args['translation_file']
    selected_metric   = args['score']
    main(reference_file, translation_file, selected_metric)