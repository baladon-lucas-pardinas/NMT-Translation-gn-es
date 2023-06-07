import argparse
import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(current_dir, '..', '..'))

from src.components.evaluation import metrics
from src.components.processing import tokenization
from src.logger import logging
from src.utils import wrappers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_file', type=str, required=True)
    parser.add_argument('--translation_file', type=str, required=True)
    parser.add_argument('--score', type=str, default='sacrebleu')
    parser.add_argument('--significant_figures', type=int, default=4)
    args = parser.parse_args()
    return vars(args)

@wrappers.warning_filter
def main(reference_file, translation_file, score='sacrebleu'):
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference_lines = f.readlines()
    with open(translation_file, 'r', encoding='utf-8') as f:
        translation_lines = f.readlines()

    if score != 'sacrebleu':
        tokenizer = tokenization.get_tokenizer()            
        reference_lines   = [tokenization.tokenize(tokenizer, line) for line in reference_lines]
        translation_lines = [tokenization.tokenize(tokenizer, line) for line in translation_lines]

    scores = []
    for reference_line, translation_line in zip(reference_lines, translation_lines):
        metric = metrics.calculate_metric(reference_line, translation_line, score)
        scores.append(metric)
        
    numeric_result = sum(scores) / len(scores)
    rounded_result = '{:g}'.format(float('{:.6g}'.format(numeric_result)))
    print(rounded_result)

if __name__ == '__main__':
    args = parse_args()
    reference_file = args['reference_file']
    translation_file = args['translation_file']
    selected_score = args['score']
    significant_figures = args['significant_figures']
    main(reference_file, translation_file, selected_score, significant_figures)
