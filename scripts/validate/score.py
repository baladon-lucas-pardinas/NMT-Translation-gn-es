import argparse
import collections

import sacrebleu
import nltk

def bleu_score(hypothesis, reference):
    return nltk.translate.bleu_score.corpus_bleu(hypothesis, reference)

def sacrebleu_score(hypothesis, reference):
    return sacrebleu.corpus_bleu(hypothesis, [reference])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_file', type=str, required=True)
    parser.add_argument('--score', type=str, default='bleu')
    args = parser.parse_args()
    return vars(args)

import sys
import sacrebleu

def calculate_bleu(reference_file, translation_file):
    # Load the reference translations
    with open(reference_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]

    # Load the system translations
    with open(translation_file, 'r', encoding='utf-8') as f:
        translations = [line.strip() for line in f]

    # Calculate the BLEU score
    bleu = sacrebleu.corpus_bleu(translations, [references])
    return bleu.score

def main(reference_file, translation_file, score='bleu'):
    score_function = {
        'bleu': bleu_score,
        'sacrebleu': sacrebleu_score,
    }

    reference_lines = None
    translation_lines = None

    with open(reference_file, 'r', encoding='utf-8') as f:
        reference_lines = f.readlines()

    with open(translation_file, 'r', encoding='utf-8') as f:
        translation_lines = f.readlines()

    score = score_function[selected_score](translation_lines, reference_lines)
    print(score)

if __name__ == '__main__':
    args = parse_args()
    reference_file = args['reference_file']
    selected_score = args['score']
    translation_file = sys.stdin
    main(reference_file, translation_file, selected_score)



        


if __name__ == '__main__':
    reference_file = sys.argv[1]
    translation_file = sys.stdin

    bleu_score = calculate_bleu(reference_file, translation_file)
    print(bleu_score)