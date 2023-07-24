import os
import csv
import datetime
import re
from typing import Callable
import collections

from src.utils import file_manager, parsing, arrays

import sacrebleu

LOG_METRICS_REGEX = r'''
    \[(\d{4})-(\d{2})-(\d{2})\s(\d{2}):(\d{2}):(\d{2})\]\s
    \[valid\]\s
    Ep\.\s(\S+)\s:\s
    Up\.\s(\S+)\s
    (?::\s(\S+)\s:\s(\S+))+\s
    :\s(.+)
'''
LOG_METRIX_COMPILED_REGEX = re.compile(LOG_METRICS_REGEX, re.VERBOSE)

"""
https://www.nltk.org/api/nltk.translate.bleu_score.html
hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures',
                'that', 'the', 'military', 'always', 'obeys', 'the',
                'commands', 'of', 'the', 'party']
reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures',
              'that', 'the', 'military', 'will', 'forever', 'heed',
              'Party', 'commands']
print(sentence_bleu([reference1], hypothesis1))
"""
# def calculate_sentence_bleu(references, translated):
#     # type: (list[str], list[str]) -> float
#     # Each reference is a list of references (in this case, of size 1).
#     bleu_scores = [sacrebleu.sentence_bleu(translated, reference).score for reference in references]
#     bleu_score = sum(bleu_scores) / len(bleu_scores)
#     return bleu_score


"""
https://github.com/mjpost/sacrebleu

refs = [ # First set of references
         ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
         # Second set of references
         ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
       ]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
sacrebleu.corpus_bleu(sys, refs)
BLEU = 48.53 82.4/50.0/45.5/37.5 (BP = 0.943 ratio = 0.944 hyp_len = 17 ref_len = 18)
"""
def calculate_sacrebleu_corpus_bleu(references, translated):
    # type: (list[str], list[str]) -> float
    bleu_score = sacrebleu.corpus_bleu(translated, references).score
    return bleu_score

def calculate_sacrebleu_corpus_chrf(references, translated):
    # type: (list[str], list[str]) -> float
    chrf_score = sacrebleu.corpus_chrf(translated, references).score
    return chrf_score

"""
https://github.com/mjpost/sacrebleu/blob/e22640/sacrebleu/compat.py#L66-L67
Disclaimer: computing BLEU on the sentence level is not its intended use,
BLEU is a corpus-level metric.
"""
def calculate_metric(references, translated, bleu_score_type='sacrebleu_corpus_bleu', tokenize=lambda x: x.split()):
    # type: (list[str], list[str], str, Callable[[str], list[str]]) -> float
    score_functions = {
        'sacrebleu_corpus_bleu': calculate_sacrebleu_corpus_bleu,
        'sacrebleu_corpus_chrf': calculate_sacrebleu_corpus_chrf,
    }

    references = arrays.reshape_1rest(references)
    
    # SACREBLEU does not need tokenization
    bleu_score = score_functions[bleu_score_type](references, translated)
    return bleu_score

def get_results_filename(file_name):
    # type: (str) -> str
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    return file_name

def get_results_from_translation_output(model_name, source, target, reference, translation_output, parameters, metrics=['sacrebleu_corpus_bleu']):
    # type: (str, str, str, str, str, dict, list[str]) -> list
    epoch             = parameters.get('after-epochs', [''])[0]
    date              = datetime.datetime.now()
    reference_lines   = file_manager.get_file_lines(reference)
    translation_lines = file_manager.get_file_lines(translation_output)
    score_rows        = []

    for score_type in metrics:
        bleu_score = calculate_metric(reference_lines, translation_lines, bleu_score_type=score_type)
        score_rows.append([date, model_name, source, target, score_type, bleu_score, epoch, parameters,])

    return score_rows

def get_results_from_logs(model_name, source, target, validation_log, parameters):
    # type: (str, str, str, str, dict) -> list
    scores = []

    with open(validation_log, 'r') as f:
        lines = f.readlines()

    parsed_lines = parsing.parse_line_groups(lines, LOG_METRIX_COMPILED_REGEX)
    for line in parsed_lines:
        year, month, day,\
        hour, minute, second,\
        epoch, update,\
        metric, metric_value,\
        _ = line

        date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        parameters['update'] = [update]
        score_row = [date, model_name, source, target, metric, metric_value, epoch, parameters,]
        scores.append(score_row)
    return scores
    
# Saves results from logs if translation output is not available
def save_results(
    file_name,
    model_dir,
    parameters,
    metrics=['sacrebleu_corpus_bleu'],
    validation_log=None,
    translation_output=None,
    reference=None
):
    # type: (str, str, dict, list[str], str, str, str) -> dict[str, list[float]]
    columns           = ['date', 'model_name', 'source', 'target', 'score_type', 'score', 'epoch', 'parameters']
    file_name         = get_results_filename(file_name)
    first_time_saving = not os.path.isfile(file_name)
    parameters        = parsing.deep_copy_flags(parameters)
    model_name        = os.path.basename(model_dir)
    source, target    = parameters.get('valid-sets', ['', ''])
    source, target    = [os.path.basename(path) for path in [source, target]]
    scores            = []

    if translation_output is not None and reference is not None:
        scores = get_results_from_translation_output(model_name, source, target, reference, translation_output, parameters, metrics=metrics)
    elif validation_log is not None:
        scores = get_results_from_logs(model_name, source, target, validation_log, parameters)
        
    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        if first_time_saving:
            writer.writerow(columns)
        for score_row in scores:
            writer.writerow(score_row)

    returned_scores = collections.defaultdict(list)
    for score in scores:
        returned_scores[score[4]].append(score[5]) # Metric name is in score[4], metric value is in score[5].
    return returned_scores