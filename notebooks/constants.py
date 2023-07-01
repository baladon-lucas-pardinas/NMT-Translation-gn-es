import os

DATA_FOLDER = 'data'
INSPECTED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'inspected')
RESULTS_DATA_FOLDER = os.path.join(DATA_FOLDER, 'results')
CORPORA_FOLDER = os.path.join(INSPECTED_DATA_FOLDER, 'corpora')
METRIC_EVALUATION_FOLDER = os.path.join(INSPECTED_DATA_FOLDER, 'metric_evaluation')
DECODED_OUTPUTS_FOLDER = os.path.join(INSPECTED_DATA_FOLDER, 'decoded_outputs')