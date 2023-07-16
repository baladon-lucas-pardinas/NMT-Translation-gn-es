#!/bin/bash
PROJECT_DIR=$1
REFERENCE_FILE=$2
SCORE_TYPE=$3
DECODED_OUTPUT_FILE=$4
SCORE_SCRIPT_DIR=${PROJECT_DIR}/scripts/validate
python3 ${SCORE_SCRIPT_DIR}/score.py --reference_file ${REFERENCE_FILE} \
                                     --translation_file ${DECODED_OUTPUT_FILE} \
                                     --score ${SCORE_TYPE}