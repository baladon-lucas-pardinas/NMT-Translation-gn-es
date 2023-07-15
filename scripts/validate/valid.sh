#!/bin/bash
DECODED_OUTPUT_FILE=$1
PROJECT_DIR=$2
REFERENCE_FILE=$3
SCORE_TYPE=$4
SCORE_SCRIPT_DIR=${PROJECT_DIR}/scripts/validate
python3 ${SCORE_SCRIPT_DIR}/score.py --reference_file ${REFERENCE_FILE} \
                                     --translation_file ${DECODED_OUTPUT_FILE} \
                                     --score ${SCORE_TYPE}