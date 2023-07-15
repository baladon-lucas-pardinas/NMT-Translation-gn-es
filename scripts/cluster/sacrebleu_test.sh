#!/bin/bash

PROJECT_PATH=/docker/home/marianmt
SCRIPT_PATH=${PROJECT_PATH}/scripts/validate
VALIDATION_PATH=${PROJECT_PATH}/artifacts/data/validation
SRC_PATH=${VALIDATION_PATH}/valid_es.txt.es
DST_PATH=${PROJECT_PATH}/logs/marian/split_decode_checkpoint_100-80.log
METRIC=sacrebleu

python3 ${SCRIPT_PATH}/score.py --score $METRIC \
    --reference_file $SRC_PATH \
    --translation_file $DST_PATH
