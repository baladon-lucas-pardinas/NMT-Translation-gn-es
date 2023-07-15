#!/bin/bash
MODEL_NAME=checkpoints_es_gn
PROJECT_PATH=/docker/home/marianmt
TRAIN_SET_SRC=train_es.txt.es
TRAIN_SET_TRG=train_gn.txt.gn
VAL_SET_SRC=valid_es.txt.es
VAL_SET_TRG=valid_gn.txt.gn
VOCAB_SRC=es_unique_tokens.txt.es
VOCAB_TRG=gn_unique_tokens.txt.gn
LOG_NAME=$first_${MODEL_NAME}
python3 ${PROJECT_PATH}/main.py --command-path  /marian/build \
    --save-each-epochs 10 \
    --train \
    --flags "--train-sets ${PROJECT_PATH}/artifacts/data/train/${TRAIN_SET_SRC} ${PROJECT_PATH}/artifacts/data/train/${TRAIN_SET_TRG} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs 150 --save-freq 5000000 --vocabs ${PROJECT_PATH}/artifacts/data/vocabulary/${VOCAB_SRC} ${PROJECT_PATH}/artifacts/data/vocabulary/${VOCAB_TRG} --seed 1234 --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/model_${LOG_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/${LOG_NAME}_${MODEL_NAME}.log --valid-sets ${PROJECT_PATH}/artifacts/data/validation/${VAL_SET_SRC} ${PROJECT_PATH}/artifacts/data/validation/${VAL_SET_TRG} --valid-metrics bleu chrf cross-entropy perplexity valid-script --valid-script-path ${PROJECT_PATH}/scripts/validate.sh"
