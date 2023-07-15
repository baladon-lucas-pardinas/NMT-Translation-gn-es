#!/bin/bash
SRC="es"
TRG="gn"
MODEL_NAME=${SRC}_${TRG}_test_grid_july
PROJECT_PATH=/docker/home/marianmt
EPOCHS=200
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/train_${TRG}.txt.${TRG}"
VALID_SETS="${PROJECT_PATH}/artifacts/data/validation/valid_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/validation/valid_${TRG}.txt.${TRG}"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/${SRC}_unique_tokens.txt.${SRC} ${PROJECT_PATH}/artifacts/data/vocabulary/${TRG}_unique_tokens.txt.${TRG}"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/test.txt

mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}

python3 ${PROJECT_PATH}/main.py --command-path  /marian/build \
    --validate-each-epochs 10 \
    --validation-metrics "sacrebleu_corpus_bleu sacrebleu_corpus_chrf" \
    --hyperparameter-tuning \
    --tuning-grid-files "${PROJECT_PATH}/artifacts/parameters/grid_level1_test.json" \
    --train \
    --flags "--train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS} --seed 1234 --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/valid_log_${MODEL_NAME}.log --valid-sets ${VALID_SETS} --valid-metrics cross-entropy translation --valid-translation-output ${TRANSLATION_OUTPUT} --quiet-translation --overwrite"
