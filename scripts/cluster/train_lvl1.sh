#!/bin/bash
TYPE="transformer"
SRC="es"
TRG="gn"
RUN_ID="first_10_${SRC}_${TRG}_lvl1_${TYPE}"
MODEL_NAME=${RUN_ID}
PROJECT_PATH=/docker/home/marianmt
EPOCHS=100
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/train_${TRG}.txt.${TRG}"
VALID_SETS="${PROJECT_PATH}/artifacts/data/validation/valid_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/validation/valid_${TRG}.txt.${TRG}"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/${SRC}_unique_tokens.txt.${SRC} ${PROJECT_PATH}/artifacts/data/vocabulary/${TRG}_unique_tokens.txt.${TRG}"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/decoded_lvl1_${RUN_ID}.txt
mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}
python3 ${PROJECT_PATH}/main.py --command-path  /marian/build \
    --validate-each-epochs 2 \
    --validation-metrics "sacrebleu_corpus_chrf sacrebleu_corpus_bleu" \
    --train \
    --run-id ${RUN_ID} \
    --flags "--early-stopping 3 --devices 0 1 --train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS} --seed 1234 --type ${TYPE} --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/valid_log_${MODEL_NAME}.log --valid-sets ${VALID_SETS} --valid-metrics cross-entropy translation --valid-translation-output ${TRANSLATION_OUTPUT} --quiet-translation --overwrite"
