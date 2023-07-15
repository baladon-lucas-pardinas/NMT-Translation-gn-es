#!/bin/bash
MODEL_NAME=gn_es_scored_june_transformer
PROJECT_PATH=/docker/home/marianmt
EPOCHS=200
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_gn.txt.gn ${PROJECT_PATH}/artifacts/data/train/train_es.txt.es"
VALID_SETS="${PROJECT_PATH}/artifacts/data/validation/valid_gn.txt.gn ${PROJECT_PATH}/artifacts/data/validation/valid_es.txt.es"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/gn_unique_tokens.txt.gn ${PROJECT_PATH}/artifacts/data/vocabulary/es_unique_tokens.txt.es"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/decoded-transformer-gn-es.txt
mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}
python3 ${PROJECT_PATH}/main.py --command-path  /marian/build \
    --validate-each-epochs 10 \
    --validation_metrics "sacrebleu_corpus_bleu sacrebleu_corpus_chrf" \
    --train \
    --flags "--train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS} --seed 1234 --type transformer --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/valid_log_${MODEL_NAME}.log --valid-sets ${VALID_SETS} --valid-metrics cross-entropy translation --valid-translation-output ${TRANSLATION_OUTPUT} --quiet-translation --overwrite"
