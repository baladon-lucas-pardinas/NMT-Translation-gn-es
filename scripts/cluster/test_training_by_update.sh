#!/bin/bash
MODEL_NAME=gn_es_test_by_update_${SLURM_JOB_NODELIST}
PROJECT_PATH=/docker/home/marianmt
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_gn.txt.gn ${PROJECT_PATH}/artifacts/data/train/train_es.txt.es"
SOURCE_VALID_SET="${PROJECT_PATH}/artifacts/data/validation/valid_gn.txt.gn"
TARGET_VALID_SET="${PROJECT_PATH}/artifacts/data/validation/valid_es.txt.es"
VALID_SETS="${SOURCE_VALID_SET} ${TARGET_VALID_SET}"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/gn_unique_tokens.txt.gn ${PROJECT_PATH}/artifacts/data/vocabulary/es_unique_tokens.txt.es"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/decoded-transformer-gn-es.txt
VALID_SCRIPT_PATH="${PROJECT_PATH}/scripts/validate/valid.sh"
chmod +x $VALID_SCRIPT_PATH
SCORE_TYPE="sacrebleu_corpus_bleu"
VALID_SCRIPT_ARGS="${PROJECT_PATH} ${TARGET_VALID_SET} ${SCORE_TYPE}"
EPOCHS=200
mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}
python3 ${PROJECT_PATH}/main.py --command-path  /marian/build \
    --validation-metrics "sacrebleu_corpus_bleu sacrebleu_corpus_chrf" \
    --train \
    --flags "--valid-script-path ${VALID_SCRIPT_PATH} --valid-script-args ${VALID_SCRIPT_ARGS} --early-stopping 3 --valid-freq 3000 --train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --vocabs ${VOCABS} --seed 1234 --type transformer --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/valid_log_${MODEL_NAME}.log --valid-sets ${VALID_SETS} --valid-metrics chrf translation --valid-translation-output ${TRANSLATION_OUTPUT} --quiet-translation --overwrite"
