#!/bin/bash
GPUS="0"
SRC="gn"
TRG="es"
TYPE="transformer"
EPOCHS=1000
FROM=16
TO=18
EARLY_STOPPING=7
SEED="30883"
RUN_ID="grid_${SRC}_${TRG}_${TYPE}_from${FROM}_to${TO}_seed${SEED}"
MODEL_NAME=lvl3_${RUN_ID}
PROJECT_PATH=/docker/home/marianmt
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/train_${TRG}.txt.${TRG}"
VALID_SETS="${PROJECT_PATH}/artifacts/data/validation/valid_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/validation/valid_${TRG}.txt.${TRG}"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/${SRC}_unique_tokens_lvl3.txt.${SRC}.spm ${PROJECT_PATH}/artifacts/data/vocabulary/${TRG}_unique_tokens_lvl3.txt.${TRG}.spm"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/decoded_${RUN_ID}.txt
HYPERPARAMETER_SEARCH_SPACE_FILE="${PROJECT_PATH}/artifacts/parameters/level3/${TYPE}/random_config.json"
SPEED_FLAGS="--quiet-translation --overwrite --early-stopping ${EARLY_STOPPING} --fp16 --tied-embeddings-all --workspace 6500 --mini-batch-fit --maxi-batch 1000"
DEFAULT_FLAGS="--type ${TYPE} --max-length-crop --transformer-dropout 0.1 --layer-normalization --exponential-smoothing --label-smoothing 0.1"

mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}

python3 ${PROJECT_PATH}/main.py --command-path /marian/build \
    --run-id ${RUN_ID} \
    --from-flags ${FROM} \
    --to-flags ${TO} \
    --validate-each-epochs 10 \
    --validation-metrics "sacrebleu_corpus_chrf sacrebleu_corpus_bleu" \
    --hyperparameter-tuning \
    --tuning-grid-files ${HYPERPARAMETER_SEARCH_SPACE_FILE} \
    --tuning-strategy "randomsearch" \
    --max-iters 20 \
    --seed 26548 \
    --train \
    --flags "${DEFAULT_FLAGS} ${SPEED_FLAGS} --devices ${GPUS} --train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS} --seed ${SEED} --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/valid_log_${MODEL_NAME}.log --valid-sets ${VALID_SETS} --valid-metrics cross-entropy translation --valid-translation-output ${TRANSLATION_OUTPUT}"
