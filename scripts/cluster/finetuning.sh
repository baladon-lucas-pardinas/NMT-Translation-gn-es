#!/bin/bash
GPUS="0"
SRC="gn"
TRG="es"
TYPE="transformer"
EPOCHS=100
RUN_ID="default_finetuning"
MODEL_NAME=${TYPE}_${RUN_ID}
PROJECT_PATH=/docker/home/marianmt
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/augmented_data.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/augmented_data.txt.${TRG}"
VALID_SETS="${PROJECT_PATH}/artifacts/data/validation/valid_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/validation/valid_${TRG}.txt.${TRG}"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/full_augmented_vocab.${SRC} ${PROJECT_PATH}/artifacts/data/vocabulary/full_augmented_vocab.${TRG}"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/decoded_${RUN_ID}.txt
SEED=1234

export PYTHONPATH=${PROJECT_PATH}/libs
mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}

python3 ${PROJECT_PATH}/main.py --command-path /marian/build \
    --run-id ${RUN_ID} \
    --train \
    --finetuning \
    --finetuning-epochs 50 \
    --finetuning-augmented-sets "${PROJECT_PATH}/artifacts/data/train/full_augmented_data.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/full_augmented_data.txt.${TRG}" \
    --finetuning-full-sets "${PROJECT_PATH}/artifacts/data/train/augmented_data.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/augmented_data.txt.${TRG}" \
    --not-delete-model-after \
    --flags "--valid-metrics cross-entropy translation --valid-sets ${VALID_SETS} --valid-translation-output ${TRANSLATION_OUTPUT} --train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --early-stopping 1000 --valid-freq 50000000 --vocabs ${VOCABS} --seed ${SEED} --type ${TYPE} --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log --overwrite"
