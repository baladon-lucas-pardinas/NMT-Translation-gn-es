#!/bin/bash
GPUS="0"
SRC="gn"
TRG="es"
TYPE="s2s"
EPOCHS=1000
FROM=0.5
TO=0.58
EARLY_STOPPING=7
RUN_ID="grid_${SRC}_${TRG}_${TYPE}_from${FROM}_to${TO}"
MODEL_NAME=lvl2_${RUN_ID}
PROJECT_PATH=/docker/home/marianmt
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/train_${TRG}.txt.${TRG}"
VALID_SETS="${PROJECT_PATH}/artifacts/data/validation/valid_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/validation/valid_${TRG}.txt.${TRG}"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/${SRC}_unique_tokens_gridlvl2.txt.${SRC}.spm ${PROJECT_PATH}/artifacts/data/vocabulary/${TRG}_unique_tokens_gridlvl2.txt.${TRG}.spm"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/decoded_${RUN_ID}.txt
GRIDS_PATH="${PROJECT_PATH}/artifacts/parameters/level2"
SPEED_FLAGS="--quiet-translation --overwrite --early-stopping ${EARLY_STOPPING} --fp16 --tied-embeddings-all --workspace 5500 --mini-batch-fit"

mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}

python3 ${PROJECT_PATH}/main.py --command-path /marian/build \
    --run-id ${RUN_ID} \
    --from-flags ${FROM} \
    --to-flags ${TO} \
    --validate-each-epochs 10 \
    --validation-metrics "sacrebleu_corpus_chrf sacrebleu_corpus_bleu" \
    --hyperparameter-tuning \
    --tuning-grid-files "${GRIDS_PATH}/level2_default.json ${GRIDS_PATH}/level2_depth2.json ${GRIDS_PATH}/level2_depth4.json ${GRIDS_PATH}/level2_depth6.json ${GRIDS_PATH}/level2_depth8.json ${GRIDS_PATH}/level2_dropout.json ${GRIDS_PATH}/level2_enc_cell.json ${GRIDS_PATH}/level2_label_smoothing.json ${GRIDS_PATH}/level2_skip_connections.json ${GRIDS_PATH}/level2_word_length.json" \
    --train \
    --flags "--type ${TYPE} --devices ${GPUS} --train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS} --seed 1234 --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/valid_log_${MODEL_NAME}.log --valid-sets ${VALID_SETS} --valid-metrics translation --valid-translation-output ${TRANSLATION_OUTPUT} ${SPEED_FLAGS} --dim-vocabs 16384 16384 --max-length-crop"
