GPUS="0"
SRC="es"
TRG="gn"
TYPE="transformer"
EPOCHS=200
FROM=0
TO=1
RUN_ID="default_finetuning_${FINETUNING_EPOCHS}_${SRC}_${TRG}_${TYPE}"
MODEL_NAME=${TYPE}_${RUN_ID}
PROJECT_PATH=/docker/home/marianmt
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/train_${TRG}.txt.${TRG}"
VALID_SETS="${PROJECT_PATH}/artifacts/data/validation/valid_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/validation/valid_${TRG}.txt.${TRG}"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/pretrain_test_vocab.${SRC}.spm ${PROJECT_PATH}/artifacts/data/vocabulary/pretrain_test_vocab.${TRG}.spm"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/decoded_${RUN_ID}.txt
HYPERPARAMETER_GRID_FILES="${PROJECT_PATH}/artifacts/parameters/finetuning/level1/epochs_test.json"
SEED=7882

export PYTHONPATH=${PROJECT_PATH}/libs
mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}

python3 ${PROJECT_PATH}/main.py --command-path /marian/build \
    --run-id ${RUN_ID} \
    --train \
    --validation-metrics "sacrebleu_corpus_bleu sacrebleu_corpus_chrf" \
    --validate-each-epochs 10 \
    --hyperparameter-tuning \
    --from-flags ${FROM} \
    --to-flags ${TO} \
    --tuning-strategy "gridsearch" \
    --tuning-grid-files ${HYPERPARAMETER_GRID_FILES} \
    --finetuning-cache-template-dir "${PROJECT_PATH}/artifacts/models/pretraining_${TYPE}_${SRC}_${TRG}_epoch{}"\
    --finetuning \
    --finetuning-augmented-sets "${PROJECT_PATH}/artifacts/data/train/augmented_data.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/augmented_data.txt.${TRG}" \
    --finetuning-full-sets "${PROJECT_PATH}/artifacts/data/train/full_augmented_data.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/full_augmented_data.txt.${TRG}" \
    --flags "--devices ${GPUS} --tempdir ${PROJECT_PATH}/libs --valid-metrics cross-entropy translation --valid-sets ${VALID_SETS} --valid-translation-output ${TRANSLATION_OUTPUT} --train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS} --seed ${SEED} --type ${TYPE} --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log --overwrite --dim-vocabs 16384 16384 --early-stopping 1000"


