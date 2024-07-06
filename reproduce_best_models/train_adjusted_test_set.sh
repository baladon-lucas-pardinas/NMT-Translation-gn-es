GPUS="0"
SRC="es"
TRG="gn"
TYPE="s2s"
EPOCHS=1000
FROM=0
TO=1
EARLY_STOPPING=1000
RUN_ID="adjusted_${SRC}_${TRG}_${TYPE}"
MODEL_NAME=${TYPE}_${RUN_ID}
PROJECT_PATH=${1:-"/app"}
SHARED_PATH="/shared/reproduce_best_models"
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train.${SRC} ${PROJECT_PATH}/artifacts/data/train/train.${TRG}"
VALID_SETS="${PROJECT_PATH}/artifacts/data/test/test.${SRC} ${PROJECT_PATH}/artifacts/data/test/test.${SRC}"
VOCABS="${SHARED_PATH}/pretrain_test_vocab.${SRC}.spm ${SHARED_PATH}/pretrain_test_vocab.${TRG}.spm"
TRANSLATION_OUTPUT=${SHARED_PATH}/decoded_${RUN_ID}.txt
HYPERPARAMETER_GRID_FILES="${PROJECT_PATH}/reproduce_best_models/best_${TYPE}_${SRC}_${TRG}_config.json"
# --fp16 --workspace 6500 --mini-batch-fit --maxi-batch 1000 --workspace 6500
SPEED_FLAGS="--early-stopping ${EARLY_STOPPING} --overwrite --tied-embeddings-all --workspace 6500"
DEFAULT_FLAGS="--type ${TYPE} --max-length-crop --layer-normalization --exponential-smoothing --label-smoothing 0.1"
MODEL_DIR=${SHARED_PATH}/model_${MODEL_NAME}
MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}.npz

mkdir -p $MODEL_DIR
touch ${MODEL_DIR}/test.txt

python3 ${PROJECT_PATH}/main.py --command-path /marian/build \
    --run-id ${RUN_ID} \
    --not-delete-model-after \
    --train \
    --validation-metrics "sacrebleu_corpus_bleu sacrebleu_corpus_chrf" \
    --validate-each-epochs 100000 \
    --hyperparameter-tuning \
    --from-flags ${FROM} \
    --to-flags ${TO} \
    --tuning-grid-files ${HYPERPARAMETER_GRID_FILES} \
    --finetuning-cache-template-dir "${PROJECT_PATH}/artifacts/models/pretraining_merged_${TYPE}_${SRC}_${TRG}_epoch{}"\
    --finetuning \
    --finetuning-augmented-sets "${PROJECT_PATH}/artifacts/data/train/merged_corpora_without_train.${SRC} ${PROJECT_PATH}/artifacts/data/train/merged_corpora_without_train.${TRG}" \
    --finetuning-full-sets "${PROJECT_PATH}/artifacts/data/train/full_augmented_data.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/full_augmented_data.txt.${TRG}" \
    --flags "${SPEED_FLAGS} ${DEFAULT_FLAGS} --cpu-threads 12 --tempdir ${PROJECT_PATH}/libs --valid-metrics cross-entropy translation --valid-sets ${VALID_SETS} --valid-translation-output ${TRANSLATION_OUTPUT} --train-sets ${TRAIN_SETS} --model ${MODEL_PATH} --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS} --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log"