GPUS="0"
SRC="es"
TRG="gn"
TYPE="transformer"
EPOCHS=1000
FROM=0
TO=1
EARLY_STOPPING=1000
RUN_ID="adjusted_${SRC}_${TRG}_${TYPE}"
MODEL_NAME=${TYPE}_${RUN_ID}
PROJECT_PATH=/docker/home/marianmt
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/train_${TRG}.txt.${TRG}"
VALID_SETS="${PROJECT_PATH}/artifacts/data/test/test_${SRC}.txt.${SRC} ${PROJECT_PATH}/artifacts/data/test/test_${TRG}.txt.${TRG}"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/pretrain_test_vocab.${SRC}.spm ${PROJECT_PATH}/artifacts/data/vocabulary/pretrain_test_vocab.${TRG}.spm"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/decoded_${RUN_ID}.txt
HYPERPARAMETER_GRID_FILES="${PROJECT_PATH}/artifacts/parameters/finetuning/tuned/${TYPE}_${SRC}_${TRG}.json"
SPEED_FLAGS="--early-stopping ${EARLY_STOPPING} --quiet-translation --overwrite --fp16 --tied-embeddings-all --workspace 6500 --mini-batch-fit --maxi-batch 1000"
DEFAULT_FLAGS="--type ${TYPE} --max-length-crop --layer-normalization --exponential-smoothing --label-smoothing 0.1"

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
    --finetuning-cache-template-dir "${PROJECT_PATH}/artifacts/models/pretraining_merged_${TYPE}_${SRC}_${TRG}_epoch{}"\
    --finetuning \
    --finetuning-augmented-sets "${PROJECT_PATH}/artifacts/data/train/merged_corpora_without_train.${SRC} ${PROJECT_PATH}/artifacts/data/train/merged_corpora_without_train.${TRG}" \
    --finetuning-full-sets "${PROJECT_PATH}/artifacts/data/train/full_augmented_data.txt.${SRC} ${PROJECT_PATH}/artifacts/data/train/full_augmented_data.txt.${TRG}" \
    --flags "${SPEED_FLAGS} ${DEFAULT_FLAGS} --devices ${GPUS} --tempdir ${PROJECT_PATH}/libs --valid-metrics cross-entropy translation --valid-sets ${VALID_SETS} --valid-translation-output ${TRANSLATION_OUTPUT} --train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS} --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log"