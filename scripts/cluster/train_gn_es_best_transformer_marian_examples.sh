#!/bin/bash
MODEL_NAME=gn_es_scored_june_transformer_example_depth2
PROJECT_PATH=/docker/home/marianmt
EPOCHS=200
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_gn.txt.gn ${PROJECT_PATH}/artifacts/data/train/train_es.txt.es"
VALID_SETS="${PROJECT_PATH}/artifacts/data/validation/valid_gn.txt.gn ${PROJECT_PATH}/artifacts/data/validation/valid_es.txt.es"
VOCABS="${PROJECT_PATH}/artifacts/data/vocabulary/gn_unique_tokens16.txt.gn.spm ${PROJECT_PATH}/artifacts/data/vocabulary/es_unique_tokens16.txt.es.spm"
TRANSLATION_OUTPUT=${PROJECT_PATH}/evaluation/decoded-transformer-example-gn-es.txt
MODEL_FLAGS="--max-length 100 --mini-batch-fit --mini-batch 1000 --beam-size 12 --normalize 1 --valid-mini-batch 64 --enc-depth 2 --dec-depth 2 --transformer-heads 8 --transformer-postprocess-emb d --transformer-postprocess dan --transformer-dropout 0.1 --label-smoothing 0.1 --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 --tied-embeddings-all --exponential-smoothing --sync-sgd"
mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}
python3 ${PROJECT_PATH}/main.py --command-path  /marian/build \
    --validate-each-epochs 1 \
    --validation_metrics "sacrebleu_corpus_bleu sacrebleu_corpus_chrf" \
    --train \
    --flags "--train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS} --seed 1234 --type transformer --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/log_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/valid_log_${MODEL_NAME}.log --valid-sets ${VALID_SETS} --valid-metrics cross-entropy translation --valid-translation-output ${TRANSLATION_OUTPUT} --quiet-translation --overwrite ${MODEL_FLAGS}"
