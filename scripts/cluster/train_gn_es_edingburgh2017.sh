#!/bin/bash
MODEL_NAME=gn_es_edinburgh2017
PROJECT_PATH=/docker/home/marianmt
EPOCHS=100
mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}
python3 ${PROJECT_PATH}/main.py --command-path /marian/build \
    --validate-each-epochs 10 \
    --validation_metrics "sacrebleu_corpus_bleu sacrebleu_corpus_chrf" \
    --train \
    --flags "--train-sets ${PROJECT_PATH}/artifacts/data/train/train_gn.txt.gn ${PROJECT_PATH}/artifacts/data/train/train_es.txt.es --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 5000000 --vocabs ${PROJECT_PATH}/artifacts/data/vocabulary/gn_unique_tokens.txt.gn ${PROJECT_PATH}/artifacts/data/vocabulary/es_unique_tokens.txt.es --seed 1234 --type s2s --max-length 100 --mini-batch-fit --mini-batch 1000 --maxi-batch 1000 --beam-size 12 --normalize 1 --cost-type ce-mean-words --enc-type bidirectional --enc-depth 1 --enc-cell-depth 4 --dec-depth 1 --dec-cell-base-depth 8 --dec-cell-high-depth 1 --layer-normalization --dropout-rnn 0.1 --label-smoothing 0.1 --learn-rate 0.0003 --lr-decay-inv-sqrt 16000 --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 --sync-sgd --exponential-smoothing --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/split_model_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/split_tok_150_${MODEL_NAME}.log --valid-sets ${PROJECT_PATH}/artifacts/data/validation/valid_gn.txt.gn ${PROJECT_PATH}/artifacts/data/validation/valid_es.txt.es --valid-metrics cross-entropy translation --valid-translation-output ${PROJECT_PATH}/evaluation/model-epochs.txt --quiet-translation"
