#!/bin/bash
MODEL_NAME=gn_es_scored_june
PROJECT_PATH=/docker/home/marianmt
EPOCHS=2
mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}
python3 ${PROJECT_PATH}/main.py --command-path  /marian/build \
    --validation_metrics "sacrebleu_corpus_bleu sacrebleu_corpus_chrf" \
    --train \
    --flags "--train-sets ${PROJECT_PATH}/artifacts/data/train/train_gn.txt.gn ${PROJECT_PATH}/artifacts/data/train/train_es.txt.es --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 5000000 --vocabs ${PROJECT_PATH}/artifacts/data/vocabulary/gn_unique_tokens.txt.gn ${PROJECT_PATH}/artifacts/data/vocabulary/es_unique_tokens.txt.es --seed 1234 --type amun --mini-batch-fit -w 3000 --layer-normalization --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 --early-stopping 5 --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/split_model_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/split_tok_150_${MODEL_NAME}.log --valid-sets ${PROJECT_PATH}/artifacts/data/validation/valid_gn.txt.gn ${PROJECT_PATH}/artifacts/data/validation/valid_es.txt.es --valid-metrics cross-entropy translation --valid-translation-output ${PROJECT_PATH}/evaluation/model-epochs.txt --quiet-translation --sentencepiece-options '--normalization_rule_name=nmt_nfkc'"
