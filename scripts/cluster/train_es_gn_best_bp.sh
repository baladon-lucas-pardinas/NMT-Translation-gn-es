#!/bin/bash
MODEL_NAME=es_es_scored_june_sp
PROJECT_PATH=/docker/home/marianmt
EPOCHS=200
TRAIN_SETS="${PROJECT_PATH}/artifacts/data/train/train_es.txt.es ${PROJECT_PATH}/artifacts/data/train/train_gn.txt.gn"
VOCABS=" ${PROJECT_PATH}/artifacts/data/vocabulary/es_unique_tokens16.txt.es.spm ${PROJECT_PATH}/artifacts/data/vocabulary/gn_unique_tokens16.txt.gn.spm"
mkdir -p ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}
python3 ${PROJECT_PATH}/main.py --command-path  /marian/build \
    --validate-each-epochs 10 \
    --validation_metrics "sacrebleu_corpus_bleu sacrebleu_corpus_chrf" \
    --train \
    --flags "--train-sets ${TRAIN_SETS} --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs ${EPOCHS} --valid-freq 50000000 --vocabs ${VOCABS}  --seed 1234 --type amun --tied-embeddings-all --dim-vocabs 16384 16384 --normalize 0.6 --beam-size 6 --mini-batch-fit -w 3000 --layer-normalization --dropout-rnn 0.2 --dropout-src 0.1 --dropout-trg 0.1 --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/split_model_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/split_tok_150_${MODEL_NAME}.log --valid-sets ${PROJECT_PATH}/artifacts/data/validation/valid_gn.txt.gn ${PROJECT_PATH}/artifacts/data/validation/valid_es.txt.es --valid-metrics cross-entropy translation --valid-translation-output ${PROJECT_PATH}/evaluation/decoded-sp16-gn-es.txt --quiet-translation --overwrite"
