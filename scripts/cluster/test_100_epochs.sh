#!/bin/bash
MODEL_NAME=checkpoint_100
PROJECT_PATH=/docker/home/marianmt
chmod +x ${PROJECT_PATH}/scripts/validate.sh
python3 ${PROJECT_PATH}/main.py --command-path  /marian/build \
    --save-each-epochs 10 \
    --train \
    --flags "--train-sets ${PROJECT_PATH}/artifacts/data/train/train_gn.txt.gn ${PROJECT_PATH}/artifacts/data/train/train_es.txt.es --model ${PROJECT_PATH}/artifacts/models/model_${MODEL_NAME}/${MODEL_NAME}.npz --after-epochs 150 --save-freq 5000000 --vocabs ${PROJECT_PATH}/artifacts/data/vocabulary/gn_unique_tokens.txt.gn ${PROJECT_PATH}/artifacts/data/vocabulary/es_unique_tokens.txt.es --seed 1234 --cpu-threads 0 --log ${PROJECT_PATH}/logs/marian/split_model_${MODEL_NAME}.log --valid-log ${PROJECT_PATH}/logs/marian/split_tok_150_${MODEL_NAME}.log --valid-sets ${PROJECT_PATH}/artifacts/data/validation/valid_gn.txt.gn ${PROJECT_PATH}/artifacts/data/validation/valid_es.txt.es --valid-metrics bleu chrf cross-entropy perplexity valid-script  --valid-script-path ${PROJECT_PATH}/scripts/validate.sh"
