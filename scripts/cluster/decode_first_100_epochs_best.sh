#!/bin/bash
MODEL_NAME=checkpoints_es_gn-100
MODEL_DIR=models/model_checkpoints_es_gn
PROJECT_PATH=/docker/home/marianmt
/marian/build/marian-decoder -m ${PROJECT_PATH}/artifacts/${MODEL_DIR}/${MODEL_NAME}.npz \
    --vocabs ${PROJECT_PATH}/artifacts/data/vocabulary/es_unique_tokens.txt.es ${PROJECT_PATH}/artifacts/data/vocabulary/gn_unique_tokens.txt.gn \
    --seed 1234 \
    --cpu-threads 0 \
    --log ${PROJECT_PATH}/logs/marian/es_gn_decode_model_${MODEL_NAME}.log \
    --output  ${PROJECT_PATH}/logs/marian/es_gn_decode_${MODEL_NAME}.log \
    --input ${PROJECT_PATH}/artifacts/data/validation/valid_es.txt.es


