#!/bin/bash
MODEL_NAME=checkpoint_100-80
MODEL_DIR=models/model_checkpoint_100
PROJECT_PATH=/docker/home/marianmt
/marian/build/marian-decoder -m ${PROJECT_PATH}/artifacts/${MODEL_DIR}/${MODEL_NAME}.npz \
    --vocabs ${PROJECT_PATH}/artifacts/data/vocabulary/gn_unique_tokens.txt.gn ${PROJECT_PATH}/artifacts/data/vocabulary/es_unique_tokens.txt.es \
    --seed 1234 \
    --devices 0 \
    --log ${PROJECT_PATH}/logs/marian/split_decode_model_${MODEL_NAME}.log \
    --output  ${PROJECT_PATH}/logs/marian/split_decode_${MODEL_NAME}.log \
    --input ${PROJECT_PATH}/artifacts/data/validation/valid_gn.txt.gn


