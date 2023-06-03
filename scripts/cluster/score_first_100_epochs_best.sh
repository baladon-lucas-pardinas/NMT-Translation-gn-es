#!/bin/bash
MODEL_NAME=checkpoint_100-50
MODEL_DIR=model_checkpoint_100
PROJECT_PATH=/docker/home/marianmt
/marian/build/marian-scorer -m ${PROJECT_PATH}/artifacts/${MODEL_DIR}/${MODEL_NAME}.npz \
    --vocabs ${PROJECT_PATH}/tests/data/gn_unique_tokens.txt.nodup ${PROJECT_PATH}/tests/data/sp_unique_tokens.txt.nodup \
    --seed 1234 \
    --devices 0 \
    --log ${PROJECT_PATH}/logs/marian/scored_model_${MODEL_NAME}.log \
    -t  ${PROJECT_PATH}/logs/marian/first_decode_${MODEL_NAME}.log ${PROJECT_PATH}/artifacts/data/validation/val_es.txt \


