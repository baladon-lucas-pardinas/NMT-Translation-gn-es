#!/bin/bash
chmod +x /docker/home/marianmt/scripts/validate.sh
/marian/build/marian \
 --train-sets /docker/home/marianmt/tests/data/train_gn.txt /docker/home/marianmt/tests/data/train_es.txt \
 --model  /docker/home/marianmt/artifacts/models/model_gpu.npz --after-epochs 30 \
 --vocabs /docker/home/marianmt/tests/data/gn_unique_tokens.txt.nodup /docker/home/marianmt/tests/data/sp_unique_tokens.txt.nodup \
 --seed 1234 --cpu-threads 4 --devices 0 --log /docker/home/marianmt/logs/marian/model.log --valid-log /docker/home/marianmt/logs/marian/dev.log \
 --valid-sets /docker/home/marianmt/tests/data/val_gn.txt /docker/home/marianmt/tests/data/val_es.txt\
 --valid-metrics cross-entropy translation \
 --valid-script-path /docker/home/marianmt/scripts/validate.sh \
 --overwrite
