#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

cat $1 \
    | sed 's/\@\@ //g' \
    | ./moses-scripts/scripts/recaser/detruecase.perl 2>/dev/null \
    | ./moses-scripts/scripts/tokenizer/detokenizer.perl -l es 2>/dev/null \
    | ./moses-scripts/scripts/generic/multi-bleu-detok.perl ${SCRIPT_DIR}/../artifacts/data/validation/valid_es.txt.es \
    | sed -r 's/BLEU = ([0-9.]+),.*/\1/