# validate.sh
./postprocess.sh < $1 > file.out 2>/dev/null
./moses-scripts/scripts/generic/multi-bleu-detok.perl file.ref < file.out 2>/dev/null \
    | sed -r 's/BLEU = ([0-9.]+),.*/\1/