#!/bin/bash

#SBATCH --job-name=M-0.75-0.83
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=5-0
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort
#SBATCH --output=./tests/data\run_besteffort0.75-0.83.out
#SBATCH --gres=gpu:p100:1

cd ..
SCRIPT_NAME=$1
HOME=/docker/home
SCRIPT_PATH=${HOME}/scripts/${SCRIPT_NAME}
export SINGULARITY_TMPDIR=${HOME}/cache
export TMPDIR=$SINGULARITY_TMPDIR
chmod +x scripts/${SCRIPT_NAME}
export PYTHONPATH=${HOME}/libs
singularity exec -H ${HOME}/marianmt --nv --no-home --contain --bind $(pwd):$HOME marian-nmt_1.11.0_sentencepiece_cuda-11.3.0.sif $SCRIPT_PATH