#!/bin/bash
#SBATCH --job-name=MARIAN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --output=docker.out

#singularity pull  docker://intel/nmt_marian_framework_demo:latest
#singularity pull docker://lefterav/marian-nmt:1.11.0_sentencepiece_cuda-11.3.0
echo starting download...
#export SINGULARITY_DISABLE_CACHE=1
#export TMPDIR=/cache
#export SINGULARITY_CACHEDIR=/cache
export SINGULARITY_TMPDIR=$(pwd)/cache
singularity pull docker://lefterav/marian-nmt:1.11.0_sentencepiece_cuda-11.3.0

