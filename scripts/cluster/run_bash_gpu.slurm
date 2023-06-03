#!/bin/bash

#SBATCH --job-name=MARIAN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --partition=normal
# --mail-type=ALL
# --mail-user=alexis.baladon@fing.edu.uy
#SBATCH --output=minimal.out

#SBATCH --gres=gpu:1      # se solicita una gpu cualquiera( va a tomar la primera que quede disponible indistintamente si es una p100 o una a100)
# --gres=gpu:p100:1 # se solicita una nvidia p100
# --gres=gpu:a100:1 # se solicita una nvidia a100, tenga en cuenta que solamente hay 2 a100 disponibles en toda la ifraestructura y están ubicadas en servidores diferentes (node15 y node16)
#SBATCH --qos=gpu
cd ..
SCRIPT_NAME=$1
HOME=/docker/home
SCRIPT_PATH=${HOME}/scripts/${SCRIPT_NAME}
chmod +x scripts/${SCRIPT_NAME}
echo $SCRIPT_PATH
singularity exec -H ${HOME}/marianmt  --nv --no-home --contain --bind $(pwd):$HOME marian-nmt_1.11.0_sentencepiece_cuda-11.3.0.sif $SCRIPT_PATH
