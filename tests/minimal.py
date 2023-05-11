import os
import logging
from src import logger

logger = logging.getLogger(__name__)

def minimal_train_cpu(base_dir: str) -> None:
    command = f""" \
    content/marian/build/marian \
        --train-sets {base_dir}/train_gn.txt {base_dir}/train_es.txt \
        --model ./model.npz \
        --after-epochs 1 \
        --vocabs {base_dir}/gn_unique_tokens.txt {base_dir}/sp_unique_tokens.txt \
        --seed 1234 \
        --devices 0 \
        --cpu-threads 4 \
        --log model.log \
        --valid-log dev.log \
        --valid-sets {base_dir}/val_gn.txt {base_dir}/val_es.txt \
        --valid-metrics cross-entropy translation \
        --valid-script-path ./validate.sh \
        --overwrite
    """
    os.system(command)

def minimal_evaluation_cpu(base_dir: str) -> None:
    command = f""" \
    content/marian/build/marian-decoder \
        --models ./model.npz \
        --vocabs {base_dir}/gn_unique_tokens.txt {base_dir}/sp_unique_tokens.txt \
        --input {base_dir}/test_gn.txt \
        --output test_es.txt \
        --beam-size 12 \
        --normalize 1 \
        --max-length 100 \
        --max-length-crop \
        --quiet-translation \
        --log test.log \
        --quiet
    """
    os.system(command)

def minimal_train_gpu(base_dir: str) -> None:
    # TODO: Arreglar direcciones para unirlas con os.path.join
    # TODO: Docker borra nuevos datos al cerrar
    logger.info(f'Starting minimal_train_gpu with base_dir: {base_dir}')
    command = f""" \
    content/marian/build/marian \
        --train-sets {base_dir}/train_gn.txt {base_dir}/train_es.txt \
        --model ./model.npz \
        --after-epochs 1 \
        --vocabs {base_dir}/gn_unique_tokens.txt {base_dir}/sp_unique_tokens.txt \
        --seed 1234 \
        --devices 0 \
        --log model.log \
        --valid-log dev.log \
        --valid-sets {base_dir}/val_gn.txt {base_dir}/val_es.txt \
        --valid-metrics cross-entropy translation \
        --valid-script-path ./validate.sh \
        --overwrite
    """
    logger.info(f'Running command: {command}')
    #os.system(command)
    logger.info('Finished minimal_train_gpu')