import os
import logging
from src import logger
import subprocess

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
        --vocabs {base_dir}/vocabulary/gn_unique_tokens.txt {base_dir}/vocabulary/sp_unique_tokens.txt \
        --input {base_dir}/test/test_gn.txt \
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

def minimal_train_gpu(marian_dir: str, corpus_dir: str) -> None:
    # TODO: Arreglar direcciones para unirlas con os.path.join
    logger.info(f'Starting minimal_train_gpu with corpus_dir: {corpus_dir} and marian_dir: {marian_dir}')
    command = f""" \
        {marian_dir}/marian \
            --train-sets {corpus_dir}/train/train_gn.txt {corpus_dir}/train/train_es.txt \
            --model {marian_dir}/model.npz \
            --after-epochs 1 \
            --vocabs {corpus_dir}/vocabulary/gn_unique_tokens.txt {corpus_dir}/vocabulary/sp_unique_tokens.txt \
            --seed 1234 \
            --cpu-threads {os.cpu_count()} \
            --devices 0 \
            --log model.log \
            --valid-log dev.log \
            --valid-sets {corpus_dir}/validation/val_gn.txt {corpus_dir}/validation/val_es.txt \
            --valid-metrics cross-entropy translation \
            --valid-script-path scripts/validate.sh \
            --overwrite
    """
    command = 'echo This is a test example'
    logger.info(f'Running command: {command}')
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    logger.info('Finished minimal_train_gpu')
    logger.info(f'Process finished with return code: {result.returncode}')
    logger.info(f'Process output: {result.stdout}')