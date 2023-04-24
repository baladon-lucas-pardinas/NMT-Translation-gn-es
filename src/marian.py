import os
import subprocess
#import environ
#
#class MarianFiles:
#    def __init__(self, base_dir):
#        self.base_dir = base_dir
#        self.train_gn = os.path.join(self.base_dir, 'train_gn.txt')
#        self.train_es = os.path.join(self.base_dir, 'train_es.txt')
#        self.val_gn = os.path.join(self.base_dir, 'val_gn.txt')
#        self.val_es = os.path.join(self.base_dir, 'val_es.txt')
#        self.gn_unique_tokens = os.path.join(self.base_dir, 'gn_unique_tokens.txt')
#        self.sp_unique_tokens = os.path.join(self.base_dir, 'sp_unique_tokens.txt')
#        self.model = os.path.join(self.base_dir, 'model.npz')
#        self.log = os.path.join(self.base_dir, 'model.log')
#        self.dev_log = os.path.join(self.base_dir, 'dev.log')
#        self.validate_script = os.path.join(self.base_dir, 'validate.sh')

def train(base_dir_corpus: str, base_dir_scripts: str, **kwargs) -> None:
    command = f""" \
        marian \
            --train-sets {base_dir_corpus}/train_gn.txt {base_dir_corpus}/train_es.txt \
            --model ./model.npz \
            --after-epochs 1 \
            --vocabs {base_dir_corpus}/gn_unique_tokens.txt {base_dir_corpus}/sp_unique_tokens.txt \
            --seed 1234 \
            --devices 0 \
            --cpu-threads 4 \
            --log model.log \
            --valid-log dev.log \
            --valid-sets {base_dir_corpus}/val_gn.txt {base_dir_corpus}/val_es.txt \
            --valid-metrics cross-entropy translation \
            --valid-script-path {base_dir_scripts}/validate.sh \
            --overwrite \
    """

    print(command)

    #os.system(command)
    #output = subprocess.check_output(command, shell=True)
    #print(output.decode())