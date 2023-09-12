from typing import Any
import os
import re
import copy

PRETRAINING_EPOCHS = 'finetuning-epochs'
SPM_SUFFIX = '.spm'

# https://marian-nmt.github.io/docs/cmd/marian/
# --valid-translation-output TEXT       
# (Template for) path to store the translation. E.g., 
# validation-output-after-{U}-updates-{T}-tokens.txt. 
# Template parameters: {E} for epoch; {B} for No. of batches 
# within epoch; {U} for total No. of updates; {T} for total 
# No. of tokens seen.
def parse_output_filename(output_filename, epoch=None, batch=None, updates=None, tokens=None):
    # type: (str, int, int, int, int) -> str
    to_substitute = {'{E}': epoch, '{B}': batch, '{U}': updates, '{T}': tokens}
    for key, value in to_substitute.items():
        if value is None:
            continue
        output_filename = output_filename.replace(key, str(value))
    return output_filename

def handle_finetuning_flags(finetuning_config, flags):
    # type: (Any, dict) -> (Any, dict)
    pretraining_epochs = flags.pop(PRETRAINING_EPOCHS, [None])[0]
    if pretraining_epochs is not None:
        finetuning_config.epochs = pretraining_epochs
    return finetuning_config, flags

# If the flag value is bool: 
#     If its value is True, the flag value should be an empty list (e.g. '--overwrite' has no value, unlike '--model model.npz')
#     If its value is False, the flag should be deleted (because False is the default value)
def handle_boolean_flags(flags):
    # type: (dict[str, list]) -> dict[str, list]
    temp_flags = deep_copy_flags(flags) # Dicts shouldn't be changed during iteration
    for flag_name, flag_value in flags.items():
        flag_value_id = str(flag_value)
        if flag_value_id == "[True]":
            temp_flags[flag_name] = []
        elif flag_value_id == "[False]":
            del temp_flags[flag_name]
    return temp_flags

def has_sentencepiece_vocabs(src_vocab, trg_vocab, spm_suffix=SPM_SUFFIX):
    # type: (str, str, str) -> bool
    has_sentencepiece_vocabs = src_vocab.endswith(spm_suffix) \
                               or trg_vocab.endswith(spm_suffix)

    if not has_sentencepiece_vocabs:
        return False

    if not (src_vocab.endswith(spm_suffix) and trg_vocab.endswith(spm_suffix)):
        raise TypeError('Both vocabularies must be of type sentencepiece if one of them is.')
    
    return has_sentencepiece_vocabs

def already_exists_vocabulary(src_vocab, trg_vocab):
    # type: (str, str) -> bool
    return os.path.isfile(src_vocab) and os.path.isfile(trg_vocab)

# If the model uses sentencepiece, each vocabulary configuration must be in a different file.
def handle_vocabularies(flags, spm_suffix=SPM_SUFFIX):
    # type: (dict[str, list], str) -> dict[str, list]
    src_vocab, trg_vocab = flags.get('vocabs', ['', ''])

    if has_sentencepiece_vocabs(src_vocab, trg_vocab):
        dim_vocab = flags.get('dim-vocabs', [None])[0]
        if dim_vocab is not None:
            dim_vocab = dim_vocab.replace(' ', '_') # In case dim-vocabs is passed as a string instead of a list of ints
            vocab_suffix_with_size = 'V' + dim_vocab + spm_suffix
            src_new_name = src_vocab.replace(spm_suffix, vocab_suffix_with_size)
            trg_new_name = trg_vocab.replace(spm_suffix, vocab_suffix_with_size)

            flags['vocabs'] = [src_new_name, trg_new_name]
    return flags

def rename_model_file(model_name, flags):
    # type: (str, dict[str, list]) -> str
    model_name_without_extension = '.'.join(model_name.split('.')[:-1])
    model_name_extension = model_name.split('.')[-1]
    param_names = list(flags.keys())
    param_values = list(map(lambda v: str(v[0]), flags.values()))
    param_names_and_values = [param_name + '_' + param_value for param_name, param_value in zip(param_names, param_values)]
    model_name = model_name_without_extension + '_' + '_'.join(param_names_and_values) + '.' + model_name_extension
    model_name = model_name.replace(' ', '')
    return model_name

# Flags are a string of the form --flag1 value1 ... valueN --flag2 value1 ... valueM
# The output dict should be like {flag1: [value1, ..., valueN], flag2: [value1, ..., valueM]}
def parse_flags(flags, flag_separator=' '):
    # type: (str, str) -> dict
    flag_dict = {}
    flags = flags.split(flag_separator)
    for flag in flags:
        flag = flag.strip()
        if flag == "":
            continue

        splitted_flag = flag.split(' ')
        flag_name = splitted_flag[0]
        if all([c in ['"', "'"] for c in [flag[0], flag[-1]]]): # Flags of type --valid-script-path "bash script.sh"
            flag_values = [flag]
        else:
            flag_values = splitted_flag[1:]

        flag_dict[flag_name] = flag_values
    return flag_dict

def deep_copy_flags(flags):
    # type: (dict) -> dict
    return copy.deepcopy(flags)

# Flags are a list of tuples (flag, values: list)
# The output string should be like --flag1 value1 ... valueN --flag2 value1 ... valueM
def create_command_flags(flags):
    # type: (dict) -> str
    output = ""
    flags = list(flags.items())
    for flag, values in flags:
        output += " --" + flag
        for value in values:
            output += " " + value
        output += " "
    return output

def create_command(config):
    # type: (Any) -> str
    command = ""
    command += os.path.join(config.command_path, config.command_name)
    command += create_command_flags(config.flags)

    return command

def parse_line_groups(lines, regex):
    # type: (list[str], re.Pattern) -> list[tuple[str, str, str, str, str, str]]
    line_groups = []

    for line in lines:
        match = regex.match(line)
        if match:
            line_groups.append(match.groups()) 

    return line_groups

