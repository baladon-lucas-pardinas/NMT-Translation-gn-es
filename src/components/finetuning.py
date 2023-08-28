from ..config.command_config import CommandConfig

VALIDATION_FLAGS = ['valid-sets', 'valid-translation-output', 'valid-metrics']

def handle_validation_flags(finetuning_command_config, validation_flags=VALIDATION_FLAGS):
    # type: (CommandConfig, list) -> CommandConfig
    for validation_flag in validation_flags:
        if validation_flag in finetuning_command_config.flags:
            del finetuning_command_config.flags[validation_flag]
    return finetuning_command_config

def create_finetuning_vocabulary_train_config(command_config, full_sets):
    # type: (CommandConfig, list) -> CommandConfig
    finetuning_vocabulary_command_config = command_config.copy(deep=True)
    finetuning_vocabulary_command_config.flags['train-sets'] = full_sets.copy()
    finetuning_vocabulary_command_config['after-epochs'] = ['1']
    finetuning_vocabulary_command_config.not_delete_model_after = False
    finetuning_vocabulary_command_config = handle_validation_flags(finetuning_vocabulary_command_config)
    return finetuning_vocabulary_command_config

def create_finetuning_train_config(command_config, augmented_sets, finetuning_epochs):
    # type: (CommandConfig, list, int) -> CommandConfig
    finetuning_command_config = command_config.copy(deep=True)
    finetuning_command_config.flags['after-epochs'] = [str(finetuning_epochs)]  
    finetuning_command_config.flags['early-stopping'] = ['1000']
    finetuning_command_config.flags['train-sets'] = augmented_sets.copy()
    finetuning_command_config = handle_validation_flags(finetuning_command_config)
    return finetuning_command_config

def adapt_train_config(command_config, finetuning_epochs, new_model_path=None):
    # type: (CommandConfig, int, str) -> CommandConfig
    if new_model_path is not None:
        command_config.flags['model'] = [new_model_path]

    current_epochs = command_config.flags['after-epochs'][0]
    command_config.flags['after-epochs'] = [str(int(current_epochs) + int(finetuning_epochs))]
    command_config.flags['no-restore-corpus'] = []
    return command_config