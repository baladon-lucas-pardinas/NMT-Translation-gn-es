from ..utils import process_manager
from ..utils import command_handler
from ..utils import file_manager
from src.components.evaluation import metrics

# Renames the checkpoint with the format model-<after_epochs>.<extension>
def __rename_checkpoint(model_dir, added_str):
    # type: (str, str) -> None
    model_dir = model_dir.split('.')
    model_checkpoint_path = model_dir[:-1]
    model_checkpoint_path = '.'.join(model_checkpoint_path)
    model_checkpoint_path += '-{}'.format(added_str)
    model_checkpoint_path += '.' + model_dir[-1]
    return model_checkpoint_path

def train(command_config):
    # type: (command_handler.CommandConfig) -> None
    marian_config = command_config.copy()
    validate_each_epochs = marian_config.validate_each_epochs

    if validate_each_epochs is None:
        command = command_handler.create_command(marian_config)
        process_manager.run_command(command)
        return
    
    flags = marian_config.flags
    validation_translation_output = flags.get('valid-translation-output', [None])[0]
    model_dir                     = flags.get('model', [None])[0]
    after_epochs                  = flags.get('after-epochs', [None])[0]
    _, valid_tgt                  = flags.get('valid-sets', [])
    command_name                  = marian_config.command_name
    validation_metrics            = marian_config.validation_metrics
    save_checkpoints              = marian_config.save_checkpoints

    if after_epochs is None:
        raise KeyError('after-epochs flag not found in config but validate_each_epochs is not None')
    after_epochs = int(after_epochs)
    artificial_epochs = after_epochs // validate_each_epochs

    for i in range(artificial_epochs):
        current_after_epochs = validate_each_epochs * (i + 1)
        marian_config.flags['after-epochs'] = [str(current_after_epochs)]
        command = command_handler.create_command(marian_config)
        process_manager.run_command(command)

        if model_dir is None:
            continue

        if not save_checkpoints is None:
            checkpoint_path = __rename_checkpoint(model_dir, current_after_epochs)
            file_manager.save_copy(model_dir, checkpoint_path)

        if not validation_translation_output is None:
            validation_translation_output_path = __rename_checkpoint(validation_translation_output, current_after_epochs)
            file_manager.save_copy(validation_translation_output, validation_translation_output_path)

        if not (validation_metrics is None or len(validation_metrics) == 0):
            metrics.save_results(
                command_name,
                validation_translation_output_path,
                valid_tgt,
                marian_config.flags,
                bleu_score_type=validation_metrics,
            )



            
        