import os

from ..utils import process_manager
from ..utils import command_handler
from ..utils import file_manager
from src.components.evaluation import metrics

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

# Renames the checkpoint with the format model-<after_epochs>.<extension>
def rename_checkpoint(model_dir, added_str):
    # type: (str, str) -> str
    model_dir = model_dir.split('.')
    model_checkpoint_path = model_dir[:-1]
    model_checkpoint_path = '.'.join(model_checkpoint_path)
    model_checkpoint_path += '-{}'.format(added_str)
    model_checkpoint_path += '.' + model_dir[-1]
    return model_checkpoint_path

def train(command_config):
    # type: (command_handler.CommandConfig) -> None
    marian_config = command_config.copy(deep=True)
    validate_each_epochs = marian_config.validate_each_epochs

    if validate_each_epochs is None:
        command = command_handler.create_command(marian_config)
        process_manager.run_command(command)
        return
    
    flags                         = marian_config.flags
    validation_translation_output = flags.get('valid-translation-output', [None])[0]
    model_dir                     = flags.get('model', [None])[0]
    after_epochs                  = flags.get('after-epochs', [None])[0]
    batch_size                    = flags.get('after-batches', [None])[0]
    _, valid_tgt                  = flags.get('valid-sets', [])
    validation_metrics            = marian_config.validation_metrics
    save_checkpoints              = marian_config.save_checkpoints
    command_name                 = marian_config.command_name

    if after_epochs is None:
        raise KeyError('after-epochs flag not found in config but validate_each_epochs is not None')
    after_epochs, validate_each_epochs = int(after_epochs), int(validate_each_epochs)
    artificial_epochs = after_epochs // validate_each_epochs

    for i in range(artificial_epochs):
        current_after_epochs = validate_each_epochs * (i + 1)
        flags['after-epochs'] = [str(current_after_epochs)]
        command = command_handler.create_command(marian_config)
        process_manager.run_command(command)

        if model_dir is None:
            continue

        if save_checkpoints:
            checkpoint_path = rename_checkpoint(model_dir, current_after_epochs)
            file_manager.save_copy(model_dir, checkpoint_path)

        if not (
            validation_translation_output is None or 
            validation_metrics is None or 
            len(validation_metrics) == 0
        ):
            validation_translation_output_path = parse_output_filename(
                validation_translation_output, 
                epoch=current_after_epochs,
                batch=batch_size,
            )

            base_dir_evaluation = marian_config.base_dir_evaluation
            evaluation_file = os.path.join(base_dir_evaluation, command_name)
            metrics.save_results(
                file_name=evaluation_file,
                model_dir=model_dir,
                translation_output=validation_translation_output_path,
                reference=valid_tgt,
                parameters=marian_config.flags,
                metrics=validation_metrics,
            )