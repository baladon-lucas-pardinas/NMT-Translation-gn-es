from ..utils import process_manager
from ..utils import command_handler
from ..utils import file_manager

def __rename_checkpoint(model_dir, after_epochs):
    # type: (str, int) -> None
    model_dir = model_dir.split('.')
    model_checkpoint_path = model_dir[:-1]
    model_checkpoint_path = '.'.join(model_checkpoint_path)
    model_checkpoint_path += '-{}'.format(after_epochs)
    model_checkpoint_path += '.' + model_dir[-1]
    return model_checkpoint_path

def train(config, save_each_epochs=None):
    # type: (command_handler.CommandConfig, int) -> None
    marian_config = config.copy()

    if save_each_epochs is None:
        command = command_handler.create_command(marian_config)
        process_manager.run_command(command)
        return
    
    model_dir = marian_config.flags.get('model', None)[0]
    after_epochs = marian_config.flags.get('after-epochs', None)[0]
    if after_epochs is None:
        raise KeyError('after-epochs flag not found in config but save_each_epochs is not None')
    after_epochs = int(after_epochs)
    artificial_epochs = after_epochs // save_each_epochs
    for i in range(artificial_epochs):
        current_after_epochs = save_each_epochs * (i + 1)
        marian_config.flags['after-epochs'] = [str(current_after_epochs)]
        command = command_handler.create_command(marian_config)
        process_manager.run_command(command)
        if not model_dir is None:
            checkpoint_path = __rename_checkpoint(model_dir, current_after_epochs)
            file_manager.save_copy(model_dir, checkpoint_path)
        