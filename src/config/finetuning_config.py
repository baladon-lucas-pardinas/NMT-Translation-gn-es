class FinetuningConfig:
    def __init__(
        self, 
        epochs,
        augmented_sets,
        full_sets=None,
    ):
        # type: (int, list, list) -> None
        self.epochs = epochs
        self.augmented_sets = augmented_sets
        self.full_sets = full_sets

    def copy(self, deep=False):
        # type: (bool) -> FinetuningConfig
        return FinetuningConfig(
            epochs=self.epochs,
            augmented_sets=self.augmented_sets,
            full_sets=self.full_sets,
        )
    
    def __str__(self):
        # type: () -> str
        return "FinetuningConfig(epochs={}, full_sets={}, augmented_sets={})".format(
            self.epochs,
            self.full_sets,
            self.augmented_sets,
        )
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_finetuning_config(
    epochs,
    finetuning_augmented_sets,
    finetuning_full_sets,
):
    # type: (int, str, str) -> FinetuningConfig
    return FinetuningConfig(
        epochs=epochs,
        augmented_sets=finetuning_augmented_sets,
        full_sets=finetuning_full_sets,
    )