class FinetuningConfig:
    def __init__(self, 
                 epochs,
                 augmented_sets,
                 full_sets=None,
                 cache_dir_template=None):
        # type: (int, list, list, str) -> None
        self.epochs = epochs
        self.augmented_sets = augmented_sets
        self.full_sets = full_sets
        self.cache_dir_template = cache_dir_template

    def copy(self, deep=False):
        # type: (bool) -> FinetuningConfig
        return FinetuningConfig(epochs=self.epochs,
                                augmented_sets=self.augmented_sets.copy(),
                                full_sets=self.full_sets.copy() \
                                          if self.full_sets is not None \
                                          else self.full_sets,
                                cache_dir_template=self.cache_dir_template)
    
    def __str__(self):
        # type: () -> str
        return "FinetuningConfig(epochs={}, full_sets={}, augmented_sets={}, cache_dir_template={})".format(
            self.epochs,
            self.full_sets,
            self.augmented_sets,
            self.cache_dir_template)
    
    def __repr__(self):
        # type: () -> str
        return str(self)

def get_finetuning_config(epochs,
                          finetuning_augmented_sets,
                          finetuning_full_sets,
                          cache_dir_template):
    # type: (int, list, list, str) -> FinetuningConfig
    return FinetuningConfig(epochs=epochs,
                            augmented_sets=finetuning_augmented_sets,
                            full_sets=finetuning_full_sets,
                            cache_dir_template=cache_dir_template)