import torch
from torch.utils.data import DataLoader
import numpy as np

def get_dataloaders(dataset, batch_size, training_size, validation_size, test_size):

    assert np.allclose(np.array(training_size + validation_size + test_size), 1), "The splits must sum to one"

    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [training_size, validation_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f'Train dataset size: {len(train_set)}')
    print(f'Validation size: {len(valid_set)}')
    print(f'Test size: {len(test_set)}')

    print(f'Train size (batch): {len(train_loader)}')
    print(f'Validation size (batch): {len(val_loader)}')
    print(f'Test size (batch): {len(test_loader)}')

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    from get_datasets import get_pokemon_dataset

    pokemon_dataset = get_pokemon_dataset()
    batch_size      = 32
    training_size   = 0.7
    validation_size = 0.2
    test_size       = 0.1

    pokemon_dataloader = get_dataloaders(pokemon_dataset, batch_size, training_size, validation_size, test_size)
