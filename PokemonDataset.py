import torch
from torch.utils.data import Dataset

class PokemonDataset(Dataset):
    # The PokemonDataset will help us manage the data with the associated labels
    # it can also implement a series of transformations in input
    # it normalizes the items on the fly
  def __init__(self, pokemon_dataset, transform, normalization_dim): # pokemon_dataset is a datasetDict (huggingface)
    super().__init__()

    self.normalization_dim=normalization_dim # ie, [0,2,3]

    data = []
    labels = []
    for pok in pokemon_dataset:
      data_point = transform(pok["image"])
      data.append(data_point)
      labels.append(torch.as_tensor(pok['label']))

    self.data   = torch.stack(data)
    self.labels = torch.stack(labels)

    # compute per-channel stats once
    ch_mean = self.data.mean(dim=normalization_dim)   # over N,H,W -> [C]
    ch_std  = self.data.std(dim=normalization_dim)   # -> [C]
    # reshape for broadcasting in __getitem__
    self.mean = ch_mean.view(-1,1,1)
    self.std  = ch_std.view(-1,1,1)

  def __len__(self):
    return self.data.size()[0]

  def __getitem__(self, idx):
      x = self.data[idx]         # [C,H,W]
      x = (x - self.mean) / self.std
      y = self.labels[idx]
      return [x, y]