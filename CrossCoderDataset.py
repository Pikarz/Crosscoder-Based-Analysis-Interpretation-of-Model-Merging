import os
import torch 
from torch.utils.data import Dataset
from crosscoder_dataset_utils import get_model_activations

class CrossCoderDataset(Dataset):
  def __init__(self, path):
    super().__init__()
    # print("[DEBUG] CrossCoder Dataset Initialization...")

    all_activations = []
    for model in os.listdir(path):
      # print('[DEBUG] Checking a new model...')
      activations = []
      model_path = os.path.join(path, model)
      model_activations, _ = get_model_activations(model_path, activations) # Returns all activations of a model [n_crosscoder_datapoints, n_activations]  
      model_activations = torch.stack(model_activations, dim=0)
      all_activations.append(model_activations)

    all_activations = torch.stack(all_activations, dim=0) # [n_model, n_crosscoder_datapoints, n_activations]

    # Normalize across models and datapoints
    flat_activations = all_activations.view(-1, all_activations.size(-1))  # [n_models * n_datapoints, n_activations]
    self.mean = flat_activations.mean(dim=0)       # [n_activations]
    self.std = flat_activations.std(dim=0) + 1e-6  # avoid division by zero

    all_activations = (all_activations - self.mean[None, None, :]) / self.std[None, None, :]
    self.all_activations = all_activations

  def __len__(self):
    return self.all_activations.size()[1] # [n_models, n_crosscoder_datapoints, n_activations]

  def __getitem__(self, idx):
      x = self.all_activations[:, idx, :]
      return x
  
  def get_n_activations(self):
     return self.all_activations.size()[2]
  

if __name__ == '__main__':
    crosscoder_dataset = CrossCoderDataset('./activations')
    print(crosscoder_dataset.all_activations.max(), crosscoder_dataset.all_activations.min())