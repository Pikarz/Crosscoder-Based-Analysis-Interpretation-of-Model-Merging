import torch 
from torch.utils.data import Dataset
from crosscoder_dataset_utils import get_model_activations

from get_dataloaders import get_dataloaders

class CrossCoderDataset(Dataset):
  def __init__(self, dice_path, pokemon_path, merged_path): # 
    super().__init__()
    # print("[DEBUG] CrossCoder Dataset Initialization...")

    dice_activations, _ = get_model_activations(dice_path)
    pokemon_activations, _ = get_model_activations(pokemon_path)
    merged_activations, _ = get_model_activations(merged_path)

    all_activations = torch.stack([torch.stack(dice_activations, dim=0),    \
                                   torch.stack(pokemon_activations, dim=0), \
                                   torch.stack(merged_activations, dim=0)], \
                                   dim=0) # [n_model, n_crosscoder_datapoints, n_activations]

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
  
  def get_model_activations(self, model_path):
    activations = []
    model_activations, _ = get_model_activations(model_path, activations) # Returns all activations of a model [n_crosscoder_datapoints, n_activations]  
    model_activations = torch.stack(model_activations, dim=0)
    
    return model_activations
  
def prepare_crosscoder_suite(
    name, enabled, dice_path, pkmn_path, merge_path, weights_path,
    batch_size, train_frac, val_frac, test_frac,
  ):
    if not enabled: # skip if not enabled, otherwise load the dataset
        return None

    # build dataset
    ds = CrossCoderDataset(dice_path, pkmn_path, merge_path)
    n_acts = ds.get_n_activations()

    # split
    train_loader, val_loader, test_loader = get_dataloaders(
        ds, batch_size, train_frac, val_frac, test_frac
    )

    total_steps = len(train_loader)

    return {
        "name": name,
        "dataset": ds,
        "n_activations": n_acts,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "weights_path": weights_path,
        "total_steps": total_steps,
    }


if __name__ == '__main__':
    dice_path = './activations_layer4/dice'
    pkmn_path = './activations_layer4/pokemon'
    merged_path = './activations_layer4/interpolated'

    crosscoder_dataset = CrossCoderDataset(dice_path, pkmn_path, merged_path)
    print(crosscoder_dataset.all_activations.size())
    print(crosscoder_dataset.all_activations.max(), crosscoder_dataset.all_activations.min())