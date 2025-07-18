import os
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from crosscoder_dataset_utils import get_model_activations

class CrossCoderDataset(Dataset):
    def __init__(self, pokemon_path: str, dice_path: str, merge_path: str):
        super().__init__()
        
        # Create ordered list of model paths and names
        model_paths = [dice_path, pokemon_path, merge_path]
        self.model_names = ['dice', 'pokemon' , 'merge']
        
        print(f"[INFO] Loading Models: {self.model_names}")
        
        # Get activations for each model
        all_activations = []
        n_datapoints_per_model = []
        
        for i, model_path in enumerate(model_paths):
            model_name = self.model_names[i]
            activations = []
            model_activations, *_ = get_model_activations(model_path, activations)
            model_activations = torch.stack(model_activations, dim=0)
            
            all_activations.append(model_activations)
            n_datapoints_per_model.append(model_activations.shape[0])
            
            # print(f"[INFO] {model_name}: {model_activations.shape[0]} datapoints, "
            #       f"{model_activations.shape[1]} activations")
        
        # Stack delle attivazioni
        all_activations = torch.stack(all_activations, dim=0)  # [n_models, n_datapoints, n_activations]
        
        # Calcola statistiche per normalizzazione
        flat_activations = all_activations.view(-1, all_activations.size(-1))
        self.mean = flat_activations.mean(dim=0)
        self.std = flat_activations.std(dim=0) + 1e-6
        
        # normalization
        self.all_activations = (all_activations - self.mean[None, None, :]) / self.std[None, None, :]
        
        # metadata
        self.n_models = len(self.model_names)
        self.n_datapoints = all_activations.shape[1]
        self.n_activations = all_activations.shape[2]
        
        print(f"[INFO] Dataset created: {self.n_models} models, "
              f"{self.n_datapoints} datapoints, {self.n_activations} activations")
    
    def __len__(self):
        return self.n_datapoints
    
    def __getitem__(self, idx):
        # return the same activation for all the models
        return self.all_activations[:, idx, :]
    
    def get_n_activations(self):
        return self.n_activations
    
    def denormalize(self, normalized_activations: torch.Tensor) -> torch.Tensor:
        """Denormalize the activations."""
        return normalized_activations * self.std + self.mean
    
    def get_model_activation(self, model_idx: int, datapoint_idx: int) -> torch.Tensor:
        return self.all_activations[model_idx, datapoint_idx, :]
    
    def get_stats(self) -> dict:
        return {
            'n_models': self.n_models,
            'n_datapoints': self.n_datapoints,
            'n_activations': self.n_activations,
            'model_names': self.model_names,
            'mean_shape': self.mean.shape,
            'std_shape': self.std.shape
        }