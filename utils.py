import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage, Compose, Resize, CenterCrop, ToTensor
from torchvision import datasets
import random
import CrossCoder
import os
from analysis import analyze_crosscoder

def seed_run():
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(0)

  torch.cuda.manual_seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

#Function to plot datasets, all the classes
def plot_dataset_examples(train_loader, n_classes, std, mean):
  images = {}

  for (image, label_tensor) in train_loader.dataset:
    label = label_tensor.item()

    if label not in images.keys():
      denorm_image = (image*std)+mean # denormalization
      images[label] = denorm_image

      if len(images.keys()) >= n_classes:
        break


  keys = sorted(images.keys())
  n = len(keys)

  fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))

  to_pil = ToPILImage()
  for ax, k in zip(axes, keys):
      tensor_img = images[k]        # C×H×W, already de-normalized
      # Convert to PIL then to NumPy (H×W×C)
      pil_img = to_pil(tensor_img.clamp(0,1))
      ax.imshow(np.array(pil_img))
      ax.set_title(f'Class {k}')
      ax.axis('off')

  plt.tight_layout()
  plt.show()

  ## test

# Given the path of the downloaded dice dataset files this function computes the mean and std
def compute_dice_mean_and_std(path):
    
    # Define transformations to be applied to the images
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        # Normalize(mean=mean, std=std),
        ToTensor()
    ])

    # Load the dataset using ImageFolder (implicitly assumes directories for class labels)
    dataset = datasets.ImageFolder(root=path, transform=transform)

    images, _ = zip(*dataset)
    images_tensor = torch.stack(images, dim=0)

    dim = [0, 2, 3]

    mean, std = torch.mean(images_tensor, dim=dim), torch.std(images_tensor, dim=dim)

    return mean, std

def get_equally_distributed_subset(pokemon_dataset, dice_dataset, small_imagenet_dataset, n_models,
                                    crosscoder_datapoints=None):    
    # If we do not set a number of datapoints per dataset, we just take the smallest possible number so that the whole dataset remains equally distributed among the three different datasets
    if not crosscoder_datapoints:
      num_points_per_dataset = min(len(pokemon_dataset), len(dice_dataset), len(small_imagenet_dataset))
    else:
      num_points_per_dataset = int(crosscoder_datapoints // n_models) # we take the integer
    
    # Get subsets so that all datasets have an equal number of datapoints
    pkmn_subset             = torch.utils.data.Subset(pokemon_dataset, np.arange(0, num_points_per_dataset))
    dice_subset             = torch.utils.data.Subset(dice_dataset, np.arange(0, num_points_per_dataset))
    small_imagenet_subset   = torch.utils.data.Subset(small_imagenet_dataset, np.arange(0, num_points_per_dataset))

    return pkmn_subset, dice_subset, small_imagenet_subset


if __name__ == '__main__':
    from get_dataloaders import get_dataloaders
    from get_datasets import get_pokemon_dataset

    pokemon_dataset = get_pokemon_dataset()
    batch_size      = 32
    training_size   = 0.7
    validation_size = 0.2
    test_size       = 0.1
    number_classes  = 5

    train_loader, validation_loader, test_loader = get_dataloaders(pokemon_dataset, batch_size, training_size, validation_size, test_size)

    plot_dataset_examples(train_loader, number_classes, pokemon_dataset.std, pokemon_dataset.mean)
    
def train_crosscoder_and_save_weights(
    latent_dim, 
    n_activations,
    lambda_sparse,
    total_steps,
    train_loader,
    num_epochs,
    training_lr,
    crosscoder_weights_path,
    experiment_name,
    wandb_config,
    project_name,
    description
  ):
    crosscoder = CrossCoder.CrossCoder(latent_dim, n_activations, lambda_sparse, total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crosscoder.to(device)

    crosscoder.train_cross(train_loader, num_epochs, training_lr, experiment_name, wandb_config, project_name, description)

    dirpath = os.path.dirname(crosscoder_weights_path)
    
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    # save the model’s state dict
    torch.save(crosscoder.state_dict(), crosscoder_weights_path)
    
def validate_crosscoder(
  latent_dim,
  n_activations,
  lambda_sparse,
  total_steps,
  crosscoder_weights_path,
  crosscoder_val_loader,
  ):
  device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  crosscoder = CrossCoder.CrossCoder(latent_dim, n_activations, lambda_sparse, total_steps)
  crosscoder = crosscoder.to(device)
  crosscoder.load_state_dict(torch.load(crosscoder_weights_path, weights_only=True))

  crosscoder.val_cross(crosscoder_val_loader)