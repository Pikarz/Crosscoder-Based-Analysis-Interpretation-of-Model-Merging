import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage, Compose, Resize, CenterCrop, ToTensor
from torchvision import datasets

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

if __name__ == '__main__':
    from get_dataloaders import get_pokemon_dataloader
    from get_datasets import get_pokemon_dataset

    pokemon_dataset = get_pokemon_dataset()
    batch_size      = 32
    training_size   = 0.7
    validation_size = 0.2
    test_size       = 0.1

    train_loader, validation_loader, test_loader = get_pokemon_dataloader(pokemon_dataset, batch_size, training_size, validation_size, test_size)

    plot_dataset_examples(train_loader, 5, pokemon_dataset.std, pokemon_dataset.mean)