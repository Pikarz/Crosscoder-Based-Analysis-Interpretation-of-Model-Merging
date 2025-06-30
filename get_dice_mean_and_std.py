import torch

from torchvision import datasets
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

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

