import torch
import kagglehub

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Lambda, Normalize
from torchvision import datasets

from PokemonDataset import PokemonDataset

from datasets import load_dataset # hugging face dependency


def get_pokemon_dataset():
    pokemon_dataset = load_dataset("manuel-yao/pokemon-keras-community", split='train')

    # The PokemonDataset will help us manage the data with the associated labels
    # it can also implement a series of transformations in input
    # and it normalizes the datapoints on the fly 


    # lambda to convert all images to RGB -- removes alpha channel issue. Now all images have 3 channels
    # center crop to extract the relevant features
    # resize to 224 the images -- same as resnet input
    # the normalization is computed on the fly on __getitem__()
    transform = Compose([
        Lambda(lambda img: img.convert("RGB")), # we remove the alpha channel
        Resize(256),
        CenterCrop(224),
        ToTensor() # we normalize on the fly on __getitem__
    ])

    normalization_dim = [0,2,3] # we normalize on the second dim -- the channels
    pokemon_dataset = PokemonDataset(pokemon_dataset, transform, normalization_dim) 

    return pokemon_dataset

def get_dice_dataset():
    # Download dice dataset latest version
    path = kagglehub.dataset_download("ucffool/dice-d4-d6-d8-d10-d12-d20-images")

    # already calculated by running get_dice_mean_and_std() on utils.py
    mean = [0.6590, 0.6188, 0.6082]
    std = [0.2242, 0.2224, 0.2398]

    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    # this must be done because the kaggle dataset has a weird directory hierarchy
    # we want to have everything on the same level (no assumptions on train/valid yet)
    # then we split in train/valid/test using different dataloaders
    dataset_1 = datasets.ImageFolder(root=path+'\\dice\\train', transform=transform)
    dataset_2 = datasets.ImageFolder(root=path+'\\dice\\valid', transform=transform)

    l = []
    l.append(dataset_1)
    l.append(dataset_2)
    image_datasets = torch.utils.data.ConcatDataset(l)

    return image_datasets

if __name__ == '__main__':
    dataset = get_pokemon_dataset()
    print(dataset)