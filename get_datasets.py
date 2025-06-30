import torch
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Lambda

from PokemonDataset import PokemonDataset


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

if __name__ == '__main__':
    dataset = get_pokemon_dataset()
    print(dataset)