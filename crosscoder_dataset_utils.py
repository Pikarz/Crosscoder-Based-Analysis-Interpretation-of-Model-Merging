import torch
from pathlib import Path
from get_dataloaders import get_dataloaders
from utils import get_equally_distributed_subset
from tqdm import tqdm
import os
import re
import CrossCoderDataset

# def save_models_activations(models, dataloaders, activations_paths):
#   # Create directories to store the activations
#   # The training of Crosscoder will be done through files due to insufficient memory
#   for path in activations_paths:
#     Path(path).mkdir(parents=True, exist_ok=True)

#   # Register hooks on every module
#   for model, activations_path in zip(models, activations_paths):
#     print(model, activations_path)

#     print(f'[DEBUG] Started model testing which activations will be saved in {activations_path}')
#     test_model_on_dataloaders(model, dataloaders)
#     print(f'[OK] All activations have been saved')
    
def create_crosscoder_dataset(  pokemon_dataset, dice_dataset, small_imagenet_dataset, batch_size,
                                models, 
                                activations_output_path, crosscoder_datapoints, regex_activations, n_models=3):

    # equally distribute the data -- we have three subsets with an equal number of points
    print('[DEBUG] Preparing equally distributed subsets')
    pkmn_subset, dice_subset, small_imagenet_subset = get_equally_distributed_subset(pokemon_dataset, dice_dataset, small_imagenet_dataset, n_models, crosscoder_datapoints)

    print(f'[DEBUG] Dataset Subsets Size:\n \
              Pokemon: {len(pkmn_subset)}\n \
              Dice: {len(dice_subset)}\n \
              Small Imagenet: {len(small_imagenet_subset)}')
    
    # Get corresponding dataloaders
    pkmn_dataloader, _, _           = get_dataloaders(pkmn_subset,           batch_size, 1.0, 0.0, 0.0) # It's already a subset so we just get the whole dataloader
    dice_dataloader, _, _           = get_dataloaders(dice_subset,           batch_size, 1.0, 0.0, 0.0) # It's already a subset so we just get the whole dataloader
    small_imagenet_dataloader, _, _ = get_dataloaders(small_imagenet_subset, batch_size, 1.0, 0.0, 0.0) # It's already a subset so we just get the whole dataloader

    # Put dataloaders into a list
    dataloaders = [pkmn_dataloader, dice_dataloader, small_imagenet_dataloader]

    print('[DEBUG] Starting saving models activations')
    for model, activation_path in zip(models, activations_output_path):
      create_hooks_and_test_model(model, dataloaders, activation_path, regex_activations)

### Create hooks allow us to save the activations along every layer of our ResNet during the forward pass
def create_hooks(model, activations_path, regex_activations):
  # Hook factory
  def get_hook(name, activations_path):
    def hook(module, inp, out):
      if bool( re.search(regex_activations, name) ):
        torch.save(out.detach(), f"{activations_path}/{name}.torch") # detach so we don’t keep the computation graph
        handle.remove() # removes the hook once it has finished its job

    return hook
  
  counter = 0
  for name, module in model.named_modules():
     # if conv in name: # We get only the module 'conv'
        handle = module.register_forward_hook(get_hook(name, activations_path))
        counter += 1

# multiple dataloaders test
def create_hooks_and_test_model(model, data_loaders, model_path, regex_activations):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  count = 0

  with torch.no_grad():
    for data_loaders in data_loaders:
      loop = tqdm(enumerate(data_loaders), total=len(data_loaders), desc="Computing Activations", leave=True)
      for _, (test_inputs, _) in loop: # we don't need the labels at all
        activations_path = model_path + '/activations_' + str(count)
        count += 1
        Path(activations_path).mkdir(parents=True, exist_ok=True) # create directory

        create_hooks(model, activations_path, regex_activations)
        test_inputs = test_inputs.to(device)

        # preds
        test_outputs = model(test_inputs)

def get_model_activations(path, activations_tensors=[], activations=torch.as_tensor([])):
  if os.path.isfile(path) and path.endswith('.torch'): # We found a tensor inside the recursive tree. The second condition is always true, it's there just to clarify things
    activations_layer = torch.load(path).cpu().view(-1)      # We view the activations as a single-dimensional tensor
    activations = torch.cat([activations, activations_layer])
    return activations_tensors, activations
  
  #print(f'[DEBUG] Checking folder {path}')
  content = os.listdir(path) 

  for file in content:    # We iterate within the content of the current folder. We have two cases: either it is another folder, or it is a tensor and we load it
    subpath = os.path.join(path, file)
    activations_tensors, activations = get_model_activations(subpath, activations_tensors, activations)

  if activations.size()[0] > 0: # if there's at least a .torch file in the current list
    # print('[DEBUG] Adding new activations...')
    activations_tensors.append(activations)
    activations = torch.as_tensor([])
    
  return activations_tensors, activations
