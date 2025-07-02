import torch
from pathlib import Path
from get_datasets import get_small_imagenet_dataset
from get_dataloaders import get_dataloaders
from utils import get_equally_distributed_subset
from tqdm import tqdm

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
                                pkmn_net, dice_net, interpolated_net,
                                activations_output_path, num_points_per_dataset):

    # equally distribute the data -- we have three subsets with an equal number of points
    print('[DEBUG] Preparing equally distributed subsets')
    pkmn_subset, dice_subset, small_imagenet_subset = get_equally_distributed_subset(pokemon_dataset, dice_dataset, small_imagenet_dataset, num_points_per_dataset)

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

    # Put all models into a list
    models = [ pkmn_net , dice_net, interpolated_net ] 

    print('[DEBUG] Starting saving models activations')
    for model, activation_path in zip(models, activations_output_path):
      create_hooks_and_test_model(model, dataloaders, activation_path)

def create_hooks(model, activations_path, skip_modules=('fc')):
  # Hook factory
  def get_hook(name, activations_path):
    def hook(module, inp, out):
      torch.save(out.detach(), f"{activations_path}/{name}.torch") # detach so we don’t keep the computation graph
      handle.remove()

    return hook
  
  counter = 0
  for name, module in model.named_modules():
      if name in skip_modules: # we don't care about the head
        continue

      handle = module.register_forward_hook(get_hook(name, activations_path))
      counter += 1

# multiple dataloaders test
def create_hooks_and_test_model(model, test_loaders, model_path):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  with torch.no_grad():

    for i, test_loader in enumerate(test_loaders):
      print(f'[DEBUG] Started dataloader {i}')
      activations_path = model_path + '/loader_' + str(i)

      loop = tqdm(enumerate(test_loader), total=len(test_loader), desc="Computing Activations", leave=True)
      for batch_idx, (test_inputs, _) in loop: # we don't need the labels at all
        activations_batch_path = activations_path + '/batch_' + str(batch_idx)
        Path(activations_batch_path).mkdir(parents=True, exist_ok=True) # create directory

        create_hooks(model, activations_batch_path)
        test_inputs = test_inputs.to(device)

        # preds
        test_outputs = model(test_inputs)