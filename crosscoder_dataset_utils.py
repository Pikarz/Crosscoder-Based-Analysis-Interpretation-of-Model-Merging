import torch
from pathlib import Path
from get_dataloaders import get_dataloaders
from utils import get_equally_distributed_subset
from tqdm import tqdm
import os
import re

def create_crosscoder_dataset(pokemon_dataset, dice_dataset, small_imagenet_dataset, batch_size,
                            models, activations_output_paths, crosscoder_datapoints, 
                            regex_activations, n_models=3):

    print('[INFO] Preparing equally distributed subsets')
    pkmn_subset, dice_subset, small_imagenet_subset = get_equally_distributed_subset(
        pokemon_dataset, dice_dataset, small_imagenet_dataset, n_models, crosscoder_datapoints
    )

    # print(f'[INFO] Dataset Subsets Size:\n \
    #           Pokemon: {len(pkmn_subset)}\n \
    #           Dice: {len(dice_subset)}\n \
    #           Small Imagenet: {len(small_imagenet_subset)}')

    # Get the corresponding dataloaders
    pkmn_dataloader, _, _           = get_dataloaders(pkmn_subset,           batch_size, 1.0, 0.0, 0.0)
    dice_dataloader, _, _           = get_dataloaders(dice_subset,           batch_size, 1.0, 0.0, 0.0)
    small_imagenet_dataloader, _, _ = get_dataloaders(small_imagenet_subset, batch_size, 1.0, 0.0, 0.0)

    # Put dataloaders into a list
    dataloaders = [pkmn_dataloader, dice_dataloader, small_imagenet_dataloader]

    print('[INFO] Starting saving models activations')
    for model, activation_path in zip(models, activations_output_paths):
        create_hooks_and_test_model(model, dataloaders, activation_path, regex_activations)


def create_hooks_and_test_model(model, data_loaders, model_path, regex_activations):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    count = 0

    with torch.no_grad():
        for data_loader in data_loaders:
            loop = tqdm(enumerate(data_loader), total=len(data_loader), desc="Computing Activations", leave=True)
            
            for batch_idx, (test_inputs, _) in loop:
                # Create the batch directory
                activations_path = model_path + '/activations_' + str(count)
                count += 1
                Path(activations_path).mkdir(parents=True, exist_ok=True)
                
                # Register hooks that self-remove after triggering (like original)
                create_hooks(model, activations_path, regex_activations)
                
                # Forward pass
                test_inputs = test_inputs.to(device)
                test_outputs = model(test_inputs)


def create_hooks(model, activations_path, regex_activations):
    """
    Create hooks that save activations and remove themselves after triggering.
    This maintains the original behavior where each hook fires only once per forward pass.
    """
    for name, module in model.named_modules():
        def make_hook(name, activations_path):
            def hook(module, inp, out):
                if bool(re.search(regex_activations, name)):
                    # Save activation to disk
                    save_path = f"{activations_path}/{name}.torch"
                    torch.save(out.detach().cpu(), save_path)
                    # Remove the hook immediately after use (original behavior)
                    hook.handle.remove()
            return hook
        
        hook_func = make_hook(name, activations_path)
        handle = module.register_forward_hook(hook_func)
        hook_func.handle = handle  # Store handle in the hook function


def get_model_activations(path, activations_tensors=[], activations=torch.as_tensor([])):
    """
    Recursively load activations from saved files.
    """
    if os.path.isfile(path) and path.endswith('.torch'):
        # We found a tensor file - load and flatten it
        activations_layer = torch.load(path).cpu().view(-1)
        activations = torch.cat([activations, activations_layer])
        return activations_tensors, activations
    
    # Check if path is a directory
    if not os.path.isdir(path):
        return activations_tensors, activations
    
    content = os.listdir(path) 

    for file in content:
        subpath = os.path.join(path, file)
        activations_tensors, activations = get_model_activations(subpath, activations_tensors, activations)

    if activations.size()[0] > 0:  # if there's at least a .torch file in the current directory
        activations_tensors.append(activations)
        activations = torch.as_tensor([])
    
    return activations_tensors, activations