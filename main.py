import get_datasets
import torch
import torchvision
from torch import nn, optim
from get_datasets import get_small_imagenet_dataset
from get_dataloaders import get_dataloaders
from utils import seed_run
from resnet_model import finetune_resnet, load_resnet_from_weights, interpolate_resnet_models, test_resnet, averaging_resnet_models
from crosscoder_dataset_utils import create_crosscoder_dataset
from CrossCoderDataset import CrossCoderDataset
from CrossCoder import CrossCoder
from analysis import analyze_crosscoder
import os
from pcb_merge import create_pcb_merge, pcb_grid_search


MERGE_WITH_INTERPOLATION    = False     # Wheter we merge models with interpolation technique
MERGE_WITH_PARAM_AVG        = True      # Wheter we merge models with parameter averaging technique

#### CONFIG ####
TRAIN                       = False # Do the actual trainining/interpolation or get the already-finetuned/interpolated versions
TEST                        = False # Test models or skip tests
CREATE_CROSSCODER_DATASET   = False # Create the dataset or use the already-created one
TRAIN_CROSSCODER            = False # Train or use the already-trained crosscoder
VAL_CROSSCODER              = False # Crosscoder validation
TEST_CROSSCODER             = False # Test/analysis crosscoder
### 
CREATE_PCB_MERGE            = True # Creation of new merged model using Parameter Competition Balancing for Model Merging (https://arxiv.org/pdf/2410.02396)
PCB_GRID_SEARCH             = True # Grid search on pcb_ratio
TEST_PCB                    = True # Test the new merged model on the two original datasets
CREATE_PCB_DATASET          = True # Creates the activations for the PCB merging technique

PROJECT_NAME                = 'deep_learning'

# Same across all the fine-tuned models
BATCH_SIZE      = 32
NUM_EPOCHS      = 10
TRAINING_SIZE   = 0.7
VALIDATION_SIZE = 0.2
TEST_SIZE       = 0.1
LR              = 0.0001
MOMENTUM        = 0.9

PKMN_WEIGHTS_PATH   = './pokemon_resnet/model_weights.pth'
PKMN_NUM_CLASSES    = 5 # num of pokemons in the dataset

DICE_WEIGHTS_PATH   ='./dice_resnet/model_weights.pth'
DICE_NUM_CLASSES    = 6 # num of dice in the dataset

### Wandb data
PKMN_CLASS_NAMES        = ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle']
PKMN_EXPERIMENT_NAME    = 'pokemon_resnet_v3'
PKMN_DESCRIPTION        = "Pokemon ResNet finetuning run after refactor"
PKMN_WANDB_CONFIG       = {
    "learning_rate": LR,
    "momentum": MOMENTUM,
    "architecture": "ResNet",
    "dataset": "manuel-yao/pokemon-keras-community",
    "epochs": NUM_EPOCHS,
}

DICE_CLASS_NAMES    = ['d10','d12','d20','d4','d6','d8']
DICE_EXPERIMENT_NAME= 'dice_resnet_v3'
DICE_DESCRIPTION    = "Dice ResNet finetuning run after refactor"
DICE_WANDB_CONFIG   = {
    "learning_rate": LR,
    "momentum": MOMENTUM,
    "architecture": "ResNet",
    "dataset": "ucffool/dice-d4-d6-d8-d10-d12-d20-images", # from Kaggle
    "epochs": NUM_EPOCHS
}

# Interpolation Config
OUT_INTERPOLATED_DIR                 = './interpolated'
DEFAULT_INTERPOLATED_RESNET_HEAD     = 'model_weights_v3.pth'
PKMN_INTERPOLATED_HEAD               = 'model_weights_v3_pkmn.pth'
DICE_INTERPOLATED_HEAD               = 'model_weights_v3_dice.pth'

# Parameter Averaging Config
OUT_PARAM_AVERAGE_DIR           = './parameter_avg'
DEFAULT_AVG_RESNET_HEAD         = 'parameter_avg_weights.pth'
PKMN_AVG_HEAD                   = 'parameter_avg_pkmn.pth'
DICE_AVG_HEAD                   = 'parameter_avg_weights_dice.pth'

#### CrossCoder Dataset Config
RESNET_BATCH_SIZE = 1 # number of images given to our resnets to compute a single set of activations
# Number of datapoints generated per model (dimension [n_models, n_crosscoder_datapoints, n_activations]) -- we do not take everything because otherwise we would have terabytes of data
N_CROSSCODER_DATAPOINTS = 500
REGEX_ACTIVATIONS = '^layer4$'  # Regex to get only the big sequential layers inside the resnets. We were not able to get the whole activations due to computational/memory power limit

# Crosscoder dataset with Interpolation merging technique
ACTIVATIONS_INTERPOLATED_PATH = './interpolated_activations_layer4'
MODEL_ACTIVATIONS_INTERPOLATED_PATHS = [f'{ACTIVATIONS_INTERPOLATED_PATH}/pokemon', f'{ACTIVATIONS_INTERPOLATED_PATH}/dice', f'{ACTIVATIONS_INTERPOLATED_PATH}/interpolated']

# Crosscoder dataset with Parameter Averaging merging technique
ACTIVATIONS_PARAM_AVG_PATH = './parameter_avg_activations_layer4'
MODEL_ACTIVATIONS_PARAM_AVG_PATHS = [f'{ACTIVATIONS_PARAM_AVG_PATH}/pokemon', f'{ACTIVATIONS_PARAM_AVG_PATH}/dice', f'{ACTIVATIONS_PARAM_AVG_PATH}/interpolated']

### CrossCoder Model Config
BATCH_SIZE_CROSS = 64 # crosscoder batch -- number of datapoints fetched by the crosscoder dataloader
NUM_EPOCHS_CROSS = 150
LATENT_DIM = 900
TRAINING_SIZE_CROSS   = 0.7
VALIDATION_SIZE_CROSS = 0.1 # smaller validation because we just have to tune the latent_dim hyperparam
TEST_SIZE_CROSS       = 0.2
LR_CROSS = 0.01   # max_lr in OneCycleLR
LAMBDA_SPARSE = 2 # TODO boh

CROSS_INTERPOLATED_WEIGHTS_PATH = './crosscoder/interpolated/model_weights.pth'
CROSS_PARAM_AVG_WEIGHTS_PATH = './crosscoder/parameter_avg/model_weights.pth'
### Second experiment -- using a different merging technique: Parameter Competition Balancing for Model Merging (https://arxiv.org/pdf/2410.02396)
PCB_WEIGHTS_PATH   ='./pcb_resnet/'

if __name__ == '__main__':
    assert any(MERGE_CONFIG_LIST), f"ERROR: at least one merging technique must be true, see MERGE_CONFIG_LIST -> {MERGE_CONFIG_LIST}"
    seed_run() # We seed the run to replicate the results

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tags    = ['resnet', 'classification']

    if (TRAIN or TEST or CREATE_CROSSCODER_DATASET or TEST_PCB): # If we don't care about the dice/pokemon datasets, we just skip
        ##### Pokemon Finetuning ####
        ### Prepare data
        print('[DEBUG] Loading Pokemon Dataset')
        pokemon_dataset = get_datasets.get_pokemon_dataset()
        
        pkmn_train_loader, pkmn_val_loader, pkmn_test_loader = get_dataloaders(pokemon_dataset, BATCH_SIZE, TRAINING_SIZE, VALIDATION_SIZE, TEST_SIZE)

        ### Prepare Resnet

        if TRAIN:
            pkmn_net = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
            num_features = pkmn_net.fc.in_features
            pkmn_net.fc = nn.Linear(num_features, PKMN_NUM_CLASSES)
            pkmn_net = pkmn_net.to(device)

            ### Prepare training
            loss_func = nn.CrossEntropyLoss()
            # we optimize EVERY layer because we want to show the difference of the two finetuned versions of resnet
            optimizer = optim.SGD(pkmn_net.parameters(), lr=LR, momentum=MOMENTUM)

            ### Finetuning
            print("[DEBUG] Starting Pokemon finetuning")
            finetune_resnet(pkmn_net, pkmn_train_loader, pkmn_val_loader,
                                loss_func, optimizer, PKMN_WANDB_CONFIG,
                                NUM_EPOCHS,
                                PKMN_CLASS_NAMES, # ie: ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle'],
                                PKMN_WEIGHTS_PATH, # ie: './pokemon_resnet/model_weights.pth',
                                PKMN_EXPERIMENT_NAME, # ie:'pokemon_resnet_v2',
                                PKMN_DESCRIPTION, # ie: "Pokemon ResNet finetuning run"
                                project_name='deep_learning',
                                tags=['resnet', 'classification'],
            )
        else:
            pkmn_net = load_resnet_from_weights(PKMN_WEIGHTS_PATH, PKMN_NUM_CLASSES)

        ### Test 
        if TEST:
            pkmn_accuracy = test_resnet(pkmn_net, pkmn_test_loader)
            print(f"[RESULT] Pokemon Resnet tested with an accuracy equal to {pkmn_accuracy:.4f}")

        ##### End Pokemon Finetuning ####

        ##### Dice Finetuning ####

        ### Prepare data
        print('[DEBUG] Loading Dice Dataset')
        dice_dataset = get_datasets.get_dice_dataset()

        dice_train_loader, dice_val_loader, dice_test_loader = get_dataloaders(dice_dataset, BATCH_SIZE, TRAINING_SIZE, VALIDATION_SIZE, TEST_SIZE)

        ### Prepare Resnet
        if TRAIN:
            dice_net = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
            
            num_features = dice_net.fc.in_features
            dice_net.fc = nn.Linear(num_features, DICE_NUM_CLASSES)
            dice_net = dice_net.to(device)

            ### Prepare training
            loss_func = nn.CrossEntropyLoss()
            # we optimize EVERY layer because we want to show the difference of the two finetuned versions of resnet
            optimizer = optim.SGD(dice_net.parameters(), lr=LR, momentum=MOMENTUM)

            ### Finetuning
            print("[DEBUG] Starting Dice finetuning")
            finetune_resnet(dice_net, dice_train_loader, dice_val_loader,
                                loss_func, optimizer, DICE_WANDB_CONFIG,
                                NUM_EPOCHS,
                                DICE_CLASS_NAMES,
                                DICE_WEIGHTS_PATH,
                                DICE_EXPERIMENT_NAME,
                                DICE_DESCRIPTION,
                                project_name='deep_learning',
                                tags=['resnet', 'classification'],
            )
        else:
            dice_net = load_resnet_from_weights(DICE_WEIGHTS_PATH, DICE_NUM_CLASSES)

        ### Test 
        if TEST:
            dice_accuracy = test_resnet(dice_net, dice_test_loader)
            print(f"[RESULT] Dice Resnet tested with an accuracy equal to {dice_accuracy:.4f}")

        ##### End Dice Finetuning #####

        ##### Default original Resnet Tests #####
        ### Default pretrained ResNet with modified head on Pokemon
        resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device)
        resnet.fc.weight = pkmn_net.fc.weight
        resnet.fc.bias   = pkmn_net.fc.bias

        if TEST:
            resnet_pkmn_accuracy = test_resnet(resnet, pkmn_test_loader)
            print(f"[RESULT] The original Resnet tested with an accuracy equal to {resnet_pkmn_accuracy:.4f} on the Pokemon Dataset")
        ### Default pretrained ResNet with modified head on Pokemon

        ### Default pretrained ResNet with modified head on Dice
        resnet.fc.weight = dice_net.fc.weight
        resnet.fc.bias   = dice_net.fc.bias

        if TEST:
            resnet_pkmn_accuracy = test_resnet(resnet, dice_test_loader)
            print(f"[RESULT] The original Resnet tested with an accuracy equal to {resnet_pkmn_accuracy:.4f} on the Dice Dataset")
        ##### End of default Resnet Tests #####

        ##### Pokemon-Dice Merging and Tests #####

        
        if TRAIN:
            
            if MERGE_WITH_INTERPOLATION:
                print('[DEBUG] Starting Interpolation')
                interpolate_resnet_models(
                    PKMN_WEIGHTS_PATH, DICE_WEIGHTS_PATH,
                    OUT_INTERPOLATED_DIR, DEFAULT_INTERPOLATED_RESNET_HEAD,
                    PKMN_INTERPOLATED_HEAD, DICE_INTERPOLATED_HEAD,
                    interpolation_weight=0.5
                )
                
            if MERGE_WITH_PARAM_AVG:
                print('[DEBUG] Starting parameter averaging')
                averaging_resnet_models(
                    PKMN_WEIGHTS_PATH,
                    DICE_WEIGHTS_PATH,
                    OUT_PARAM_AVERAGE_DIR,
                    out_head1_name=PKMN_AVG_HEAD,
                    out_head2_name=DICE_AVG_HEAD,
                    out_name=DEFAULT_AVG_RESNET_HEAD
                )

            
        
        ### Test Interpolated Models on Original Datasets

        if TEST:
            
            if MERGE_WITH_INTERPOLATION:
                # Load interpolated models
                pkmn_interpolated_model_path = OUT_INTERPOLATED_DIR+'/'+PKMN_INTERPOLATED_HEAD
                pkmn_interpolated = load_resnet_from_weights(pkmn_interpolated_model_path, PKMN_NUM_CLASSES)

                dice_interpolated_model_path = OUT_INTERPOLATED_DIR+'/'+DICE_INTERPOLATED_HEAD
                dice_interpolated = load_resnet_from_weights(dice_interpolated_model_path, DICE_NUM_CLASSES)
                
                # Test on pkmn dataset
                interpolated_pkmn_accuracy_on_pkmn = test_resnet(pkmn_interpolated, pkmn_test_loader)
                print(f"[RESULT INTERPOLATED] Interpolated Resnet tested with an accuracy equal to {interpolated_pkmn_accuracy_on_pkmn:.4f} on the Pokemon Dataset")
                # Test on dice dataset
                interpolated_dice_accuracy_on_dice = test_resnet(dice_interpolated, dice_test_loader)
                print(f"[RESULT INTERPOLATED] Interpolated Resnet tested with an accuracy equal to {interpolated_dice_accuracy_on_dice:.4f} on the Dice Dataset")
            
            
            if MERGE_WITH_PARAM_AVG:
                # Load parameter average models
                pkmn_avg_model_path = os.path.join(OUT_PARAM_AVERAGE_DIR, PKMN_AVG_HEAD)
                dice_avg_model_path = os.path.join(OUT_PARAM_AVERAGE_DIR, DICE_AVG_HEAD)

                pkmn_avg = load_resnet_from_weights(pkmn_avg_model_path, PKMN_NUM_CLASSES)
                dice_avg = load_resnet_from_weights(dice_avg_model_path, DICE_NUM_CLASSES)
                
                # Test on pkmn dataset
                acc_pkmn_on_pkmn = test_resnet(pkmn_avg, pkmn_test_loader)
                print(f"[RESULT AVG] Parameter Avg Resnet tested on Pokémon: {acc_pkmn_on_pkmn:.4f}")

                # Test on dice dataset
                acc_dice_on_dice = test_resnet(dice_avg, dice_test_loader)
                print(f"[RESULT AVG] Parameter Avg Resnet tested on Dice: {acc_dice_on_dice:.4f}")

        ##### End Pokemon-Dice Merging and Tests #####

    ##### Crosscoder Dataset Creation #####
    if CREATE_CROSSCODER_DATASET:
        # The small imagenet_dataset contains a small subset of the original imagenet, which should activate the "old" features, learned during the original training
        small_imagenet_dataset = get_small_imagenet_dataset()
        
        if MERGE_WITH_INTERPOLATION:
            interpolated_model_path = OUT_INTERPOLATED_DIR+'/'+DEFAULT_INTERPOLATED_RESNET_HEAD
            interpolated_net = load_resnet_from_weights(interpolated_model_path)
            
            create_crosscoder_dataset(pokemon_dataset, dice_dataset, small_imagenet_dataset, RESNET_BATCH_SIZE,\
                        [pkmn_net, dice_net, interpolated_net],
                        MODEL_ACTIVATIONS_INTERPOLATED_PATHS, N_CROSSCODER_DATAPOINTS, REGEX_ACTIVATIONS)
        if MERGE_WITH_PARAM_AVG:
            parameter_avg_model_path = OUT_PARAM_AVERAGE_DIR+'/'+DEFAULT_AVG_RESNET_HEAD
            parameter_avg_net = load_resnet_from_weights(parameter_avg_model_path)
            
            create_crosscoder_dataset(pokemon_dataset, dice_dataset, small_imagenet_dataset, RESNET_BATCH_SIZE,\
                        [pkmn_net, dice_net, parameter_avg_net],
                        MODEL_ACTIVATIONS_PARAM_AVG_PATHS, N_CROSSCODER_DATAPOINTS, REGEX_ACTIVATIONS)
    
    # Initialize variables for future use
    cross_interpolated_train_loader, cross_interpolated_val_loader, cross_interpolated_test_loader = None, None, None
    cross_param_avg_train_loader, cross_param_avg_val_loader, cross_param_avg_test_loader = None, None, None
    
    cross_interpolated_weights_path = None
    cross_param_avg_weights_path = None
    
    total_steps_interpolated    = None
    total_steps_param_avg       = None
    
    # Load crosscoder dataset
    if MERGE_WITH_INTERPOLATION:
        crosscoder_dataset = CrossCoderDataset(ACTIVATIONS_INTERPOLATED_PATH) # [n_models, n_crosscoder_datapoints, n_activations]
        n_activations = crosscoder_dataset.get_n_activations() 
        cross_interpolated_train_loader, cross_interpolated_val_loader, cross_interpolated_test_loader = get_dataloaders(crosscoder_dataset, BATCH_SIZE_CROSS, TRAINING_SIZE_CROSS, VALIDATION_SIZE_CROSS, TEST_SIZE_CROSS)
        
        # Load crosscoder weights
        cross_interpolated_weights_path = CROSS_INTERPOLATED_WEIGHTS_PATH
        total_steps_interpolated = len(cross_interpolated_train_loader) # total steps per epoch
    
    if MERGE_WITH_PARAM_AVG:
        crosscoder_dataset = CrossCoderDataset(ACTIVATIONS_PARAM_AVG_PATH) # [n_models, n_crosscoder_datapoints, n_activations]
        n_activations = crosscoder_dataset.get_n_activations() 
        cross_param_avg_train_loader, cross_param_avg_val_loader, cross_param_avg_test_loader = get_dataloaders(crosscoder_dataset, BATCH_SIZE_CROSS, TRAINING_SIZE_CROSS, VALIDATION_SIZE_CROSS, TEST_SIZE_CROSS)
        
        # Load crosscoder weights
        cross_param_avg_weights_path = CROSS_PARAM_AVG_WEIGHTS_PATH
        total_steps_param_avg = len(cross_param_avg_train_loader)
    
    if TRAIN_CROSSCODER:
        # print(f"n_activations: {n_activations}")
        if MERGE_WITH_INTERPOLATION:
            crosscoder = CrossCoder(LATENT_DIM, n_activations, LAMBDA_SPARSE, total_steps_interpolated)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            crosscoder.to(device)

            crosscoder.train_cross(cross_interpolated_train_loader, NUM_EPOCHS_CROSS, LR_CROSS)

            dirpath = os.path.dirname(cross_interpolated_weights_path)
            
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            # save the model’s state dict
            torch.save(crosscoder.state_dict(), cross_interpolated_weights_path)
            
        if MERGE_WITH_PARAM_AVG:
            crosscoder = CrossCoder(LATENT_DIM, n_activations, LAMBDA_SPARSE, total_steps_param_avg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            crosscoder.to(device)

            crosscoder.train_cross(cross_param_avg_train_loader, NUM_EPOCHS_CROSS, LR_CROSS)

            dirpath = os.path.dirname(cross_param_avg_weights_path)
            
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            # save the model’s state dict
            torch.save(crosscoder.state_dict(), cross_param_avg_weights_path)

    
    if VAL_CROSSCODER or TEST_CROSSCODER:
        
        if MERGE_WITH_INTERPOLATION:
            crosscoder = CrossCoder(LATENT_DIM, n_activations, LAMBDA_SPARSE, total_steps_interpolated)
            crosscoder = crosscoder.to(device)
            crosscoder.load_state_dict(torch.load(cross_interpolated_weights_path, weights_only=True))

            if VAL_CROSSCODER:
                crosscoder.val_cross(cross_interpolated_val_loader)

            if TEST_CROSSCODER:
                analyze_crosscoder(crosscoder, cross_interpolated_test_loader)
                
        if MERGE_WITH_PARAM_AVG:
            crosscoder = CrossCoder(LATENT_DIM, n_activations, LAMBDA_SPARSE, total_steps_param_avg)
            crosscoder = crosscoder.to(device)
            crosscoder.load_state_dict(torch.load(cross_param_avg_weights_path, weights_only=True))

            if VAL_CROSSCODER:
                crosscoder.val_cross(cross_param_avg_val_loader)

            if TEST_CROSSCODER:
                analyze_crosscoder(crosscoder, cross_param_avg_test_loader)
                
            # Validate and Test Crosscoder
    # The test is done through an analysis that shows relevant plots
    if VAL_CROSSCODER or TEST_CROSSCODER: 
        crosscoder = CrossCoder(LATENT_DIM, n_activations, LAMBDA_SPARSE, total_steps)
        crosscoder = crosscoder.to(device)
        crosscoder.load_state_dict(torch.load(CROSS_WEIGHTS_PATH, weights_only=True))

        if VAL_CROSSCODER:
            crosscoder.val_cross(cross_val_loader)

        if TEST_CROSSCODER:
           analyze_crosscoder(crosscoder, cross_test_loader)

    if CREATE_PCB_MERGE:
        if PCB_GRID_SEARCH:
            pcb_grid_search(PKMN_WEIGHTS_PATH, DICE_WEIGHTS_PATH, PCB_WEIGHTS_PATH, PKMN_HEAD, DICE_HEAD, PKMN_NUM_CLASSES, DICE_NUM_CLASSES, pkmn_val_loader, dice_val_loader)

    if CREATE_PCB_DATASET:
        pass
        
    if TEST_PCB:
        best_pcb_ratio = 0.7138571739196777 # from grid search -- average acc 0.9391 on pokemon/dice
