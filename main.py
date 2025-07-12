import os
import torch
import torchvision
from torch import nn, optim

from get_datasets import get_dice_dataset, get_pokemon_dataset, get_small_imagenet_dataset
from get_dataloaders import get_dataloaders
from utils import seed_run, train_crosscoder_and_save_weights, validate_crosscoder
from resnet_model import finetune_resnet, load_resnet_from_weights, interpolate_resnet_models, test_resnet, averaging_resnet_models
from crosscoder_dataset_utils import create_crosscoder_dataset, save_merge_activations
from pcb_merge import create_pcb_merge, pcb_grid_search
from CrossCoderDataset import prepare_crosscoder_suite

#### CONFIG ####
TRAIN_BASELINE              = False # Trains the original networks -- pokemon/dice resnets
TEST_BASELINE               = False # Tests the original networks -- pokemon/dice resnets
TEST_MERGES                 = False  # Test merged models or skip
PCB_GRID_SEARCH             = False  # Grid search on pcb_ratio
CREATE_CROSSCODER_DATASET   = False  # Create the dataset or use the already-created one

### MERGING CONFIG ###
CREATE_INTERPOLATION_MERGE = False      # Whether we merge models with the interpolation technique
CREATE_AVG_MERGE           = False      # Whether we merge models with the parameter averaging technique
CREATE_PCB_MERGE           = False      # Whether we merge models using the Parameter Competition Balancing for Model Merging (https://arxiv.org/pdf/2410.02396)

# Actually compute the activations for the merging techniques, which will be fed to the crosscoder
COMPUTE_INTERPOLATION_ACTS  = False 
COMPUTE_AVG_ACTS            = False
COMPUTE_PCB_ACTS            = False

# Load the corresponding datasets (train, val, test) in memory -- We suggest to try one model at a time because it is GPU-intensive
LOAD_INTERPOLATION_DS       = True
LOAD_AVG_DS                 = True
LOAD_PCB_DS                 = True

TEST_INTERPOLATION = True      # Whether we test the interpolated model
TEST_AVG           = True      # Whether we test the soup model (average parameters)
TEST_PCB           = True      # Whether we test the pcb model (https://arxiv.org/pdf/2410.02396)

### CROSSCODER CONFIG ###
TRAIN_CROSSCODER            = False  # Train or use the already-trained crosscoder
VAL_CROSSCODER              = True  # Crosscoder validation
TEST_CROSSCODER             = True  # Test/analysis crosscoder

### OTHER PARAMS ###
PROJECT_NAME                = 'deep_learning'

# Same across all the fine-tuned models
BATCH_SIZE      = 32
NUM_EPOCHS      = 10
TRAINING_SIZE   = 0.7
VALIDATION_SIZE = 0.2
TEST_SIZE       = 0.1
LR              = 0.0001
MOMENTUM        = 0.9

PKMN_WEIGHTS_PATH   = './models/pokemon_resnet/model_weights.pth'
PKMN_NUM_CLASSES    = 5 # num of pokemons in the dataset

DICE_WEIGHTS_PATH   ='./models/dice_resnet/model_weights.pth'
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
OUT_INTERPOLATED_DIR                 = './models/interpolated_resnet'
DEFAULT_INTERPOLATED_RESNET_HEAD     = 'model_weights_v3.pth'
PKMN_INTERPOLATED_HEAD               = 'model_weights_v3_pkmn.pth'
DICE_INTERPOLATED_HEAD               = 'model_weights_v3_dice.pth'

# Parameter Averaging Config
OUT_PARAM_AVERAGE_DIR           = './models/parameter_avg_resnet'
DEFAULT_AVG_RESNET_HEAD         = 'parameter_avg_weights.pth'
PKMN_AVG_HEAD                   = 'parameter_avg_weights_pkmn.pth'
DICE_AVG_HEAD                   = 'parameter_avg_weights_dice.pth'

# Using a different merging technique: Parameter Competition Balancing for Model Merging (https://arxiv.org/pdf/2410.02396)
OUT_PCB_DIR                     ='./models/pcb_resnet/'
PKMN_PCB_HEAD                   = 'pcb_weights_pkmn.pth'
DICE_PCB_HEAD                   = 'pcb_weights_dice.pth'

#### CrossCoder Dataset Config
RESNET_BATCH_SIZE = 8 # number of images given to our resnets to compute a single set of activations
# Number of datapoints generated per model (dimension [n_models, n_crosscoder_datapoints, n_activations]) -- we do not take everything because otherwise we would have terabytes of data
N_CROSSCODER_DATAPOINTS = 500
REGEX_ACTIVATIONS = '^layer4$'  # Regex to get only the big sequential layers inside the resnets. We were not able to get the whole activations due to computational/memory power limit
ACTIVATIONS_POKEMON_PATH = './activations_layer4/pokemon'
ACTIVATIONS_DICE_PATH = './activations_layer4/dice'

# Crosscoder dataset with Interpolation merging technique
ACTIVATIONS_INTERPOLATED_PATH = './activations_layer4/interpolated'

# Crosscoder dataset with Parameter Averaging merging technique
ACTIVATIONS_PARAM_AVG_PATH = './activations_layer4/average'

# Crosscoder dataset with PCB merging technique
ACTIVATIONS_PCB_PATH = './activations_layer4/pcb'

### CrossCoder Model Config
BATCH_SIZE_CROSS = 64 # crosscoder batch -- number of datapoints fetched by the crosscoder dataloader
NUM_EPOCHS_CROSS = 1
LATENT_DIM = 900
TRAINING_SIZE_CROSS   = 0.7
VALIDATION_SIZE_CROSS = 0.1 # smaller validation because we just have to tune the latent_dim hyperparam
TEST_SIZE_CROSS       = 0.2

LR_CROSS = 0.01   # max_lr in OneCycleLR
LAMBDA_SPARSE = 2 
CROSSCODER_WANDB_CONFIG   = {
    "Lambda Sparse": LAMBDA_SPARSE,
    "lr": LR_CROSS,
    "latent_dim": LATENT_DIM,
    "architecture": "CrossCoder",
    "epochs": NUM_EPOCHS_CROSS
}

CROSS_INTERPOLATED_WEIGHTS_PATH = './models/crosscoder/interpolated/model_weights.pth'
CROSS_PARAM_AVG_WEIGHTS_PATH = './models/crosscoder/parameter_avg/model_weights.pth'
CROSS_PCB_WEIGHTS_PATH = './models/crosscoder/pcb/model_weights.pth'


if __name__ == '__main__':
    seed_run() # We seed the run to replicate the results

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tags    = ['resnet', 'classification']

    ### Prepare Pokemon data
    print('[DEBUG] Loading Pokemon Dataset')
    pokemon_dataset = get_pokemon_dataset()
    pkmn_train_loader, pkmn_val_loader, pkmn_test_loader = get_dataloaders(pokemon_dataset, BATCH_SIZE, TRAINING_SIZE, VALIDATION_SIZE, TEST_SIZE)

    ### Prepare Dice data
    print('[DEBUG] Loading Dice Dataset')
    dice_dataset = get_dice_dataset()
    dice_train_loader, dice_val_loader, dice_test_loader = get_dataloaders(dice_dataset, BATCH_SIZE, TRAINING_SIZE, VALIDATION_SIZE, TEST_SIZE)

    if (TRAIN_BASELINE or TEST_BASELINE): # We fine-tune or test the baseline ResNet models
        ##### Pokemon Finetuning ####
        if TRAIN_BASELINE: # Baseline method
            ### Prepare fine-tuned Resnet
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
                                project_name=PROJECT_NAME,
                                tags=['resnet', 'classification'],
            )
        else:
            pkmn_net = load_resnet_from_weights(PKMN_WEIGHTS_PATH, PKMN_NUM_CLASSES)

        ### Test 
        if TEST_BASELINE:
            pkmn_accuracy = test_resnet(pkmn_net, pkmn_test_loader)
            print(f"[RESULT] Pokemon Resnet tested with an accuracy equal to {pkmn_accuracy:.4f}")

        ##### End Pokemon Finetuning ####

        ##### Dice Finetuning ####

        ### Prepare Resnet
        if TRAIN_BASELINE:
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
                                project_name=PROJECT_NAME,
                                tags=['resnet', 'classification'],
            )
        else:
            dice_net = load_resnet_from_weights(DICE_WEIGHTS_PATH, DICE_NUM_CLASSES)

        ### Test 
        if TEST_BASELINE:
            dice_accuracy = test_resnet(dice_net, dice_test_loader)
            print(f"[RESULT] Dice Resnet tested with an accuracy equal to {dice_accuracy:.4f}")

        ##### End Dice Finetuning #####

        ##### Default original Resnet Tests #####
        ### Default pretrained ResNet with modified head on Pokemon
        resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device)
        resnet.fc.weight = pkmn_net.fc.weight
        resnet.fc.bias   = pkmn_net.fc.bias

        if TEST_BASELINE:
            resnet_pkmn_accuracy = test_resnet(resnet, pkmn_test_loader)
            print(f"[RESULT] The original Resnet tested with an accuracy equal to {resnet_pkmn_accuracy:.4f} on the Pokemon Dataset")
        ### Default pretrained ResNet with modified head on Pokemon

        ### Default pretrained ResNet with modified head on Dice
        resnet.fc.weight = dice_net.fc.weight
        resnet.fc.bias   = dice_net.fc.bias

        if TEST_BASELINE:
            resnet_pkmn_accuracy = test_resnet(resnet, dice_test_loader)
            print(f"[RESULT] The original Resnet tested with an accuracy equal to {resnet_pkmn_accuracy:.4f} on the Dice Dataset")
        ##### End of default Resnet Tests #####

    ###### Using different merging techniques ######
    #### With the PCB_GRID_SEARCH, we also perform the grid search for the parameter pcb_ratio for the PCB technique ####
    if (CREATE_INTERPOLATION_MERGE or CREATE_AVG_MERGE or CREATE_PCB_MERGE or PCB_GRID_SEARCH): 
        
        ### Interpolation
        if CREATE_INTERPOLATION_MERGE:
            print('[DEBUG] Starting Interpolation')
            interpolate_resnet_models(
                PKMN_WEIGHTS_PATH, DICE_WEIGHTS_PATH,
                OUT_INTERPOLATED_DIR, DEFAULT_INTERPOLATED_RESNET_HEAD,
                PKMN_INTERPOLATED_HEAD, DICE_INTERPOLATED_HEAD,
                interpolation_weight=0.5
            )
            
        # Parameter Soup (averaging)
        if CREATE_AVG_MERGE:
            print('[DEBUG] Starting Parameter Averaging')
            averaging_resnet_models(
                PKMN_WEIGHTS_PATH,
                DICE_WEIGHTS_PATH,
                OUT_PARAM_AVERAGE_DIR,
                out_head1_name=PKMN_AVG_HEAD,
                out_head2_name=DICE_AVG_HEAD,
                out_name=DEFAULT_AVG_RESNET_HEAD
            )
            
        # Parameter Competition Balancing 
        if (CREATE_PCB_MERGE or PCB_GRID_SEARCH):
            print('[DEBUG] Starting PCB merge')
            if PCB_GRID_SEARCH:
                pcb_grid_search(
                    pkmn_weights_path=PKMN_WEIGHTS_PATH,
                    dice_weights_path=DICE_WEIGHTS_PATH,
                    pcb_weights_path=OUT_PCB_DIR,
                    pkmn_head=PKMN_PCB_HEAD,
                    dice_head=DICE_PCB_HEAD,
                    pkmn_num_classes=PKMN_NUM_CLASSES,
                    dice_num_classes=DICE_NUM_CLASSES,
                    pkmn_val_loader=pkmn_val_loader,
                    dice_val_loader=dice_val_loader)
            else:
                best_pcb_ratio = 0.8666 # from grid search -- average acc 0.8156 on pokemon/dice
                create_pcb_merge(
                    PKMN_WEIGHTS_PATH,
                    DICE_WEIGHTS_PATH,
                    OUT_PCB_DIR,
                    PKMN_PCB_HEAD,
                    DICE_PCB_HEAD,
                    best_pcb_ratio)    
    
        ### Test Interpolated Models on Original Datasets ###
        if TEST_MERGES:

            if TEST_INTERPOLATION:
                # Load interpolated models (one per head)
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
            
            
            if TEST_AVG:
                # Load parameter average models (one per head)
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
                
            if TEST_PCB:
                # Load pcb models (one per head)
                pkmn_pcb_model_path = os.path.join(OUT_PCB_DIR, PKMN_PCB_HEAD)
                dice_pcb_model_path = os.path.join(OUT_PCB_DIR, DICE_PCB_HEAD)

                pkmn_pcb = load_resnet_from_weights(pkmn_pcb_model_path, PKMN_NUM_CLASSES)
                dice_pcb = load_resnet_from_weights(dice_pcb_model_path, DICE_NUM_CLASSES)
                
                # Test on pkmn dataset
                acc_pkmn_on_pkmn = test_resnet(pkmn_pcb, pkmn_test_loader)
                print(f"[RESULT PCB] PCB Resnet tested on Pokémon: {acc_pkmn_on_pkmn:.4f}")

                # Test on dice dataset
                acc_dice_on_dice = test_resnet(dice_pcb, dice_test_loader)
                print(f"[RESULT PCB] PCB Resnet tested on Dice: {acc_dice_on_dice:.4f}")          

        ###### End merging techniques ######

    ##### Crosscoder Dataset Creation #####
    if CREATE_CROSSCODER_DATASET:
        # The small imagenet_dataset contains a small subset of the original imagenet, which should activate the "old" features, learned during the original training
        small_imagenet_dataset = get_small_imagenet_dataset()
        
        ### Depending on the conditions, we create the different crosscoder activations datasets for the different merged technique
        ### The datasets are created to different locations, so that we're able to reuse the data
        print('[DEBUG] Saving Pokemon/Dice ResNets Activations')
        create_crosscoder_dataset(pokemon_dataset, dice_dataset, small_imagenet_dataset, RESNET_BATCH_SIZE,\
                        [pkmn_net, dice_net],
                        [ACTIVATIONS_POKEMON_PATH, ACTIVATIONS_DICE_PATH], N_CROSSCODER_DATAPOINTS, REGEX_ACTIVATIONS)

        ### Saving Merged Activations ###
        datasets = (pokemon_dataset, dice_dataset, small_imagenet_dataset)

        # Save the activations of the desired merged models
        save_merge_activations(
            COMPUTE_INTERPOLATION_ACTS,
            name='Interpolated',
            model_dir=OUT_INTERPOLATED_DIR,
            model_head=DEFAULT_INTERPOLATED_RESNET_HEAD,
            act_paths=[ACTIVATIONS_INTERPOLATED_PATH],
            datasets=datasets,
            batch_size=RESNET_BATCH_SIZE,
            n_points=N_CROSSCODER_DATAPOINTS,
            regex=REGEX_ACTIVATIONS
        )

        save_merge_activations(
            COMPUTE_AVG_ACTS,
            name='Averaging',
            model_dir=OUT_PARAM_AVERAGE_DIR,
            model_head=DEFAULT_AVG_RESNET_HEAD,
            act_paths=[ACTIVATIONS_PARAM_AVG_PATH],
            datasets=datasets,
            batch_size=RESNET_BATCH_SIZE,
            n_points=N_CROSSCODER_DATAPOINTS,
            regex=REGEX_ACTIVATIONS
        )

        save_merge_activations(
            COMPUTE_PCB_ACTS,
            name='PCB',
            model_dir=OUT_PCB_DIR,
            model_head=PKMN_PCB_HEAD,
            act_paths=[ACTIVATIONS_PCB_PATH],
            datasets=datasets,
            batch_size=RESNET_BATCH_SIZE,
            n_points=N_CROSSCODER_DATAPOINTS,
            regex=REGEX_ACTIVATIONS,
            num_classes=PKMN_NUM_CLASSES
        )

        ### End Saving Merged Activations ###
            
    if TRAIN_CROSSCODER or VAL_CROSSCODER or TEST_CROSSCODER:
        # Load crosscoder activations datasets
        interpolated_suite = prepare_crosscoder_suite(
                                "interpolated",
                                LOAD_INTERPOLATION_DS,
                                ACTIVATIONS_DICE_PATH,
                                ACTIVATIONS_POKEMON_PATH,
                                ACTIVATIONS_INTERPOLATED_PATH,
                                CROSS_INTERPOLATED_WEIGHTS_PATH,
                                BATCH_SIZE_CROSS,
                                TRAINING_SIZE_CROSS,
                                VALIDATION_SIZE_CROSS,
                                TEST_SIZE_CROSS,
                            )
        
        avg_suite          =  prepare_crosscoder_suite(
                                    "param_avg",
                                    LOAD_AVG_DS,
                                    ACTIVATIONS_DICE_PATH,
                                    ACTIVATIONS_POKEMON_PATH,
                                    ACTIVATIONS_PARAM_AVG_PATH,
                                    CROSS_PARAM_AVG_WEIGHTS_PATH,
                                    BATCH_SIZE_CROSS,
                                    TRAINING_SIZE_CROSS,
                                    VALIDATION_SIZE_CROSS,
                                    TEST_SIZE_CROSS,
                                )
        
        pcb_suite          =  prepare_crosscoder_suite(
                                    "pcb",
                                    LOAD_PCB_DS,
                                    ACTIVATIONS_DICE_PATH,
                                    ACTIVATIONS_POKEMON_PATH,
                                    ACTIVATIONS_PCB_PATH,
                                    CROSS_PCB_WEIGHTS_PATH,
                                    BATCH_SIZE_CROSS,
                                    TRAINING_SIZE_CROSS,
                                    VALIDATION_SIZE_CROSS,
                                    TEST_SIZE_CROSS,
                                )
                                
        
        # Crosscoder Training
        if TRAIN_CROSSCODER:

            # We train crosscoder with activations dataset from interpolation merge model
            if LOAD_INTERPOLATION_DS:
                train_crosscoder_and_save_weights(
                    LATENT_DIM,
                    interpolated_suite['n_activations'],
                    LAMBDA_SPARSE,
                    interpolated_suite['total_steps'],
                    interpolated_suite['train_loader'],
                    NUM_EPOCHS_CROSS,
                    LR_CROSS,
                    interpolated_suite['weights_path'],
                    experiment_name='Interpolation_Cross',
                    wandb_config=CROSSCODER_WANDB_CONFIG, 
                    project_name=PROJECT_NAME,
                    description='Crosscoder -- Interpolated Merging'
                )
                
            # We train crosscoder with activations dataset from parameter averagin merge model
            if LOAD_AVG_DS:
                train_crosscoder_and_save_weights(
                    LATENT_DIM,
                    avg_suite['n_activations'],
                    LAMBDA_SPARSE,
                    avg_suite['total_steps'],
                    avg_suite['train_loader'],
                    NUM_EPOCHS_CROSS,
                    LR_CROSS,
                    avg_suite['weights_path'],
                    experiment_name='Avg_Cross',
                    wandb_config=CROSSCODER_WANDB_CONFIG,
                    project_name=PROJECT_NAME, 
                    description='Crosscoder -- Soup Merging'
                )

            # We train crosscoder with activations dataset from PCB merge model
            if LOAD_PCB_DS:
                train_crosscoder_and_save_weights(
                    LATENT_DIM,
                    pcb_suite['n_activations'],
                    LAMBDA_SPARSE,
                    pcb_suite['total_steps'],
                    pcb_suite['train_loader'],
                    NUM_EPOCHS_CROSS,
                    LR_CROSS,
                    pcb_suite['weights_path'],
                    experiment_name='PCB_Cross',
                    wandb_config=CROSSCODER_WANDB_CONFIG, 
                    project_name=PROJECT_NAME,
                    description='Crosscoder -- PCB Merging'
                )

        # Validate  Crosscoder
        if VAL_CROSSCODER:

            # 1) We validate crosscoder model trained on activations dataset from interpolation merge model
            if LOAD_INTERPOLATION_DS:
                validate_crosscoder(
                    LATENT_DIM,
                    interpolated_suite['n_activations'],
                    LAMBDA_SPARSE,
                    interpolated_suite['total_steps'],
                    interpolated_suite['weights_path'],
                    interpolated_suite['val_loader'],
                )

            # 2) We validate crosscoder model trained on activations dataset from parameter averaging merge model   
            if LOAD_AVG_DS:
                validate_crosscoder(
                    LATENT_DIM,
                    avg_suite['n_activations'],
                    LAMBDA_SPARSE,
                    avg_suite['total_steps'],
                    avg_suite['weights_path'],
                    avg_suite['val_loader'],
                )

            # 3) We validate crosscoder model trained on activations dataset from PCB merge model 
            if LOAD_PCB_DS:
                validate_crosscoder(
                    LATENT_DIM,
                    pcb_suite['n_activations'],
                    LAMBDA_SPARSE,
                    pcb_suite['total_steps'],
                    pcb_suite['weights_path'],
                    pcb_suite['val_loader'],
                )
        
        if TEST_CROSSCODER: # Actual analysis
            pass
