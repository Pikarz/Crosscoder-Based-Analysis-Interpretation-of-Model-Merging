import get_datasets
import torch
import torchvision
from torch import nn, optim
from get_datasets import get_small_imagenet_dataset
from get_dataloaders import get_dataloaders
from utils import seed_run
from resnet_model import finetune_resnet, load_resnet_from_weights, interpolate_resnet_models, test_resnet
from create_crosscoder_dataset import create_crosscoder_dataset

#### CONFIG ####
TRAIN                       = False # Variable to either do the actual trainining/interpolation or get the already-finetuned/interpolated versions
CREATE_CROSSCODER_DATASET   = True
PROJECT_NAME                = 'deep_learning'

# same across all the fine-tuned models
BATCH_SIZE      = 32
NUM_EPOCHS      = 10
TRAINING_SIZE   = 0.7
VALIDATION_SIZE = 0.2
TEST_SIZE       = 0.1
LR              = 0.001
MOMENTUM        = 0.9

PKMN_WEIGHTS_PATH   = './pokemon_resnet/model_weights.pth'
PKMN_NUM_CLASSES    = 5 # num of pokemons in the dataset

DICE_NUM_CLASSES    = 6 # num of dice in the dataset
DICE_WEIGHTS_PATH   ='./dice_resnet/model_weights.pth'

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
    "epochs": NUM_EPOCHS,
}

# Interpolation Config
OUT_INTERPOLATED_DIR    = './interpolated'
DEFAULT_RESNET_HEAD     = 'model_weights_v3.pth'
PKMN_HEAD               = 'model_weights_v3_pkmn.pth'
DICE_HEAD               = 'model_weights_v3_dice.pth'

# CrossCoder Dataset Config
BATCH_SIZE_CROSS = 8
# TRAIN_SIZE_CROSS, VALIDATION_SIZE_CROSS, TEST_SIZE_CROSS = 0.4, 0.1, 0.5 # weird split to actually train the crosscoder with sufficient data, we do not use train/val
ACTIVATIONS_PATHS = ['./activations/pokemon', './activations/dice', './activations/interpolated']
# Number of points per dataset -- we do not take everything because otherwise we would have terabytes of data
NUM_POINTS_PER_DATASET = 20

if __name__ == '__main__':
    seed_run() # We seed the run to replicate the results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tags=['resnet', 'classification'],

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
    dice_accuracy = test_resnet(dice_net, dice_test_loader)
    print(f"[RESULT] Dice Resnet tested with an accuracy equal to {dice_accuracy:.4f}")

    ##### End Dice Finetuning #####

    ##### Default original Resnet Tests #####
    ### Default pretrained ResNet with modified head on Pokemon
    resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device)
    resnet.fc.weight = pkmn_net.fc.weight
    resnet.fc.bias   = pkmn_net.fc.bias
    resnet_pkmn_accuracy = test_resnet(resnet, pkmn_test_loader)
    print(f"[RESULT] The original Resnet tested with an accuracy equal to {resnet_pkmn_accuracy:.4f} on the Pokemon Dataset")
    ### Default pretrained ResNet with modified head on Pokemon

    ### Default pretrained ResNet with modified head on Dice
    resnet.fc.weight = dice_net.fc.weight
    resnet.fc.bias   = dice_net.fc.bias
    resnet_pkmn_accuracy = test_resnet(resnet, dice_test_loader)
    print(f"[RESULT] The original Resnet tested with an accuracy equal to {resnet_pkmn_accuracy:.4f} on the Dice Dataset")
    ##### End of default Resnet Tests #####

    ##### Pokemon-Dice Interpolation and Tests #####

    if TRAIN:
        print('[DEBUG] Starting Interpolation')
        interpolate_resnet_models(PKMN_WEIGHTS_PATH, DICE_WEIGHTS_PATH,
                        OUT_INTERPOLATED_DIR, DEFAULT_RESNET_HEAD,
                        PKMN_HEAD, DICE_HEAD,
                        interpolation_weight=0.5
                        )
    
    ### Test Interpolated Models on Original Datasets
    pkmn_interpolated_model_path = OUT_INTERPOLATED_DIR+'/'+PKMN_HEAD
    pkmn_interpolated = load_resnet_from_weights(pkmn_interpolated_model_path, PKMN_NUM_CLASSES)

    dice_interpolated_model_path = OUT_INTERPOLATED_DIR+'/'+DICE_HEAD
    dice_interpolated = load_resnet_from_weights(dice_interpolated_model_path, DICE_NUM_CLASSES)
    # Test on pkmn dataset
    interpolated_pkmn_accuracy_on_pkmn = test_resnet(pkmn_interpolated, pkmn_test_loader)
    print(f"[RESULT] Interpolated Resnet tested with an accuracy equal to {interpolated_pkmn_accuracy_on_pkmn:.4f} on the Pokemon Dataset")
    # test on dice dataset
    interpolated_pkmn_accuracy_on_dice = test_resnet(dice_interpolated, dice_test_loader)
    print(f"[RESULT] Interpolated Resnet tested with an accuracy equal to {interpolated_pkmn_accuracy_on_dice:.4f} on the Dice Dataset")

    ##### End Pokemon-Dice Interpolation and Tests #####

    ##### Crosscoder Dataset Creation #####
    if CREATE_CROSSCODER_DATASET:
        interpolated_model_path = OUT_INTERPOLATED_DIR+'/'+DEFAULT_RESNET_HEAD
        interpolated_net = load_resnet_from_weights(interpolated_model_path)

        # The small imagenet_dataset contains a small subset of the original imagenet, which should activate the "old" features, learned during the original training
        small_imagenet_dataset = get_small_imagenet_dataset()

        create_crosscoder_dataset(pokemon_dataset, dice_dataset, small_imagenet_dataset, BATCH_SIZE_CROSS,\
                                pkmn_net, dice_net, interpolated_net,
                                ACTIVATIONS_PATHS, NUM_POINTS_PER_DATASET)