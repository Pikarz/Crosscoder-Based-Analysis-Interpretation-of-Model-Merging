import get_datasets
import torch
import torchvision
from torch import nn, optim
from get_datasets import get_small_imagenet_dataset
from get_dataloaders import get_dataloaders
from utils import seed_run
from resnet_model import finetune_resnet, load_resnet_from_weights, interpolate_resnet_models, test_resnet
from crosscoder_dataset_utils import create_crosscoder_dataset
from CrossCoderDataset import CrossCoderDataset
from CrossCoder import CrossCoder

#### CONFIG ####
TRAIN                       = False # Do the actual trainining/interpolation or get the already-finetuned/interpolated versions
TEST                        = False # Test models or skip tests
CREATE_CROSSCODER_DATASET   = False # Create the dataset or use the already-created one
TRAIN_CROSSCODER            = True  # Train or use the already-trained crosscoder
VAL_CROSSCODER              = True  # Crosscoder validation
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
OUT_INTERPOLATED_DIR    = './interpolated'
DEFAULT_RESNET_HEAD     = 'model_weights_v3.pth'
PKMN_HEAD               = 'model_weights_v3_pkmn.pth'
DICE_HEAD               = 'model_weights_v3_dice.pth'

#### CrossCoder Dataset Config
RESNET_BATCH_SIZE = 1 # number of images given to our resnets to compute a single set of activations
# Number of datapoints generated per model (dimension [n_models, n_crosscoder_datapoints, n_activations]) -- we do not take everything because otherwise we would have terabytes of data
N_CROSSCODER_DATAPOINTS = 500
ACTIVATIONS_PATH = './activations_layer4'
MODEL_ACTIVATIONS_PATHS = [f'{ACTIVATIONS_PATH}/pokemon', f'{ACTIVATIONS_PATH}/dice', f'{ACTIVATIONS_PATH}/interpolated']
REGEX_ACTIVATIONS = '^layer4$'  # Regex to get only the big sequential layers inside the resnets. We were not able to get the whole activations due to computational/memory power limit

### CrossCoder Model Config
BATCH_SIZE_CROSS = 4 # crosscoder batch -- number of datapoints fetched by the crosscoder dataloader
NUM_EPOCHS_CROSS = 10
LATENT_DIM = 4
TRAINING_SIZE_CROSS   = 0.7
VALIDATION_SIZE_CROSS = 0.1 # smaller validation because we just have to tune the latent_dim hyperparam
TEST_SIZE_CROSS       = 0.2
LR_CROSS = 0.01
LAMBDA_SPARSE = 2 # TODO boh

if __name__ == '__main__':
    seed_run() # We seed the run to replicate the results

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tags    = ['resnet', 'classification']

    if (TRAIN or TEST or CREATE_CROSSCODER_DATASET): # If we just care about the crosscoder model part, then we skip everything
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

        if TEST:
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

        create_crosscoder_dataset(pokemon_dataset, dice_dataset, small_imagenet_dataset, RESNET_BATCH_SIZE,\
                                pkmn_net, dice_net, interpolated_net,
                                MODEL_ACTIVATIONS_PATHS, N_CROSSCODER_DATAPOINTS, REGEX_ACTIVATIONS)
        
    crosscoder_dataset = CrossCoderDataset(ACTIVATIONS_PATH)
    n_activations = crosscoder_dataset.get_n_activations() # [n_models, n_crosscoder_datapoints, n_activations]
    cross_train_loader, cross_val_loader, cross_test_loader = get_dataloaders(crosscoder_dataset, BATCH_SIZE_CROSS, TRAINING_SIZE_CROSS, VALIDATION_SIZE_CROSS, TEST_SIZE_CROSS)
    
    # TODO validation on latent dim
    if TRAIN_CROSSCODER:
        # print(f"n_activations: {n_activations}")
        total_steps = len(cross_train_loader) # total steps per epoch
        crosscoder = CrossCoder(LATENT_DIM, n_activations, LAMBDA_SPARSE, total_steps)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        crosscoder.to(device)

        # param_size = 0
        # for param in crosscoder.parameters():
        #     param_size += param.nelement() * param.element_size()
        # buffer_size = 0
        # for buffer in crosscoder.buffers():
        #     buffer_size += buffer.nelement() * buffer.element_size()

        # size_all_mb = (param_size + buffer_size) / 1024**2
        # print('model size: {:.3f}MB'.format(size_all_mb))

        crosscoder.train_cross(cross_train_loader, NUM_EPOCHS_CROSS, LR_CROSS)
    
        #TODO save logic here and loading in val

    if VAL_CROSSCODER:
        crosscoder.val_cross(cross_val_loader)
        
    