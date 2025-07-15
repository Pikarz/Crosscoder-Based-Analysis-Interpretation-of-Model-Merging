import torch
import torchvision
from torch import nn

from resnet_model import load_resnet_from_weights
PKMN_WEIGHTS_PATH   = './pokemon_resnet/model_weights.pth'
PKMN_NUM_CLASSES    = 5 # num of pokemons in the dataset

DICE_WEIGHTS_PATH   ='./dice_resnet/model_weights.pth'
DICE_NUM_CLASSES    = 6 # num of dice in the dataset

def count_sign_conflicts(model_a, model_b):
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    common_keys = set(sd_a.keys()).intersection(sd_b.keys())
    diff = 0
    total = 0

    for k in sorted(common_keys):
        wa = sd_a[k]
        wb = sd_b[k]

        if wa.shape != wb.shape:
            # skip layers that don't match (e.g. the different fc heads)
            continue

        sa = wa.sign().view(-1)
        sb = wb.sign().view(-1)
        diff += int(((sa * sb) < 0).sum().item())
        total += sa.numel()

    return diff, total

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pkmn_net = load_resnet_from_weights(PKMN_WEIGHTS_PATH, PKMN_NUM_CLASSES)

    # dice_net = load_resnet_from_weights(DICE_WEIGHTS_PATH, DICE_NUM_CLASSES)

    # print(dice_net)

    torch.clamp(torch.abs(all_checks), min_ratio=0.0001, max_ratio=0.0001)
