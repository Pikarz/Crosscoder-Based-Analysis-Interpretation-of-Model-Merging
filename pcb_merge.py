import torch
import torchvision
import os

from resnet_model import load_resnet_from_weights, test_resnet

def create_pcb_merge(path_A, path_B, pcb_out_dir, pcb_model_A_head, pcb_model_B_head, pcb_ratio=1.0, min_ratio=0.0001, max_ratio=0.0001):
    # load the two fine‐tuned state_dicts
    model_A = torch.load(path_A, map_location="cpu")
    model_B = torch.load(path_B, map_location="cpu")

    # only merge same‐shape parameters, skip heads and BN buffers
    shared_keys = [
        k for k in model_A
        if (
            k in model_B
            and model_A[k].shape == model_B[k].shape
            and not k.startswith("fc.")
            and "bn." not in k            # skip BN affine
            and not any(substr in k for substr in ("running_mean","running_var","num_batches_tracked"))
        )
    ]

    # gather and flatten
    tensors, shapes = [], []
    for k in shared_keys:
        w1 = model_A[k].flatten()
        w2 = model_B[k].flatten()
        tensors.append(torch.stack([w1, w2], dim=0))
        shapes.append(model_A[k].shape)
    flat_checks = torch.cat(tensors, dim=1)

    # do the PCB merge
    merged_flat, _, _ = PCB_merge(flat_checks, pcb_ratio, min_ratio=min_ratio, max_ratio=max_ratio)
    merged_flat = merged_flat.detach().cpu()

    # load fresh ResNet and prepare new state dict
    model = torchvision.models.resnet50(weights=None)
    new_sd = model.state_dict()
    cursor = 0
    for k, shape in zip(shared_keys, shapes):
        numel = int(torch.tensor(shape).prod().item())
        new_sd[k] = merged_flat[cursor:cursor+numel].view(shape)
        cursor += numel

    # restore BN running stats from model_A
    for k in model_A:
        if "bn.weight" in k or "bn.bias" in k or "running_" in k:
            new_sd[k] = model_A[k].clone()

    # load merged backbone & freeze for eval
    model.load_state_dict(new_sd)
    model.eval()

    # --- save A‐head ---
    num_classes_A, _ = model_A['fc.weight'].shape
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes_A)
    model.fc.weight = torch.nn.Parameter(model_A['fc.weight'].clone())
    model.fc.bias   = torch.nn.Parameter(model_A['fc.bias'].clone())
    os.makedirs(pcb_out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(pcb_out_dir, pcb_model_A_head))

    # --- now restore BN running stats from model_B for the B‐head save ---
    for k in model_B:
        if "bn.weight" in k or "bn.bias" in k or "running_" in k:
            new_sd[k] = model_B[k].clone()

    # reload merged backbone with B stats & freeze for eval
    model = torchvision.models.resnet50(weights=None)
    model.load_state_dict(new_sd)
    model.eval()

    # --- save B‐head ---
    num_classes_B, _ = model_B['fc.weight'].shape
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes_B)
    model.fc.weight = torch.nn.Parameter(model_B['fc.weight'].clone())
    model.fc.bias   = torch.nn.Parameter(model_B['fc.bias'].clone())
    torch.save(model.state_dict(), os.path.join(pcb_out_dir, pcb_model_B_head))

def pcb_grid_search(pkmn_weights_path, dice_weights_path, pcb_weights_path, pkmn_head, dice_head, pkmn_num_classes, dice_num_classes, pkmn_val_loader, dice_val_loader):
        ### PCB Merging Hyperparams Grid Search ###
    best_acc = 0
    best_pcb_ratio  = None
    best_min_ratio  = None
    best_max_ratio  = None
    
    for pcb_ratio in torch.linspace(0.5, 0.999, 50):
        #for min_ratio in (0.0001, 0.01, 0.1):
           # for max_ratio in (0.01, 0.1, 0.5, 0.7, 0.9):
                    create_pcb_merge(pkmn_weights_path, dice_weights_path, pcb_weights_path, pkmn_head, dice_head, pcb_ratio=pcb_ratio) #, min_ratio=min_ratio, max_ratio=max_ratio)

                    pkmn_path = pcb_weights_path + '/' + pkmn_head
                    pcb_resnet = load_resnet_from_weights(pkmn_path, pkmn_num_classes)
                    pcb_resnet.eval()

                    pcb_pkmn_accuracy = test_resnet(pcb_resnet, pkmn_val_loader)
                    # print(f"[RESULT] The PCB Resnet tested with an accuracy equal to {pcb_pkmn_accuracy:.4f} on the Pokemon Dataset")

                    dice_path = pcb_weights_path + '/' + dice_head
                    pcb_resnet = load_resnet_from_weights(dice_path, dice_num_classes)
                    pcb_resnet.eval()

                    pcb_dice_accuracy = test_resnet(pcb_resnet, dice_val_loader)
                    #print(f"[RESULT] The PCB Resnet tested with an accuracy equal to {pcb_dice_accuracy:.4f} on the Dice Dataset")   

                    accuracy = (pcb_pkmn_accuracy + pcb_dice_accuracy)/2
                    print(f"[RESULT] The PCB Resnet tested with an average accuracy equal to {accuracy:.4f} on the Pokemon and Dice Datasets with pcb_ratio={pcb_ratio}")#, min_ratio={min_ratio}, max_ratio={max_ratio}")
                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_pcb_ratio  = pcb_ratio
                       # best_min_ratio  = min_ratio
                       # best_max_ratio  = max_ratio
                        print(f"\t [RESULT] New Best Result!")
                    
    print(f"[RESULT] The best PCB Resnet tested with an average accuracy equal to {best_acc:.4f} on the Pokemon and Dice Datasets with pcb_ratio={best_pcb_ratio}") #, min_ratio={best_min_ratio}, max_ratio={best_max_ratio}")

    import torch


### From Github: https://github.com/duguodong7/pcb-merging
def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y

def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min=sorted_x[int(d * min_ratio)]
        max=sorted_x[int(d * (1-max_ratio)-1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min=sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max=sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
    clamped_x = torch.clamp(x, min, max)
    return clamped_x

def act(x):
    y = torch.tanh(x)  # torch.relu(x)
    return y

def PCB_merge(flat_task_checks, pcb_ratio=0.1, min_ratio=0.0001, max_ratio=0.0001):
    all_checks = flat_task_checks.clone()
    n, d = all_checks.shape   

    # all_checks_abs = clamp(torch.abs(all_checks), min_ratio=min_ratio, max_ratio=max_ratio) # original code, flattened too much the signals
    all_checks_abs = torch.abs(all_checks) # -- one working solution

    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    self_pcb = normalize(all_checks_abs, 1)**2
    self_pcb_act = torch.exp(n*self_pcb)
    cross_pcb = all_checks * torch.sum(all_checks, dim=0)
    cross_pcb_act = act(cross_pcb)
    task_pcb = self_pcb_act * cross_pcb_act

    scale = normalize(clamp(task_pcb, 1-pcb_ratio, 0), dim=1)
    tvs = clamped_all_checks
    merged_tv = torch.sum(tvs * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
    return merged_tv, clamped_all_checks, scale
