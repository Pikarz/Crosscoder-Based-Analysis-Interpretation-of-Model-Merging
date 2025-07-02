import os
import torch
import torchvision
import wandb
from tqdm import tqdm, trange

def load_resnet_from_weights(weights_path, num_classes):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = torchvision.models.resnet50() # By default, no pre-trained weights are used
  num_features = model.fc.in_features
  model.fc = torch.nn.Linear(num_features, num_classes)
  model = model.to(device)

  model.load_state_dict(torch.load(weights_path, weights_only=True))

  return model

def finetune_resnet(net, train_loader, val_loader,
                    loss_func, optimizer, wandb_config,
                    num_epochs,
                    class_names, # ie: ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle'],
                    weights_path, # ie: './pokemon_resnet/model_weights.pth',
                    experiment_name, # ie:'pokemon_resnet_v2',
                    description, # ie: "Pokemon ResNet finetuning run"
                    project_name='deep_learning',
                    tags=['resnet', 'classification'],
                    ):
      # --- W&B init ---
    wandb.login()
    run = wandb.init(
        project=project_name,
        name=experiment_name,
        config=wandb_config,
        tags=tags,
        notes=description,
        reinit=True,
    )

    # Watch model: log gradients & parameter histograms every 100 steps
    run.watch(net, criterion=loss_func, log='all', log_freq=100)

    # Initialize metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    try:
        # Training loop
        for epoch in trange(num_epochs, desc="Epochs"): # trange is a shortcut of tqdm(range(...))
            running_loss = 0.0
            correct = 0
            total = 0
            total_batches = 0

            # Training Phase
            net.train()

            loop = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=True)
            for batch_idx, (data, targets) in loop:
                # Get data to cuda if possible 
                inputs, labels = data.to(device=device), targets.to(device=device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = net(inputs)

                # Compute loss
                loss = loss_func(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_batches += 1

                # torch.max output --> out (tuple, optional) – the result tuple of two output tensors (max, max_indices)
                _, preds = torch.max(outputs.data, dim=1)
                total += labels.size(0) # the number of elements in the current batch
                correct += (preds == labels).sum().item()

            # Calculate average loss for the training epoch
            avg_train_loss = running_loss/total_batches
            train_losses.append(avg_train_loss)

            # Calculate training accuracy
            train_accuracy = correct / total
            train_accuracies.append(train_accuracy)

            current_lr = optimizer.param_groups[0]['lr']

            print(f'\n\tEpoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

            # Validation Phase
            net.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", leave=True)
            with torch.no_grad():
                for batch_idx, (data, targets) in loop:
                    val_inputs, val_labels = data.to(device=device), targets.to(device=device)

                    # Forward pass
                    val_outputs = net(val_inputs)

                    # Compute loss
                    val_loss = loss_func(val_outputs, val_labels)
                    val_running_loss += val_loss.item()

                    # Calculate accuracy
                    # torch.max output --> out (tuple, optional) – the result tuple of two output tensors (max, max_indices)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    total += val_labels.size(0)
                    correct += (val_predicted == val_labels).sum().item()

                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())

            # Calculate average loss for the validation epoch
            avg_val_loss = val_running_loss/len(val_loader)
            val_losses.append(avg_val_loss)

            # Calculate validation accuracy
            val_accuracy = correct/total
            val_accuracies.append(val_accuracy)

            run.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/acc": train_accuracy,
                "val/loss": avg_val_loss,
                "val/acc": val_accuracy,
                "lr": current_lr,
            })

            # Log metrics + confusion matrix
            cm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=class_names
            )
            print(f'\n\tEpoch {epoch + 1}, Average Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

            # save the best model
            if val_accuracy > best_val_acc:
                dirpath = os.path.dirname(weights_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            # save the model’s state dict
            torch.save(net.state_dict(), weights_path)

            artifact = wandb.Artifact(
                        name=f"{experiment_name}-best-model",
                        type="model",
                        description=f"Best val_acc={best_val_acc:.4f}"
                    )
            artifact.add_file(weights_path)
            run.log_artifact(artifact)


        print('[OK] Finished Training')
        print(f"[OK] Weights saved in {weights_path}")

    finally:
        wandb.finish()

    return train_losses, train_accuracies, val_losses, val_accuracies


def test_resnet(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    # test
    correct = 0
    total = 0

    loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

    with torch.no_grad():
        for batch_idx, (data, targets) in loop:
            test_inputs, test_labels = data.to(device=device), targets.to(device=device)

            # preds
            test_outputs = model(test_inputs)

            # Calculate accuracy
            # torch.max outputs --> two tensors (max, max_indices)
            _, test_predicted = torch.max(test_outputs.data, 1)
            total += test_labels.size(0)
            correct += (test_predicted == test_labels).sum().item()

    # Calculate validation accuracy
    test_accuracy = correct/total

    # print(f'Test Accuracy: {test_accuracy:.4f}')

    return test_accuracy

def interpolate_resnet_models(path_A, path_B,
                       out_dir_path, out_name,
                       out_head1_name, out_head2_name,
                       interpolation_weight=0.5,
                       skip_layers : tuple = ("fc.weight", "fc.bias", "num_batches_tracked")):
  """
    path_A, path_B are the paths to two model weights. This is assuming the weights have the same underlying architecture (i.e., ResNet)
    out_dir_path is the output directory
    out_name, out_head1_name and out_head2_name are the outputs of the models with the original resnet's head, the first model head and the second model head respectively.

    The interpolation weight is used in lerp(); we want both model to have the same weight, thus 0.5.
    The interpolation is not done on the last layer for two main reasons:
      1: The last layer may have a different number of classes. In our example, the two ResNets have 5 and 6 classes.
      2: It may not make sense conceptually. Let's assume that, after performing the interpolation and we get a new model,
         we feed one datapoint with label zero. The datapoint comes from the union of the two datasets that have been used to
         finetune model_A and model_B. Which class name has actually the datapoint? The datasets can be different.
         In our practical example: is it a pokemon, or a dice?
  """
  model_A = torch.load(path_A)
  model_B = torch.load(path_B)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  new_model = torchvision.models.resnet50().to(device)
  state_dict = new_model.state_dict()

  for param, weight in state_dict.items():
    if any([ param.endswith(skip) for skip in skip_layers ]): # we skip the last fc layer and all the num_batches_tracked because we don't need them
      continue

    w1 = model_A[param]
    w2 = model_B[param]

    w_new = torch.lerp(w1.cuda(), w2.cuda(), interpolation_weight)
    weight.data = w_new

  new_model.load_state_dict(state_dict)

  os.makedirs(out_dir_path, exist_ok=True)
  out_path = os.path.join(out_dir_path, out_name)
  # save the model's state dict -- general
  torch.save(new_model.state_dict(), out_path)

  # save two models, each with a different head to test them later
  new_model.fc.weight = torch.nn.Parameter(model_A['fc.weight'])
  new_model.fc.bias   = torch.nn.Parameter(model_A['fc.bias'])
  out_path_head1 = os.path.join(out_dir_path, out_head1_name)
  torch.save(new_model.state_dict(), out_path_head1)

  new_model.fc.weight = torch.nn.Parameter(model_B['fc.weight'])
  new_model.fc.bias   = torch.nn.Parameter(model_B['fc.bias'])
  out_path_head2 = os.path.join(out_dir_path, out_head2_name)
  torch.save(new_model.state_dict(), out_path_head2)


if __name__ == '__main__':
    import get_datasets
    from torch import nn, optim
    from get_dataloaders import get_dataloaders
    from utils import seed_run

    seed_run() # We seed the run to replicate the results

    do_the_train = False # Variable to either do the actual trainining/interpolation or get the already-finetuned/interpolated versions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    project_name='deep_learning',
    tags=['resnet', 'classification'],

    ##### Pokemon Finetuning ####
    ### Prepare data
    print('[DEBUG] Loading Pokemon Dataset')
    pokemon_dataset = get_datasets.get_pokemon_dataset()
    batch_size      = 32
    training_size   = 0.7
    validation_size = 0.2
    test_size       = 0.1

    pkmn_train_loader, pkmn_val_loader, pkmn_test_loader = get_dataloaders(pokemon_dataset, batch_size, training_size, validation_size, test_size)

    ### Prepare Resnet
    pkmn_weights_path = './pokemon_resnet/model_weights.pth'
    pkmn_num_classes = 5 # num of pokemons in the dataset

    if do_the_train:
        pkmn_net = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_features = pkmn_net.fc.in_features
        pkmn_net.fc = nn.Linear(num_features, pkmn_num_classes)
        pkmn_net = pkmn_net.to(device)

        ### Prepare training
        loss_func = nn.CrossEntropyLoss()
        lr = 0.001
        momentum = 0.9
        # we optimize EVERY layer because we want to show the difference of the two finetuned versions of resnet
        optimizer = optim.SGD(pkmn_net.parameters(), lr=lr, momentum=momentum)

        ### Wandb data
        pkmn_class_names = ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle']
        pkmn_experiment_name = 'pokemon_resnet_v3'
        pkmn_description = "Pokemon ResNet finetuning run after refactor"
        pkmn_wandb_config={
            "learning_rate": lr,
            "momentum": momentum,
            "architecture": "ResNet",
            "dataset": "manuel-yao/pokemon-keras-community",
            "epochs": num_epochs,
        }

        ### Finetuning
        print("[DEBUG] Starting Pokemon finetuning")
        finetune_resnet(pkmn_net, pkmn_train_loader, pkmn_val_loader,
                            loss_func, optimizer, pkmn_wandb_config,
                            num_epochs,
                            pkmn_class_names, # ie: ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle'],
                            pkmn_weights_path, # ie: './pokemon_resnet/model_weights.pth',
                            pkmn_experiment_name, # ie:'pokemon_resnet_v2',
                            pkmn_description, # ie: "Pokemon ResNet finetuning run"
                            project_name='deep_learning',
                            tags=['resnet', 'classification'],
        )
    else:
        pkmn_net = load_resnet_from_weights(pkmn_weights_path, pkmn_num_classes)

    ### Test 
    pkmn_accuracy = test_resnet(pkmn_net, pkmn_test_loader)
    print(f"[RESULT] Pokemon Resnet tested with an accuracy equal to {pkmn_accuracy:.4f}")

    ##### End Pokemon Finetuning ####

    ##### Dice Finetuning ####

    ### Prepare Resnet
    dice_num_classes = 6 # num of dice in the dataset
    dice_weights_path='./dice_resnet/model_weights.pth'

    ### Prepare data
    print('[DEBUG] Loading Dice Dataset')
    dice_dataset = get_datasets.get_dice_dataset()
    batch_size      = 64
    training_size   = 0.7
    validation_size = 0.2
    test_size       = 0.1

    dice_train_loader, dice_val_loader, dice_test_loader = get_dataloaders(dice_dataset, batch_size, training_size, validation_size, test_size)
    if do_the_train:
        dice_net = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        num_features = dice_net.fc.in_features
        dice_net.fc = nn.Linear(num_features, dice_num_classes)
        dice_net = dice_net.to(device)

        ### Prepare training
        loss_func = nn.CrossEntropyLoss()
        lr = 0.001
        momentum = 0.9
        # we optimize EVERY layer because we want to show the difference of the two finetuned versions of resnet
        optimizer = optim.SGD(dice_net.parameters(), lr=lr, momentum=momentum)

        ### Wandb data
        dice_class_names=['d10','d12','d20','d4','d6','d8']
        dice_experiment_name = 'dice_resnet_v3'
        dice_description = "Dice ResNet finetuning run after refactor"
        dice_wandb_config={
            "learning_rate": lr,
            "momentum": momentum,
            "architecture": "ResNet",
            "dataset": "ucffool/dice-d4-d6-d8-d10-d12-d20-images", # from Kaggle
            "epochs": num_epochs,
        }


        ### Finetuning
        print("[DEBUG] Starting Dice finetuning")
        finetune_resnet(dice_net, dice_train_loader, dice_val_loader,
                            loss_func, optimizer, dice_wandb_config,
                            num_epochs,
                            dice_class_names,
                            dice_weights_path,
                            dice_experiment_name,
                            dice_description,
                            project_name='deep_learning',
                            tags=['resnet', 'classification'],
        )
    else:
        dice_net = load_resnet_from_weights(dice_weights_path, dice_num_classes)

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

    ##### End original Resnet Tests #####

    ##### Pokemon-Dice Interpolation and Tests #####
    out_dir_path = './interpolated'
    out_name= 'model_weights_v3.pth'
    out_pkmn_head_name = 'model_weights_v3_pkmn.pth'
    out_dice_head_name = 'model_weights_v3_dice.pth'

    if do_the_train:
        print('[DEBUG] Starting Interpolation')
        interpolate_resnet_models(pkmn_weights_path, dice_weights_path,
                        out_dir_path, out_name,
                        out_pkmn_head_name, out_dice_head_name,
                        interpolation_weight=0.5
                        )
    
    ### Test Interpolated Models on Original Datasets
    pkmn_interpolated_model_path = out_dir_path+'/'+out_pkmn_head_name
    pkmn_interpolated = load_resnet_from_weights(pkmn_interpolated_model_path, pkmn_num_classes)

    dice_interpolated_model_path = out_dir_path+'/'+out_dice_head_name
    dice_interpolated = load_resnet_from_weights(dice_interpolated_model_path, dice_num_classes)
    # Test on pkmn dataset
    interpolated_pkmn_accuracy_on_pkmn = test_resnet(pkmn_interpolated, pkmn_test_loader)
    print(f"[RESULT] Interpolated Resnet tested with an accuracy equal to {interpolated_pkmn_accuracy_on_pkmn:.4f} on the Pokemon Dataset")
    # test on dice dataset
    interpolated_pkmn_accuracy_on_dice = test_resnet(dice_interpolated, dice_test_loader)
    print(f"[RESULT] Interpolated Resnet tested with an accuracy equal to {interpolated_pkmn_accuracy_on_dice:.4f} on the Dice Dataset")

    ##### End Pokemon-Dice Interpolation and Tests #####

