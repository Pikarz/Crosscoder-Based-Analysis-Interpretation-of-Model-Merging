import einops
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import OneCycleLR
import math
import copy 
import wandb

N_MODELS = 3

class CrossCoder(nn.Module):
  def __init__(self, latent_dim, n_activations, lambda_sparse, total_steps):
    super().__init__()

    self.latent_dim = latent_dim
    self.n_activations = n_activations
    self.lambda_sparse = lambda_sparse
    self.n_models = N_MODELS        # 3 in our case

    self.total_steps    = total_steps # number of batches
    self.current_step   = 0

    self.W_enc = nn.Parameter( 
        torch.empty(self.n_models, self.n_activations, self.latent_dim)
    )

    self.W_dec = nn.Parameter(
        torch.nn.init.kaiming_normal_( # normal initializes values close to zero
            torch.empty(
                self.latent_dim, self.n_models, self.n_activations
            )
        )
    )

    # initialize w_enc to be the transpose of w_dec because we naturally
    # want a decoder that is able to reverse encoder's transformation
    self.W_enc.data = einops.rearrange(
          self.W_dec.data.clone(),
          "latent_dim n_models n_activations -> n_models n_activations latent_dim"
      )
    
    self.b_enc = nn.Parameter(torch.ones(latent_dim))
    self.b_dec = nn.Parameter(torch.ones(self.n_models, n_activations))

    ### Other possible initializations:
    ### The repeat() one seems to perform the best, but since it could be too much biased towards a 'good' result, we have decided to not use it
    
    ## biases
    # self.b_enc = nn.Parameter(torch.zeros(latent_dim))
    # self.b_dec = nn.Parameter(
    #     torch.zeros((self.n_models, self.n_activations))
    # )

    ## encoder/decoder with repeat
    # self.W_enc = nn.Parameter( 
    #     torch.empty(self.n_models, self.n_activations, self.latent_dim)
    # )
    
    ## Initialize a single decoder weight and repeat across all models
    # w_dec_base = torch.empty(latent_dim, n_activations)
    # nn.init.kaiming_normal_(w_dec_base, nonlinearity="linear")
    # self.W_dec = nn.Parameter(
    #     w_dec_base
    #         .unsqueeze(1)                   # shape: [latent_dim, 1, n_activations]
    #         .repeat(1, self.n_models, 1)    # shape: [latent_dim, n_models, n_activations]
    # )

    # # initialize w_enc to be the transpose of w_dec because we naturally
    # # want a decoder that is able to reverse encoder's transformation
    # self.W_enc.data = einops.rearrange(
    #       self.W_dec.data.clone(),
    #       "latent_dim n_models n_activations -> n_models n_activations latent_dim"
    #   )
    
  def encode(self, activations):
    # activations: [crosscoder_batch, n_models, n_activations]
    activations_enc = einops.einsum(
        activations,
        self.W_enc,
        "crosscoder_batch n_models n_activations, n_models n_activations latent_dim -> \
         crosscoder_batch latent_dim",
    )
    return activations_enc + self.b_enc

  def decode(self, activations):
    # activations: [crosscoder_batch, latent_dim]
    activations_dec = einops.einsum(
        activations,
        self.W_dec,
         "crosscoder_batch latent_dim, latent_dim n_models n_activations -> \
         crosscoder_batch n_models n_activations"
    )
    return activations_dec + self.b_dec

  def forward(self, activations):
    # activations:  [crosscoder_batch, n_models, n_activations]
    activations_encoder = self.encode(activations)
    activations_encoder_relu = F.relu(activations_encoder)
    activations_reconstruct = self.decode(activations_encoder_relu) # activations_dec is the reconstructions given our latent space

    return activations_reconstruct
  
  def get_loss(self, activations):
    u = self.encode(activations)
    z = F.relu(u)
    recon = self.decode(z)

    l2_loss = (recon - activations).pow(2).mean()

    decoder_norms = self.W_dec.norm(dim=-1)
    total_decoder_norm = decoder_norms.sum(dim=1)   # [latent_dim]
    # sparsity on pre-act
    l1_loss = (u.abs() * total_decoder_norm[None, :]).mean()

    return l2_loss + self.get_l1_coeff()*l1_loss
  
  def get_lambdas(self):
    def LR_lambda(step, total_steps=self.total_steps):
        if step < 0.8 * total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * total_steps) / (0.2 * total_steps)
    return LR_lambda

  def get_l1_coeff(self):
    # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps, then keeps it constant
    if self.current_step < 0.05 * self.total_steps:
        return self.lambda_sparse * self.current_step / (0.05 * self.total_steps)
    else:
        return self.lambda_sparse

  def train_cross(self, train_loader, val_loader, num_epochs, lr, out_path, experiment_name, wandb_config, 
                  project_name,
                  description,
                  tags=['resnet', 'classification'],):
    
    print('[DEBUG] Start Training')
   
    optimizer = torch.optim.Adam(
        self.parameters(),
    )

    scheduler = OneCycleLR( 
        optimizer,
        max_lr=lr,
        steps_per_epoch=self.total_steps,
        epochs=num_epochs
    )

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

    # Watch model
    run.watch(self, log='all', log_freq=100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(device)

    best_val_loss = math.inf
    best_epoch = -1

    # Training loop
    for epoch in trange(num_epochs, desc="Epochs"): # trange is a shortcut of tqdm(range(...))
        # Training Phase
        self.train()

        running_loss = 0
        total_batches = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
        for _, data in loop:
            # Get data to cuda if possible 
            activations = data.to(device)

            # Forward pass + Compute loss
            loss = self.get_loss(activations)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Zero the parameter gradients
            optimizer.zero_grad()

            self.current_step += 1
            running_loss += loss.item()
            total_batches += 1

        # Calculate average loss for the training epoch
        avg_train_loss = running_loss/total_batches
        self.current_step = 0 # At each epoch, we reset the step

        print(f'\nEpoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}')
        
        ### Validation and save best current model
        self.eval()
        val_loss = self.val_cross(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(self.state_dict())
            print(f"[INFO] New best (epoch {best_epoch} | val_loss {best_val_loss:.4f})")

        run.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss
        })
    
    # save the best model
    if best_model_wts is not None:
        torch.save(best_model_wts, out_path)
        print(f"[INFO] Saved best model")
           
    # before finish, also log to W&B
    artifact = wandb.Artifact(
        name=f"{experiment_name}_best_model",
        type="model",
        description=(
            f"Best model at epoch {best_epoch} "
            f"(val_loss={best_val_loss:.4f})"
        )
    )
    artifact.add_file(out_path)
    run.log_artifact(artifact)
        
    # Finalize W&B
    wandb.finish()
    
    print('[OK] Finished CrossCoder Training')
    print(f"[OK] Best model was from epoch {best_epoch} with val_loss = {best_val_loss:.4f}")

    return best_val_loss
   

  def val_cross(self, val_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(device)

    running_loss = 0
    total_batches = 0

    # Evaluation Phase
    self.eval()

    loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", leave=True)
    with torch.no_grad():
        for _, data in loop:
            # Get data to cuda if possible 
            activations = data.to(device)

            # Forward pass + Compute loss
            loss = self.get_loss(activations)

            running_loss += loss.item()
            total_batches += 1

        avg_train_loss = running_loss/total_batches
        print(f'[INFO] Validation Loss: {avg_train_loss:.4f}')
            
    return avg_train_loss


  

