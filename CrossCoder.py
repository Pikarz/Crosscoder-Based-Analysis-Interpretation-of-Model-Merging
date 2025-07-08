import einops
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
import wandb
import copy

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

    # biases
    self.b_enc = nn.Parameter(torch.zeros(latent_dim))
    self.b_dec = nn.Parameter(
        torch.zeros((self.n_activations))
    )

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
    return activations_dec + self.b_dec[None, None, :]

  def forward(self, activations):
    # activations:  [crosscoder_batch, n_models, n_activations]
    activations_encoder = self.encode(activations)
    activations_encoder_relu = F.relu(activations_encoder)
    activations_reconstruct = self.decode(activations_encoder_relu) # activations_dec is the reconstructions given our latent space

    return activations_reconstruct
  
  def get_loss(self, activations):
    activations_encoder = self.encode(activations)

    l1_coeff = self.get_l1_coeff()

    # print('[DEBUG] Activations encoded')
    activations_encoder_relu = F.relu(activations_encoder)
    activations_reconstruct = self.decode(activations_encoder_relu) 
    # print('[DEBUG] Activations reconstructed')

    reconstruction_loss = (activations_reconstruct - activations).pow(2)

    l2_per_batch = einops.reduce(reconstruction_loss, 'crosscoder_batch n_models n_activations -> crosscoder_batch', 'sum')
    l2_loss = l2_per_batch.mean()

    decoder_norms = self.W_dec.norm(dim=-1) # [latent_dim n_models]
    total_decoder_norm = einops.reduce(decoder_norms, 'latent_dim n_models -> latent_dim', 'sum') # the idea is that we want to maintain sparsity to reconstruct each dimension of the latent space
    l1_loss = (activations_encoder_relu * total_decoder_norm[None, :]).sum(-1).mean(0)
 
    loss = l2_loss + l1_coeff * l1_loss

    return loss
  
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

  def train_cross(self, train_loader, num_epochs, lr):
    
    print('[DEBUG] Start Training')
   
    optimizer = torch.optim.Adam(
        self.parameters(),
        lr=lr,
       # betas=(adam_beta_1, adam_beta_2),
    )

    lambdas = self.get_lambdas()
   
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lambdas
    # )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=self.total_steps,
        epochs=num_epochs
    )

   # scheduler = ExponentialLR(optimizer, gamma=0.9)

    ### TODO WANDB  STUFF ###

    train_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(device)

    # Training Phase
    self.train()

    # Training loop
    for epoch in trange(num_epochs, desc="Epochs"): # trange is a shortcut of tqdm(range(...))
        running_loss = 0
        total_batches = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=True)
        old_enc = copy.deepcopy(self.W_enc)
        old_dec = copy.deepcopy(self.W_dec)

        for batch_idx, data in loop:
            # Get data to cuda if possible 
            activations = data.to(device)

            self.current_step += 1

            # Forward pass + Compute loss
            loss = self.get_loss(activations)

            # Backward pass and optimization
            loss.backward()

            optimizer.step()
            scheduler.step()
            # Zero the parameter gradients
            optimizer.zero_grad()

            running_loss += loss.item()
            total_batches += 1

        # Calculate average loss for the training epoch
        avg_train_loss = running_loss/total_batches
        train_losses.append(avg_train_loss)
        self.current_step = 0 # At each epoch, we reset the step
        delta = (self.W_enc - old_enc).abs().max().item()
        delta_dec = (self.W_dec - old_dec).abs().max().item()

        print(f'\nEpoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}\n \
                Encoders similarity: {delta}\n\
                Decoders similarity: {delta_dec}')


        # run.log({
        #     "epoch": epoch + 1,
        #     "train/loss": avg_train_loss,
        # })

        
    print('[OK] Finished CrossCoder Training')

    # finally:
    #     wandb.finish()

  def val_cross(self, val_loader):
    print('[DEBUG] Start Validation')

    ### TODO WANDB  STUFF ###

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(device)

    running_loss = 0

    # Evaluation Phase
    self.eval()

    loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Training", leave=True)
    with torch.no_grad():
        for batch_idx, data in loop:
            # Get data to cuda if possible 
            activations = data.to(device)

            # Forward pass + Compute loss
            loss = self.get_loss(activations)

            running_loss += loss.item()

        print(f'[DEBUG] Validation Loss: {running_loss}')

        # run.log({
        #     "epoch": epoch + 1,
        #     "train/loss": avg_train_loss,
        # })

            
        print('[OK] Finished CrossCoder Evaluation')





  

