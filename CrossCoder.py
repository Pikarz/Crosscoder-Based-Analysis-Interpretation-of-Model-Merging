import einops
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
import wandb

N_MODELS = 3

class CrossCoder(nn.Module):
  def __init__(self, latent_dim, n_activations, lambda_sparse):
    super().__init__()

    self.latent_dim = latent_dim
    self.n_activations = n_activations
    self.lambda_sparse = lambda_sparse
    self.n_models = N_MODELS        # 3 in our case

    self.W_enc = nn.Parameter( 
        torch.empty(self.n_models, self.n_activations, self.latent_dim)
    )

    self.W_dec = nn.Parameter(
        torch.nn.init.normal_( # normal initializes values close to zero
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
        torch.zeros((self.n_models, self.n_activations))
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
    return activations_dec + self.b_dec

  def forward(self, activations):
    # activations:  [crosscoder_batch, n_models, n_activations]
    activations_encoder = self.encode(activations)
    activations_encoder_relu = nn.Relu(activations_encoder)
    activations_reconstruct = self.decoder(activations_encoder_relu) # activations_dec is the reconstructions given our latent space

    return activations_reconstruct

  def get_loss(self, activations):
    activations_encoder = self.encode(activations)
    print('[DEBUG] Activations encoded')
    activations_reconstruct = self.decode(activations_encoder) 
    print('[DEBUG] Activations reconstructed')

    reconstruction_loss = (activations_reconstruct - activations).pow(2)

    l2_per_batch = einops.reduce(reconstruction_loss, 'crosscoder_batch n_models n_activations -> crosscoder_batch', 'sum')
    l2_loss = l2_per_batch.mean()
    decoder_norms = self.W_dec.norm(dim=-1) # [latent_dim n_models]
    total_decoder_norm = decoder_norms.sum(dim=-1) # [latent_dim] -- the idea is that we want to maintain sparsity to reconstruct each dimension of the latent space
    l1_loss = (activations_encoder * total_decoder_norm[None, :]).sum(-1).mean(0)

    loss = l2_loss + self.lambda_sparse * l1_loss

    return loss
  

def get_lambdas(total_steps):
    def LR_lambda(step, total_steps=total_steps):
        if step < 0.8 * total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * total_steps) / (0.2 * total_steps)
    return LR_lambda

def train_crosscoder(crosscoder, train_loader,
                     num_epochs,
                     lr, adam_beta_1, adam_beta_2):
    
    print('[DEBUG] Start Training')
   
    optimizer = torch.optim.Adam(
        crosscoder.parameters(),
        lr=lr,
        betas=(adam_beta_1, adam_beta_2),
    )

    total_steps = len(train_loader) * num_epochs
    lambdas = get_lambdas(total_steps)
   
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambdas
    )

    ### TODO WANDB  STUFF ###

    train_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crosscoder.to(device)

    # Training loop
    for epoch in trange(num_epochs, desc="Epochs"): # trange is a shortcut of tqdm(range(...))
        running_loss = 1
        total_batches = 1

        # Training Phase
        crosscoder.train()

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=True)
        for batch_idx, data in loop:
            # Get data to cuda if possible 
            activations = data.to(device=device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass + Compute loss
            loss = crosscoder.get_loss(activations)

            # Backward pass and optimization
            loss.backward()
            clip_grad_norm_(crosscoder.parameters(), max_norm=1.0) # gradient norm is at most 1

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            total_batches += 1

        # Calculate average loss for the training epoch
        avg_train_loss = running_loss/total_batches
        train_losses.append(avg_train_loss)


        print(f'\n\tEpoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}')


        # run.log({
        #     "epoch": epoch + 1,
        #     "train/loss": avg_train_loss,
        # })

        
    print('[OK] Finished CrossCoder Training')

    # finally:
    #     wandb.finish()




  

