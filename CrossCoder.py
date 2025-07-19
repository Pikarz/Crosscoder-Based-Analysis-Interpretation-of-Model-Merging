import einops
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import OneCycleLR

N_MODELS = 3

class CrossCoder(nn.Module):
  def __init__(self, latent_dim, n_activations, lambda_sparse):
    super().__init__()

    self.latent_dim = latent_dim
    self.n_activations = n_activations
    self.lambda_sparse = lambda_sparse
    self.n_models = N_MODELS        # 3 in our case
    

   # self.total_steps    = total_steps # number of batches
    self.current_step   = 0

    self.W_enc = nn.Parameter(
        torch.empty(self.n_models, self.n_activations, self.latent_dim)
    )

    self.W_dec = nn.Parameter(
        torch.nn.init.normal_(
            torch.empty(
                self.latent_dim, self.n_models, self.n_activations
            )
        )
    )

    # normalize
    self.W_dec.data = (
        self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * 0.1
    )
    
    self.W_enc.data = einops.rearrange(
        self.W_dec.data.clone(),
        "latent_dim n_models n_activations -> n_models n_activations latent_dim",
    )

    self.b_enc = nn.Parameter(torch.ones(self.latent_dim))
    self.b_dec = nn.Parameter(
        torch.ones((self.n_models, self.n_activations))
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
    u = self.encode(activations)
    z = F.relu(u)
    recon = self.decode(z) # reconstructions given our latent space

    return recon
  
  def get_loss(self, activations):
    u = self.encode(activations)
    z = F.relu(u)
    recon = self.decode(z)

    l2_loss = (recon - activations).pow(2).mean()

    decoder_norms = self.W_dec.norm(dim=-1)
    total_decoder_norm = decoder_norms.sum(dim=1)   # [latent_dim]
    # sparsity 
    l1_loss = (z * total_decoder_norm[None, :]).mean()

    return l2_loss, l1_loss
  
  def lr_lambda(self, step):
    if step < 0.8 * self.total_steps:
        return 1.0
    else:
        return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)


  def get_l1_coeff(self):
    # Ramp up more gradually over 10% of training
    if self.current_step < 0.10 * self.total_steps:
        return self.lambda_sparse * self.current_step / (0.10 * self.total_steps)
    else:
        return self.lambda_sparse

  def train_cross(self, train_loader, num_epochs, lr):
    print('[DEBUG] Start Training')
    self.total_steps = len(train_loader)*num_epochs
   
    optimizer = torch.optim.Adam(
        self.parameters()
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

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

        for _, data in loop:
            # Get data to cuda if possible 
            activations = data.to(device)

            self.current_step += 1

            # Forward pass + Compute loss
            l2_loss, l1_loss = self.get_loss(activations)
            loss = l2_loss + l1_loss*self.get_l1_coeff()

            # Backward pass and optimization
            loss.backward()
           # clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            # Zero the parameter gradients
            optimizer.zero_grad()

            running_loss += loss.item()
            total_batches += 1

        # Calculate average loss for the training epoch
        avg_train_loss = running_loss/total_batches
        train_losses.append(avg_train_loss)
      #  self.current_step = 0 # At each epoch, we reset the step

        print(f'\nEpoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}. L2 Loss: {l2_loss:.4f}, L1 Loss: {l1_loss:.4f}')
        
    print('[OK] Finished CrossCoder Training')
    return self
  
  def val_cross(self, val_loader):
    print('[DEBUG] Start Validation')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.to(device)

    running_loss = 0

    # Evaluation Phase
    self.eval()

    loop = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", leave=True)
    with torch.no_grad():
        for _, data in loop:
            # Get data to cuda if possible 
            activations = data.to(device)

            # Forward pass + Compute loss
            l2_loss, l1_loss = self.get_loss(activations)
            loss = l2_loss + self.get_l1_coeff()*l1_loss

            running_loss += loss.item()

        print(f'[DEBUG] Validation Loss: {running_loss}')

        print('[OK] Finished CrossCoder Evaluation')
