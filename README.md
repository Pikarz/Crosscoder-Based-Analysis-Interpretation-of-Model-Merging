# Crosscoder-Based Analysis of Model Merging
A project for the Deep Learning course taught by Emanuele Rodolà A.Y. Summer 2024

This project examines how merging two pretrained ResNet-50 models alters their internal representations. We apply Crosscoder model diffing, a sparse autoencoder trained on original and concatenated activations, to identify which features are:

- **Shared:** present in both original models and the merged model  
- **Exclusive:** unique to one original model or to the merged model  
- **New:** emerging only after merging

## Methodology

1. **Model Merging:** Two ResNet-50 checkpoints (e.g. base and fine-tuned) are combined via weight interpolation.  
2. **Activation Extraction:** Intermediate activations from the networks' body layers are concatenated into a single tensor per sample.  
3. **Crosscoder Training:** A sparse autoencoder learns a shared latent space by minimizing reconstruction error for each model's activations, with a sparsity penalty to encourage feature specialization, as described in https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html
4. **Feature Classification:** Latent dimensions are analyzed and labeled as shared, exclusive, or new.

## Outcomes

- A set of interpretable latent features that align or distinguish model representations  
- Quantitative measures of feature overlap and divergence  
- Insights into how merging strategies preserve or alter critical concepts within deep networks  
