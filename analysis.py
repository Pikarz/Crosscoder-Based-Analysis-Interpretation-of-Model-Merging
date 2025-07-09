import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def norm_analysis(crosscoder,
                  model_names=('dice', 'interpolated', 'pokemon'),
                  thr_shared=None,
                  thr_excl=0.8):
    """
    Analyze shared and model-exclusive features for a multi-model crosscoder.

    Args:
        crosscoder: An object with attribute W_dec of shape [latent_dim, num_models, feature_dim].
        model_names: Tuple of names for each model in the order of W_dec's second dimension.
        thr_shared: Optional tuple (low, high) for relative-norm threshold to select shared features.
                    If None, defaults to around uniform share ±0.1.
        thr_excl: Relative-norm threshold above which a feature is considered exclusive to a model.
    Returns:
        A dict with keys:
          - norms: Tensor of shape [latent_dim, num_models] of decoder norms.
          - rel_norms: Tensor of shape [latent_dim, num_models] of relative norms.
          - shared_idx: 1D LongTensor of indices of shared features.
          - exclusive_idx: Dict mapping each model name to a LongTensor of its exclusive feature indices.
    """
    torch.set_grad_enabled(False)
    W = crosscoder.W_dec  # [latent_dim, num_models, feature_dim]
    latent_dim, num_models, feature_dim = W.shape

    # Compute L2 norms of decoder vectors for each latent and model
    norms = W.norm(dim=-1)  # [latent_dim, num_models]
    total = norms.sum(dim=1, keepdim=True)  # [latent_dim, 1]
    rel_norms = norms / (total + 1e-10)

    # Determine thresholds for shared features
    if thr_shared is None:
        mean_share = 1.0 / num_models
        delta = 0.1
        thr_low, thr_high = mean_share - delta, mean_share + delta
    else:
        thr_low, thr_high = thr_shared
    thr_excl_small = (1.0 - thr_excl) / (num_models - 1)

    # Identify shared features (all models have relative norm in [low, high])
    shared_mask = ((rel_norms > thr_low) & (rel_norms < thr_high)).all(dim=1)
    shared_idx = shared_mask.nonzero(as_tuple=True)[0]

    # Identify exclusive features for each model
    exclusive_idx = {}
    for m in range(num_models):
        others = [k for k in range(num_models) if k != m]
        mask = (rel_norms[:, m] > thr_excl)
        for k in others:
            mask &= (rel_norms[:, k] < thr_excl_small)
        exclusive_idx[model_names[m]] = mask.nonzero(as_tuple=True)[0]

    # Plot distribution of relative norms
    plt.figure(figsize=(8, 5))
    bins = 50
    for m, name in enumerate(model_names):
        plt.hist(rel_norms[:, m].cpu().numpy(), bins=bins, alpha=0.5, label=name)
    plt.xlabel('Relative Decoder Norm')
    plt.ylabel('Count')
    plt.title('Distribution of Relative Decoder Norms')
    plt.legend()
    plt.show()

    # Plot cosine similarity of shared decoders between each pair of models
    for i in range(num_models):
        for j in range(i + 1, num_models):
            cos_sim = F.cosine_similarity(
                W[shared_idx, i], W[shared_idx, j], dim=-1
            )  # [n_shared]
            plt.figure(figsize=(6, 4))
            plt.hist(cos_sim.cpu().numpy(), bins=bins)
            plt.title(f'Cosine Similarity: {model_names[i]} vs {model_names[j]} (Shared)')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Count')
            plt.show()

    # Optionally print counts
    print(f"Shared features: {shared_idx.numel()}")
    for name, idx in exclusive_idx.items():
        print(f"Exclusive to {name}: {idx.numel()}")

    return {
        'norms': norms,
        'rel_norms': rel_norms,
        'shared_idx': shared_idx,
        'exclusive_idx': exclusive_idx,
    }


# Example usage:
# result = norm_analysis(crosscoder)
# shared = result['shared_idx']
# exclusive = result['exclusive_idx']
# print("Done analysis.")
