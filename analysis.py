import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Union, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

def plot_feature_density_log10(densities, shared_idx, exclusive_idx, names):
    """
    densities: 1D numpy array of length D, per-feature density
    shared_idx: 1D LongTensor or array of indices
    exclusive_idx: dict model→indices
    names: dict model→string
    """
    # avoid log(0)
    eps = 1e-12
    log_d = np.log10(densities + eps)

    plt.figure(figsize=(8,6))
    # shared
    plt.hist(
        log_d[shared_idx],
        bins=50,
        density=True,
        histtype='step',
        linewidth=2,
        label='Shared features'
    )
    # each model-exclusive
    for m, idx in exclusive_idx.items():
        plt.hist(
            log_d[idx],
            bins=50,
            density=True,
            histtype='step',
            linewidth=2,
            label=f"{names[m]}-only features"
        )

    plt.xlabel(r'$\log_{10}$ Feature Density', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Feature densities, shared and model-exclusive features', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


def analyze_crosscoder(
    crosscoder,
    data: Union[torch.Tensor, DataLoader] = None,
    shared_tol: float = 0.05,
    exclusive_threshold: float = 0.95,
    density_threshold: float = 1e-3,
) -> Dict[str, object]:
    """
    As before, but `data` is either:
      • a Tensor [N, M, A], or
      • a DataLoader yielding batches [B, M, A].
    Computes densities by treating each (sample,model) pair equally.
    """
    # ─── Unpack weights ─────────────────────────────────────────────────────────
    W_dec = crosscoder.W_dec.detach().cpu()   # [D, M, A]
    W_enc = crosscoder.W_enc.detach().cpu()   # [M, A, D]
    b_enc = crosscoder.b_enc.detach().cpu()   # [D]
    D, M, _ = W_dec.shape
    names = {0: "dice", 1: "interpolated", 2: "pokemon"}

    # ─── Norms & relative norms ─────────────────────────────────────────────────
    norms     = W_dec.norm(dim=2)                                # [D, M]
    rel_norms = norms / (norms.sum(1, keepdim=True) + 1e-12)     # [D, M]

    # ─── Shared vs exclusive indices ────────────────────────────────────────────
    shared_mask  = (rel_norms - 1.0/M).abs().max(dim=1).values < shared_tol
    shared_idx   = torch.where(shared_mask)[0]
    exclusive_idx= {
        m: torch.where(rel_norms[:, m] > exclusive_threshold)[0]
        for m in range(M)
    }
    new_idx = exclusive_idx[1]  # “interpolated” as new

    # ─── Pairwise rel‐norm & cos‐sim ─────────────────────────────────────────────
    pairs = [(0,1),(0,2),(1,2)]
    pairwise_rel, pairwise_cos = {}, {}
    for i, j in pairs:
        pairwise_rel[(i,j)] = (rel_norms[:,i] / (rel_norms[:,i] + rel_norms[:,j] + 1e-12)).numpy()
        vi = W_dec[shared_idx, i, :]
        vj = W_dec[shared_idx, j, :]
        pairwise_cos[(i,j)] = torch.nn.functional.cosine_similarity(vi, vj, dim=1).numpy()

    # ─── Density computation over a single Tensor or DataLoader ─────────────────
    densities = None
    if data is not None:
        densities = torch.zeros(D)
        total = 0

        # helper to process a batch of shape [B, M, A]
        def process_batch(X_batch):
            nonlocal densities, total
            B, M_, A = X_batch.shape
            assert M_ == M, f"expected {M} models dim, got {M_}"
            for m in range(M):
                Xm = X_batch[:, m, :]                 # [B, A]
                f  = torch.relu(Xm @ W_enc[m] + b_enc) # [B, D]
                densities += (f > density_threshold).sum(0).float()
                total += B

        for batch in data:
            # if loader yields (inputs, labels), grab inputs
            Xb = batch if isinstance(batch, torch.Tensor) else batch[0]
            process_batch(Xb.cpu())

        densities /= total
        densities = densities.numpy()

    # ─── Plotting ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18,10), constrained_layout=True)
    for col, (i, j) in enumerate(pairs):
        xi, xj = names[i], names[j]

        ax = axes[0, col]
        ax.hist(pairwise_rel[(i,j)], bins=100)
        ax.set_yscale("log")
        ax.set_xlabel(f"{xi}/({xi}+{xj})")
        if col == 0: ax.set_ylabel("Number of features")
        ax.set_title(f"Rel‐norm: {xi} vs {xj}")

        ax = axes[1, col]
        ax.hist(pairwise_cos[(i,j)], bins=100)
        ax.set_yscale("log")
        ax.set_xlabel("Cosine similarity")
        if col == 0: ax.set_ylabel("Number of features")
        ax.set_title(f"Cos‐sim: {xi} vs {xj}")

    fig.suptitle("Pairwise Decoder‐Vector Comparisons", fontsize=18)
    plt.show()

    if densities is not None:
        plot_feature_density_log10(
            densities,
            shared_idx.numpy(),
            {m: idx.numpy() for m, idx in exclusive_idx.items()},
            names
        )

    # ─── Print feature‐index lists ───────────────────────────────────────────────
    print(f"🔹 {len(shared_idx)} shared features: {shared_idx.tolist()}")
    for m in range(M):
        idx = exclusive_idx[m]
        print(f"🔸 {len(idx)} exclusive to {names[m]}: {idx.tolist()}")
    print(f"✨ {len(new_idx)} “new” features (exclusive to interpolated): {new_idx.tolist()}")

    return {
        "shared_idx":      shared_idx,
        "exclusive_idx":   exclusive_idx,
        "new_idx":         new_idx,
        "pairwise_rel":    pairwise_rel,
        "pairwise_cos":    pairwise_cos,
        "densities":       densities,
    }
