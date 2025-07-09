import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def analyze_crosscoder(
    crosscoder,
    datasets: List[torch.Tensor] = None,
    shared_tol: float = 0.05,
    exclusive_threshold: float = 0.90,
    density_threshold: float = 1e-3,
) -> Dict[str, object]:
    """
    As before, but now 'new' features ≡ those exclusive to INTERPOLATED (model 1).
    """
    # 1) unpack & CPU
    W_dec = crosscoder.W_dec.detach().cpu()      # [D, M, A]
    W_enc = crosscoder.W_enc.detach().cpu()      # [M, A, D]
    b_enc = crosscoder.b_enc.detach().cpu()      # [D]
    D, M, _ = W_dec.shape
    names = {0:"dice", 1:"interpolated", 2:"pokemon"}

    # 2) norms & rel‐norms
    norms     = W_dec.norm(dim=2)                             # [D, M]
    rel_norms = norms / (norms.sum(1,keepdim=True) + 1e-12)   # [D, M]

    # 3) shared & exclusive
    shared_mask   = (rel_norms - 1.0/M).abs().max(dim=1).values < shared_tol
    shared_idx    = torch.where(shared_mask)[0]
    exclusive_idx = {
        m: torch.where(rel_norms[:,m] > exclusive_threshold)[0]
        for m in range(M)
    }
    # ← change here: new = interpolated (model 1)
    new_idx = exclusive_idx[1]

    # 4) pairwise rel & cos
    pairs = [(0,1),(0,2),(1,2)]
    pairwise_rel, pairwise_cos = {}, {}
    for i,j in pairs:
        # rel‐norm ratio
        pairwise_rel[(i,j)] = (
            (rel_norms[:,i] / (rel_norms[:,i] + rel_norms[:,j] + 1e-12))
            .numpy()
        )
        # cos‐sim
        pairwise_cos[(i,j)] = torch.nn.functional.cosine_similarity(
            W_dec[:,i,:], W_dec[:,j,:], dim=1
        ).numpy()

    # 5) densities (optional)
    densities = None
    if datasets is not None:
        densities = torch.zeros(D)
        total = sum(ds.shape[0] for ds in datasets)
        for m, X in enumerate(datasets):
            f = torch.relu(X.cpu() @ W_enc[m] + b_enc)  # [N, D]
            densities += (f > density_threshold).sum(0).float()
        densities /= total
        densities = densities.numpy()

    # 6) 2×3 hist grid
    fig, axes = plt.subplots(2,3, figsize=(18,10), constrained_layout=True)
    for col,(i,j) in enumerate(pairs):
        xi, xj = names[i], names[j]
        # rel‐norm
        ax = axes[0,col]
        ax.hist(pairwise_rel[(i,j)], bins=100)
        ax.set_yscale("log")
        ax.set_xlabel(f"{xi}/({xi}+{xj})")
        if col==0: ax.set_ylabel("Number of features")
        ax.set_title(f"Rel-norm: {xi} vs {xj}")
        # cos‐sim
        ax = axes[1,col]
        ax.hist(pairwise_cos[(i,j)], bins=100)
        ax.set_yscale("log")
        ax.set_xlabel("Cosine similarity")
        if col==0: ax.set_ylabel("Number of features")
        ax.set_title(f"Cos-sim: {xi} vs {xj}")

    fig.suptitle("Pairwise Decoder-Vector Comparisons", fontsize=18)
    plt.show()

    # 7) density‐by‐category
    if densities is not None:
        plt.figure(figsize=(6,4))
        plt.hist(densities[shared_idx],       bins=50, alpha=0.6, label="shared")
        for m in range(M):
            idx = exclusive_idx[m]
            plt.hist(densities[idx], bins=50, alpha=0.6, label=f"exclusive {names[m]}")
        plt.yscale("log")
        plt.xlabel("Activation density")
        plt.ylabel("Number of features")
        plt.legend()
        plt.title("Feature Activation Densities by Category")
        plt.show()

    # 8) print lists
    print(f"🔹 {len(shared_idx)} shared features: {shared_idx.tolist()}")
    for m in range(M):
        idx = exclusive_idx[m]
        print(f"🔸 {len(idx)} exclusive to {names[m]}: {idx.tolist()}")
    print(f"✨ {len(new_idx)} “new” features (exclusive to interpolated): {new_idx.tolist()}")

    return {
        "shared_idx":      shared_idx,
        "exclusive_idx":   exclusive_idx,
        "new_idx":         new_idx,
        "relative_norms":  rel_norms,
        "pairwise_rel":    pairwise_rel,
        "pairwise_cos":    pairwise_cos,
        "densities":       densities,
    }
