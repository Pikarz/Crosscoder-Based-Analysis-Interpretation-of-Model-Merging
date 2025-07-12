import torch
import matplotlib.pyplot as plt

def analyze_crosscoder(
    crosscoder,
    shared_tol = 0.05,
    exclusive_threshold = 0.95
):

    # --- Unpack decoder ---------------------------------------------------------
    W_dec = crosscoder.W_dec.detach().cpu()   # [D, M, A]
    D, M, _ = W_dec.shape
    names = {0: "dice", 1: "interpolated", 2: "pokemon"}

    # --- Norms & relative norms -------------------------------------------------
    norms     = W_dec.norm(dim=2)                                # [D, M]
    rel_norms = norms / (norms.sum(1, keepdim=True) + 1e-12)     # [D, M]

    # --- Shared vs exclusive indices --------------------------------------------
    shared_mask  = (rel_norms - 1.0/M).abs().max(dim=1).values < shared_tol
    shared_idx   = torch.where(shared_mask)[0]
    exclusive_idx= {
        m: torch.where(rel_norms[:, m] > exclusive_threshold)[0]
        for m in range(M)
    }
    new_idx = exclusive_idx[1]  # “interpolated” as new

    # --- Pairwise rel‐norm & cos‐sim ---------------------------------------------
    pairs = [(0,1),(0,2),(1,2)]
    pairwise_rel, pairwise_cos = {}, {}
    for i, j in pairs:
        pairwise_rel[(i,j)] = (rel_norms[:,i] / (rel_norms[:,i] + rel_norms[:,j] + 1e-12)).numpy()
        vi = W_dec[shared_idx, i, :]
        vj = W_dec[shared_idx, j, :]
        pairwise_cos[(i,j)] = torch.nn.functional.cosine_similarity(vi, vj, dim=1).numpy()

    # --- Plotting ----------------------------------------------------------------
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

    # --- Print feature‐index lists -----------------------------------------------
    print(f"🔹 {len(shared_idx)} shared features: {shared_idx.tolist()}")
    for m in range(M):
        idx = exclusive_idx[m]
        print(f"🔸 {len(idx)} exclusive to {names[m]}: {idx.tolist()}")
    print(f"✨ {len(new_idx)} “new” features (exclusive to interpolated): {new_idx.tolist()}")