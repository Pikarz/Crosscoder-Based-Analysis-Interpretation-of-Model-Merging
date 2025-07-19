from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import torch
import seaborn as sns

# -----------------------------------------------------------------------------
#                                CORE CLASS
# -----------------------------------------------------------------------------
class CrossCoderAnalysis:
    """Light‑weight analysis for a 3‑model Cross‑Coder."""

    # ----------------------------- construction ----------------------------- #
    def __init__(
        self,
        crosscoder,
        model_names: List[str] | None = None,
        *,
        active_thresh: float = 0.05,
    ) -> None:
        self.crosscoder = crosscoder
        self.W_dec: torch.Tensor = crosscoder.W_dec  # [latent_dim, n_models, n_act]

        # Basic dims --------------------------------------------------------- #
        self.latent_dim, self.n_models, self.n_act = self.W_dec.shape
        if self.n_models != 3:
            raise ValueError("This refactor only supports exactly **three** models")

        self.model_names: List[str] = (
            model_names if model_names is not None else [f"Model{i}" for i in range(3)]
        )
        self.active_thresh = active_thresh

        # Pre‑compute norms --------------------------------------------------- #
        self.decoder_norms: torch.Tensor = torch.norm(self.W_dec, dim=2)  # [latent, n_models]
        total_norms = torch.clamp(self.decoder_norms.sum(dim=1, keepdim=True), min=1e-8)
        self.rel_norms: torch.Tensor = self.decoder_norms / total_norms  # [latent, n_models]

        # Unified feature classification ------------------------------------ #
        self.features: Dict = self._classify_all_features()

    # --------------------------------------------------------------------- #
    #                            CLASSIFICATION
    # --------------------------------------------------------------------- #
    def _classify_all_features(self) -> Dict:
        """Return the dictionary described at the top of the file."""
        feats = {
            "exclusive": {m: [] for m in self.model_names},
            "pair_exclusive": {
                tuple(sorted((a, b))): [] for a, b in combinations(self.model_names, 2)
            },
            "shared": [],
        }

        for idx in range(self.latent_dim):
            active = [i for i in range(3) if self.rel_norms[idx, i] >= self.active_thresh]
            if len(active) == 1:  # single‑model exclusive ------------------ #
                feats["exclusive"][self.model_names[active[0]]].append(idx)
            elif len(active) == 2:  # pair‑exclusive ------------------------ #
                pair = tuple(sorted((self.model_names[active[0]], self.model_names[active[1]])))
                feats["pair_exclusive"][pair].append(idx)
            else:  # len == 3  -> shared ------------------------------------ #
                feats["shared"].append(idx)

        return feats

    # --------------------------------------------------------------------- #
    #                               PLOTS
    # --------------------------------------------------------------------- #
    def _palette(self) -> Dict[str, str]:
        """Colour palette for 7 categories."""
        ex_cols = {
            self.model_names[0]: "#E74C3C",  # red  – dice    (default order)
            self.model_names[1]: "#9B59B6",  # purple – pokemon
            self.model_names[2]: "#F1C40F",  # yellow – merged
        }
        pair_cols = {
            tuple(sorted((self.model_names[0], self.model_names[1]))): "#FC8C3A",  # orange
            tuple(sorted((self.model_names[0], self.model_names[2]))): "#8BC34A",  # green
            tuple(sorted((self.model_names[1], self.model_names[2]))): "#4BADE8",  # blue
        }
        return {
            **ex_cols,
            **pair_cols,
            "shared": "#7F8C8D",  # grey
        }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def plot_3d_feature_space(self, *, figsize: Tuple[int, int] = (10, 9)) -> plt.Figure:  # unchanged
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        rs = self.rel_norms.cpu().detach().numpy()
        colours: list[str] = []
        labels: list[str] = []
        pal = self._palette()

        # Pre‑compute quick lookup dicts
        excl = {idx: m for m, lst in self.features["exclusive"].items() for idx in lst}
        pair = {
            idx: pair_name
            for pair_name, lst in self.features["pair_exclusive"].items()
            for idx in lst
        }
        shared_set = set(self.features["shared"])

        for f in range(self.latent_dim):
            if f in excl:
                m_name = excl[f]
                colours.append(pal[m_name])
                labels.append(f"{m_name} excl.")
            elif f in pair:
                pair_name = pair[f]
                colours.append(pal[pair_name])
                labels.append("+".join(pair_name))
            elif f in shared_set:
                colours.append(pal["shared"])
                labels.append("shared")
            else:  # fallback (shouldn't happen)
                colours.append("#000000")
                labels.append("unclassified")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(rs[:, 0], rs[:, 1], rs[:, 2], c=colours, s=18, alpha=0.7)
        ax.set_xlabel(f"{self.model_names[0]} rel‑norm")
        ax.set_ylabel(f"{self.model_names[1]} rel‑norm")
        ax.set_zlabel(f"{self.model_names[2]} rel‑norm")
        ax.set_title("Relative decoder norms (3‑D)")

        # legend
        legend_items: Dict[str, str] = {}
        for c, lab in zip(colours, labels):
            legend_items.setdefault(lab, c)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=8, label=lab)
            for lab, col in sorted(legend_items.items())
        ]
        ax.legend(handles=handles, loc="upper right", framealpha=0.9)
        return fig

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def plot_relative_norm_histograms(self, *, bins: int = 50, figsize: Tuple[int, int] = (14, 4)) -> plt.Figure:  # unchanged
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        for ax, (i, j) in zip(axes, combinations(range(3), 2)):
            pair_name = f"{self.model_names[i]} vs {self.model_names[j]}"
            pair_norms = self.decoder_norms[:, [i, j]]
            rel = pair_norms[:, 0] / torch.clamp(pair_norms.sum(dim=1), min=1e-8)
            r = rel.cpu().detach().numpy()
            ax.hist(r, bins=bins, alpha=0.75, edgecolor="black")
            ax.axvline(0.05, ls="--", color="#E74C3C", lw=1)
            ax.axvline(0.95, ls="--", color="#4BADE8", lw=1)
            ax.set_title(pair_name)
            ax.set_xlabel(f"rel‑norm ({self.model_names[i]})")
        axes[0].set_ylabel("#features")
        fig.suptitle("Relative‑norm histograms (unordered pairs)")
        fig.tight_layout()
        return fig

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def plot_cosine_similarity(self, *, figsize: Tuple[int, int] = (14, 4)) -> plt.Figure:  # unchanged
        shared = self.features["shared"]
        if not shared:
            raise RuntimeError("No shared features under the current active‑threshold.")
        W = self.W_dec[shared]  # [n_shared, 3, n_act]
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        for ax, (i, j) in zip(axes, combinations(range(3), 2)):
            a = torch.nn.functional.normalize(W[:, i, :], dim=1)
            b = torch.nn.functional.normalize(W[:, j, :], dim=1)
            cos = (a * b).sum(dim=1).cpu().detach().numpy()
            ax.hist(cos, bins=40, alpha=0.75, edgecolor="black")
            ax.set_title(f"{self.model_names[i]} vs {self.model_names[j]}")
            ax.set_xlabel("cosine similarity")
        axes[0].set_ylabel("#features")
        fig.suptitle("Cosine similarity – shared features")
        fig.tight_layout()
        return fig

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def plot_feature_density_log(self,
                             activations: torch.Tensor,
                             figsize: Tuple[int,int] = (10,6),
                             activation_threshold: float = 1,
                             eps: float = 1e-8) -> plt.Figure:
        """
        Step‐plot tratteggiato in log10, con istogrammi normalizzati (density=True),
        così le esclusive non vengono schiacciate.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        freq = torch.mean((activations > activation_threshold).float(), dim=0)
        freq = freq.detach().cpu().numpy()
        logf = np.log10(freq + eps)

        shared_idx = self.features['shared']
        excl_idx   = self.features['exclusive']

        logf_shared = logf[shared_idx] if shared_idx else np.array([])
        logf_excl   = {
            m: logf[excl_idx[m]] if excl_idx[m] else np.array([])
            for m in self.model_names
        }

        all_vals = np.concatenate([logf_shared] + list(logf_excl.values()))
        bins = np.linspace(all_vals.min(), all_vals.max(), 50)

        shared_color = '#888888'
        model_colors = {
            self.model_names[0]: '#d62728',
            self.model_names[1]: '#1f77b4',
            self.model_names[2]: '#2ca02c',
        }

        fig, ax = plt.subplots(1,1,figsize=figsize)
        sns.set_style('whitegrid')

        # Shared in step‑dashed, normalizzato
        if logf_shared.size > 0:
            ax.hist(logf_shared,
                    bins=bins,
                    histtype='step',
                    linestyle='--',
                    linewidth=1.5,
                    color=shared_color,
                    density=True,        
                    label=f'Shared (n={len(logf_shared)})')

        # Exclusive per modello, normalizzati
        for m in self.model_names:
            vals = logf_excl[m]
            if vals.size > 0:
                ax.hist(vals,
                        bins=bins,
                        histtype='step',
                        linestyle='--',
                        linewidth=1.5,
                        color=model_colors[m],
                        density=True,   
                        label=f'{m} exclusive (n={len(vals)})')

        ax.set_xlabel('log\u2081\u2080 Feature Density', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Feature densities, shared and model‐exclusive (log10)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def _collect_activations(self, activations) -> torch.Tensor:
        """Return a 2‑D tensor `[n_samples, latent_dim]`.

        Accepts a tensor **or** any iterable / DataLoader yielding tensors.
        """
        if isinstance(activations, torch.Tensor):
            return activations
        from torch.utils.data import DataLoader

        rows = []
        if isinstance(activations, DataLoader):
            for batch in activations:
                rows.append(batch[0] if isinstance(batch, (list, tuple)) else batch)
        else:  # generic iterable
            for a in activations:
                rows.append(a)
        return torch.cat(rows, 0)
    
    # def _build_taxonomy(self, thresh: float = 0.95):
    #     """Populate `self.features` with indices for each category."""
    #     trio = self.model_names
    #     rel = self.rel_norms
    #     for idx in range(self.latent_dim):
    #         winners = (rel[idx] > thresh).nonzero(as_tuple=True)[0].tolist()
    #         if len(winners) == 3:
    #             self.features["shared"].append(idx)
    #         elif len(winners) == 2:
    #             pair = tuple(sorted([trio[i] for i in winners]))
    #             self.features["pair_only"].setdefault(pair, []).append(idx)
    #         elif len(winners) == 1:
    #             self.features["exclusive"][trio[winners[0]]].append(idx)
            # else: ignore uncertain features

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def plot_overview_dashboard(self, *, figsize: Tuple[int, int] = (10, 7)) -> plt.Figure:
        """Pie of categories + bar of shared count per pair."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ---- pie (7 slices) ---------------------------------------------- #
        sizes: List[int] = []
        labels: List[str] = []
        colours: List[str] = []
        pal = self._palette()
        # exclusives first
        for m in self.model_names:
            sizes.append(len(self.features["exclusive"][m]))
            labels.append(f"{m} excl.")
            colours.append(pal[m])
        # pair‑excl
        for pair in combinations(self.model_names, 2):
            pair = tuple(sorted(pair))
            sizes.append(len(self.features["pair_exclusive"][pair]))
            labels.append("+".join(pair))
            colours.append(pal[pair])
        # shared
        sizes.append(len(self.features["shared"]))
        labels.append("shared")
        colours.append(pal["shared"])

        ax1.pie(sizes, labels=labels, colors=colours, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Feature taxonomy – global")

        # ---- bar of shared per pair -------------------------------------- #
        shared_counts = [
            len(
                [f for f in self.features["shared"] if all(self.rel_norms[f, i] >= self.active_thresh for i in pair_idx)]
            )
            for pair_idx in combinations(range(3), 2)
        ]
        pair_labels = ["\nvs\n".join(pair) for pair in combinations(self.model_names, 2)]
        ax2.bar(pair_labels, shared_counts, color="#3498DB", alpha=0.8)
        ax2.set_ylabel("#shared features")
        ax2.set_title("Shared count per pair")

        fig.suptitle("Cross‑Coder feature overview", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig


# -----------------------------------------------------------------------------
#                         ONE‑SHOT WRAPPER
# -----------------------------------------------------------------------------
def analyze_crosscoder(
    crosscoder,
    loader=None,
    model_names: List[str] = ("dice", "pokemon", "merged"),
):
    analysis = CrossCoderAnalysis(crosscoder, list(model_names))

    print("\n»» 3‑D scatter …")
    analysis.plot_3d_feature_space()
    plt.show()

    print("\n»» Relative‑norm histograms …")
    analysis.plot_relative_norm_histograms()
    plt.show()

    print("\n»» Cosine similarity …")
    try:
        analysis.plot_cosine_similarity()
        plt.show()
    except RuntimeError as e:
        print("   ↳", e)

    if loader is not None:
        print("\n»» Feature density …")
        all_original_activations = []
        for i, data in enumerate(loader):
            if i >= 100:  # Take only a few datapoints
                break
            with torch.no_grad():
                from torch.nn.functional import relu 
                acts = relu(crosscoder.encode(data))
                all_original_activations.append(acts)
        
        all_original_activations = torch.cat(all_original_activations, dim=0)
        try:
            analysis.plot_feature_density_log(all_original_activations)
            plt.show()
        except Exception as e:
            print(f"   ↳ Could not plot feature density: {e}")
    else:
        print("   ↳ No activations provided, skipping feature density plot")

    print("\n»» Overview dashboard …")
    analysis.plot_overview_dashboard()
    plt.show()

    # concise text summary -------------------------------------------------- #
    count=0
    for k,v in analysis.features['exclusive'].items():
        count += len(v)
    for k,v in analysis.features['pair_exclusive'].items():
        count += len(v)
    count += len(analysis.features['shared'])
    print(f'COUNT: {count}')

    tot   = analysis.latent_dim
    excl  = {m: len(analysis.features["exclusive"][m]) for m in model_names}
    pairs = {"+".join(p): len(v) for p, v in analysis.features["pair_exclusive"].items()}
    shared = len(analysis.features["shared"])
    print("\n=== SUMMARY ===")
    for m in model_names:
        print(f"{m:>8}: {excl[m]:4d} excl. ({excl[m]/tot*100:5.1f}%)")
    for k, v in pairs.items():
        print(f"{k:>8}: {v:4d} pair‑excl. ({v/tot*100:5.1f}%)")
    print(f" shared: {shared:4d} ({shared/tot*100:5.1f}%)\n")

    return analysis