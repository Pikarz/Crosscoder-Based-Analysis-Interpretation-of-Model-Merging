import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class CrossCoderAnalysis:
    """Enhanced analysis functions for CrossCoder model diffing on ResNet models."""
    
    def __init__(self, crosscoder, model_names: List[str] = None):
        self.crosscoder = crosscoder
        self.W_dec = crosscoder.W_dec  # [latent_dim, n_models, n_activations]
        self.W_enc = crosscoder.W_enc  # [n_models, n_activations, latent_dim]
        
        self.latent_dim, self.n_models, self.n_activations = self.W_dec.shape
        
        if model_names is None:
            self.model_names = [f'Model_{i}' for i in range(self.n_models)]
        else:
            self.model_names = model_names
            
        # Compute decoder norms for each model
        self.decoder_norms = torch.norm(self.W_dec, dim=2)  # [latent_dim, n_models]
        
        # Compute relative decoder norms
        self.relative_norms = self._compute_relative_norms()
        
        # Classify features for pairwise comparisons
        self.feature_classifications_pairwise = self._classify_features_pairwise()
        
        # Classify features for 3-way comparison given the pairwise
        self.feature_classifications_3way = self._classify_features_3way()
        
    def _compute_relative_norms(self) -> torch.Tensor:
        """Compute relative decoder norms for feature classification."""
        # For each feature, compute relative norm for each model
        total_norms = torch.sum(self.decoder_norms, dim=1, keepdim=True)  # [latent_dim, 1]
        
        # Avoid division by zero
        total_norms = torch.clamp(total_norms, min=1e-8)
        
        relative_norms = self.decoder_norms / total_norms  # [latent_dim, n_models]
        return relative_norms
    
    def _classify_features_3way(self) -> Dict:
        """Original 3-way classification."""
        classifications = {
            'shared':    { 
                'pokemon_dice_merged': set(),
                'pokemon_vs_dice': set(),
                'dice_vs_merged' : set(),
                'pokemon_vs_merged': set()
             },
            'exclusive': {model_name: set() for model_name in self.model_names}
        }
        
        for pair_name, features in self.feature_classifications_pairwise.items():
            ### Compute 3-way shared features
            if len(classifications['shared']['pokemon_dice_merged']) == 0:
                classifications['shared']['pokemon_dice_merged'] = set(features['shared'])
            else:
                classifications['shared']['pokemon_dice_merged'] = set.intersection(classifications['shared']['pokemon_dice_merged'], features['shared'])
            
            # Compute pair-wise shared features
            classifications['shared'][pair_name] = set(features['shared']).difference()

            for model in self.model_names:
                if model not in pair_name:
                    continue # the pairwise is not about this model
                if len(classifications['exclusive'][model]) == 0: # first step, we just add everything
                    classifications['exclusive'][model] = set(features['exclusive'][model])
                else: # second step, we take the intersection to get the features that are actually exclusive for a given model respect to the others two
                    classifications['exclusive'][model] = set.intersection(classifications['exclusive'][model], set(features['exclusive'][model]))

        # return to list
        classifications['shared'] = list(classifications['shared'])
        for model_name in self.model_names:
            classifications['exclusive'][model_name] = list(classifications['exclusive'][model_name])
        
        return classifications
                    

    def _classify_features_pairwise(self, exclusive_threshold: float = 0.95) -> Dict:
        """Pairwise classification between all model pairs."""
        pairwise_classifications = {}
        
        # Generate all possible pairs
        model_pairs = list(combinations(range(self.n_models), 2))
        
        for i, j in model_pairs:
            model_i_name = self.model_names[i]
            model_j_name = self.model_names[j]
            pair_name = f"{model_i_name}_vs_{model_j_name}"
            
            # Get relative norms for this pair only
            pair_norms = self.decoder_norms[:, [i, j]]  # [latent_dim, 2]
            pair_total_norms = torch.sum(pair_norms, dim=1, keepdim=True)
            pair_total_norms = torch.clamp(pair_total_norms, min=1e-8)
            pair_relative_norms = pair_norms / pair_total_norms
            
            classifications = {
                'shared': [],
                'exclusive': {model_i_name: [], model_j_name: []}
            }
            
            for feature_idx in range(self.latent_dim):
                rel_norms = pair_relative_norms[feature_idx]
                max_norm = torch.max(rel_norms)
                max_model_idx = torch.argmax(rel_norms)
                
                if max_norm > exclusive_threshold:
                    # Feature is exclusive to one model in this pair
                    if max_model_idx == 0:
                        classifications['exclusive'][model_i_name].append(feature_idx)
                    else:
                        classifications['exclusive'][model_j_name].append(feature_idx)
                else:
                    # Feature is shared between these two models
                    classifications['shared'].append(feature_idx)
            
            pairwise_classifications[pair_name] = classifications
            
        return pairwise_classifications
    
    def plot_pairwise_comparison_matrix(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create a matrix visualization of pairwise comparisons."""
        n_pairs = len(self.feature_classifications_pairwise)
        
        # Calculate grid size
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_pairs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        for idx, (pair_name, classifications) in enumerate(self.feature_classifications_pairwise.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Extract model names from pair
            model_names = pair_name.split('_vs_')
            model_1, model_2 = model_names[0], model_names[1]
            
            # Get counts
            shared_count = len(classifications['shared'])
            exclusive_1_count = len(classifications['exclusive'][model_1])
            exclusive_2_count = len(classifications['exclusive'][model_2])
            
            # Create pie chart
            sizes = [shared_count, exclusive_1_count, exclusive_2_count]
            labels = [f'Shared\n({shared_count})', f'{model_1} Exclusive\n({exclusive_1_count})', 
                     f'{model_2} Exclusive\n({exclusive_2_count})']
            colors = ['#2E86AB', '#F24236', '#F6AE2D']
            
            # Only plot non-zero slices
            non_zero_sizes = [s for s in sizes if s > 0]
            non_zero_labels = [l for i, l in enumerate(labels) if sizes[i] > 0]
            non_zero_colors = [c for i, c in enumerate(colors) if sizes[i] > 0]
            
            if non_zero_sizes:
                ax.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, 
                      autopct='%1.1f%%', startangle=90)
            else:
                ax.text(0.5, 0.5, 'No features\nfound', ha='center', va='center', 
                       transform=ax.transAxes)
            
            ax.set_title(f'{model_1} vs {model_2}', fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(self.feature_classifications_pairwise), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_feature_space(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Create a 3D visualization of feature classifications (for 3 models)."""
        if self.n_models != 3:
            print(f"3D visualization only available for 3 models, found {self.n_models}")
            return None
            
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get relative norms for all features
        rel_norms = self.relative_norms.detach().cpu().numpy()  # [latent_dim, 3]
        
        # Color code by 3-way classification
        colors = []
        labels = []

        print(f"3WAY EXCLUSIVE: \n{self.feature_classifications_3way['exclusive']}")
        print(f"PAIR EXCLUSIVE: \n{self.feature_classifications_pairwise}")
        
        for feature_idx in range(self.latent_dim):
            if feature_idx in self.feature_classifications_3way['shared']:
                colors.append('#2E86AB')  # Blue for shared
                labels.append('Shared')
            else:
                # Find which model this feature is exclusive to
                for model_idx, model_name in enumerate(self.model_names):
                    if feature_idx in self.feature_classifications_3way['exclusive'][model_name]:
                        if model_idx == 0:
                            colors.append('#F24236')  # Red for model 0
                            labels.append(f'{model_name} Exclusive')
                        elif model_idx == 1:
                            colors.append('#F6AE2D')  # Yellow for model 1
                            labels.append(f'{model_name} Exclusive')
                        else:
                            colors.append('#A23B72')  # Purple for model 2
                            labels.append(f'{model_name} Exclusive')
                        break
        
        # Plot points
        scatter = ax.scatter(rel_norms[:, 0], rel_norms[:, 1], rel_norms[:, 2], 
                           c=colors, alpha=0.6, s=20)
        
        # Set labels
        ax.set_xlabel(f'{self.model_names[0]} Relative Norm', fontsize=12)
        ax.set_ylabel(f'{self.model_names[1]} Relative Norm', fontsize=12)
        ax.set_zlabel(f'{self.model_names[2]} Relative Norm', fontsize=12)
        ax.set_title('3D Feature Space: Relative Decoder Norms', fontsize=14, fontweight='bold')
        
        # Add legend
        unique_labels = list(set(labels))
        unique_colors = []
        for label in unique_labels:
            idx = labels.index(label)
            unique_colors.append(colors[idx])
        
        # Create legend elements
        legend_elements = []
        for color, label in zip(unique_colors, unique_labels):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8, label=label))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        return fig
    
    def plot_cosine_similarity_shared_features(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot cosine similarity for shared features (enhanced version)."""
        shared_indices = self.feature_classifications_3way['shared']
        
        if len(shared_indices) == 0:
            print("No shared features found!")
            return None
        
        # Extract decoder weights for shared features
        shared_decoders = self.W_dec[shared_indices]  # [n_shared, n_models, n_activations]
        
        # Compute cosine similarities between models for each shared feature
        cosine_similarities = {}
        all_similarities = []
        
        for i in range(self.n_models):
            for j in range(i + 1, self.n_models):
                model_i_name = self.model_names[i]
                model_j_name = self.model_names[j]
                
                # Get decoder vectors for both models
                dec_i = shared_decoders[:, i, :]  # [n_shared, n_activations]
                dec_j = shared_decoders[:, j, :]  # [n_shared, n_activations]
                
                # Normalize vectors
                dec_i_norm = torch.nn.functional.normalize(dec_i, dim=1)
                dec_j_norm = torch.nn.functional.normalize(dec_j, dim=1)
                
                # Compute cosine similarity
                cos_sim = torch.sum(dec_i_norm * dec_j_norm, dim=1)
                cos_sim = cos_sim.detach().cpu().numpy()
                
                cosine_similarities[f'{model_i_name}_vs_{model_j_name}'] = cos_sim
                all_similarities.extend(cos_sim)
        
        # Create subplots
        n_pairs = len(cosine_similarities)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Histogram of all cosine similarities
        axes[0].hist(all_similarities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Cosine Similarities\nfor Shared Features')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(np.mean(all_similarities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_similarities):.3f}')
        axes[0].legend()
        
        # Plot 2-4: Individual model pair comparisons
        for idx, (pair_name, similarities) in enumerate(cosine_similarities.items()):
            if idx + 1 < len(axes):
                axes[idx + 1].hist(similarities, bins=30, alpha=0.7, edgecolor='black')
                axes[idx + 1].set_xlabel('Cosine Similarity')
                axes[idx + 1].set_ylabel('Frequency')
                axes[idx + 1].set_title(f'{pair_name}\nMean: {np.mean(similarities):.3f}')
                axes[idx + 1].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(cosine_similarities) + 1, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_density(self, activations: torch.Tensor, figsize: Tuple[int, int] = (10, 6), 
                        activation_threshold: float = 1e-6) -> plt.Figure:
        """Enhanced feature density plot with pairwise comparisons."""
        print(f"Activations shape: {activations.shape}")
        print(f"Activations stats: min={activations.min():.6f}, max={activations.max():.6f}, mean={activations.mean():.6f}")
        print(f"Non-zero activations: {(activations != 0).sum().item()} / {activations.numel()}")
        
        # Compute activation frequencies (density) using the threshold
        activation_freq = torch.mean((activations > activation_threshold).float(), dim=0)  # [latent_dim]
        activation_freq = activation_freq.detach().cpu().numpy()
        
        print(f"Feature density stats: min={activation_freq.min():.6f}, max={activation_freq.max():.6f}, mean={activation_freq.mean():.6f}")
        print(f"Features with non-zero density: {(activation_freq > 0).sum()} / {len(activation_freq)}")
        
        # Separate by feature type (3-way)
        shared_indices = self.feature_classifications_3way['shared']
        shared_densities = activation_freq[shared_indices] if shared_indices else []
        
        # Combine all exclusive features into one group
        all_exclusive_indices = []
        for model_name in self.model_names:
            exclusive_indices = self.feature_classifications_3way['exclusive'][model_name]
            all_exclusive_indices.extend(exclusive_indices)
        
        exclusive_densities = activation_freq[all_exclusive_indices] if all_exclusive_indices else []
        
        # Filter out zero densities
        shared_densities = [d for d in shared_densities if d > 0]
        exclusive_densities = [d for d in exclusive_densities if d > 0]
        
        print(f"Shared features with non-zero density: {len(shared_densities)}")
        print(f"Exclusive features with non-zero density: {len(exclusive_densities)}")
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if not len(shared_densities) > 0 and not len(exclusive_densities) > 0:
            ax.text(0.5, 0.5, 'No features with non-zero density found!\nTry adjusting the activation_threshold parameter.', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xlabel('Feature density', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title('Feature density, shared and model-exclusive features', fontsize=14, fontweight='bold')
            return fig
        
        # Determine if we should use log scale
        all_densities = shared_densities + exclusive_densities
        min_density = min(all_densities)
        max_density = max(all_densities)
        
        use_log_scale = (min_density > 0) and (max_density / min_density > 10)
        
        if use_log_scale:
            min_log = np.log10(max(min_density, 1e-8))
            max_log = np.log10(max_density)
            bins = np.logspace(min_log, max_log, 50)
        else:
            bins = np.linspace(min_density, max_density, 50)
        
        # Plot shared features
        if len(shared_densities) > 0:
            ax.hist(shared_densities, bins=bins, alpha=0.7, label=f'Shared features (n={len(shared_densities)})', 
                color='#2E86AB', density=True, edgecolor='black', linewidth=0.5)
        
        # Plot exclusive features
        if len(exclusive_densities) > 0:
            ax.hist(exclusive_densities, bins=bins, alpha=0.7, 
                label=f'Model-exclusive features (n={len(exclusive_densities)})', 
                color='#F24236', density=True, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Feature density', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        
        scale_note = " (log scale)" if use_log_scale else " (linear scale)"
        ax.set_title(f'Feature density, shared and model-exclusive features{scale_note}', 
                    fontsize=14, fontweight='bold')
        
        if use_log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3, which='both' if use_log_scale else 'major')
        ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        
        if len(shared_densities) > 0 and len(exclusive_densities) > 0:
            shared_median = np.median(shared_densities)
            exclusive_median = np.median(exclusive_densities)
            
            stats_text = f'Median density:\nShared: {shared_median:.6f}\nExclusive: {exclusive_median:.6f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
        
    def get_feature_summary(self) -> Dict:
        """Get a comprehensive summary of both 3-way and pairwise analysis."""
        summary = {
            'total_features': self.latent_dim,
            '3way_analysis': {
                'shared_features': len(self.feature_classifications_3way['shared']),
                'exclusive_features': {},
                'shared_percentage': len(self.feature_classifications_3way['shared']) / self.latent_dim * 100
            },
            'pairwise_analysis': {}
        }
        
        # 3-way analysis
        for model_name in self.model_names:
            count = len(self.feature_classifications_3way['exclusive'][model_name])
            summary['3way_analysis']['exclusive_features'][model_name] = count
            summary['3way_analysis'][f'{model_name}_exclusive_percentage'] = count / self.latent_dim * 100
        
        # Pairwise analysis
        for pair_name, classifications in self.feature_classifications_pairwise.items():
            shared_count = len(classifications['shared'])
            exclusive_counts = {model: len(indices) for model, indices in classifications['exclusive'].items()}
            
            summary['pairwise_analysis'][pair_name] = {
                'shared_features': shared_count,
                'shared_percentage': shared_count / self.latent_dim * 100,
                'exclusive_features': exclusive_counts
            }
        
        return summary
    
    def print_summary(self):
        """Print a comprehensive formatted summary of the analysis."""
        summary = self.get_feature_summary()
        
        print("=" * 60)
        print("ENHANCED CROSSCODER ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total Features: {summary['total_features']}")
        
        print("\n" + "=" * 40)
        print("3-WAY ANALYSIS (All models together)")
        print("=" * 40)
        print(f"Shared Features: {summary['3way_analysis']['shared_features']} ({summary['3way_analysis']['shared_percentage']:.1f}%)")
        print("\nExclusive Features:")
        for model_name in self.model_names:
            count = summary['3way_analysis']['exclusive_features'][model_name]
            percentage = summary['3way_analysis'][f'{model_name}_exclusive_percentage']
            print(f"  {model_name}: {count} ({percentage:.1f}%)")
        
        print("\n" + "=" * 40)
        print("PAIRWISE ANALYSIS")
        print("=" * 40)
        for pair_name, pair_data in summary['pairwise_analysis'].items():
            print(f"\n{pair_name}:")
            print(f"  Shared: {pair_data['shared_features']} ({pair_data['shared_percentage']:.1f}%)")
            print(f"  Exclusive:")
            for model, count in pair_data['exclusive_features'].items():
                percentage = count / self.latent_dim * 100
                print(f"    {model}: {count} ({percentage:.1f}%)")
        
        print("\n" + "=" * 60)
        
        # Diagnostic information
        print("\nDIAGNOSTIC INFORMATION:")
        print(f"Decoder norms shape: {self.decoder_norms.shape}")
        print(f"Relative norms shape: {self.relative_norms.shape}")
        
        max_rel_norms = torch.max(self.relative_norms, dim=1)[0]
        print(f"Max relative norms - Min: {max_rel_norms.min():.4f}, Max: {max_rel_norms.max():.4f}, Mean: {max_rel_norms.mean():.4f}")
        
        exclusive_threshold = 0.95
        near_exclusive = torch.sum(max_rel_norms > 0.8).item()
        very_near_exclusive = torch.sum(max_rel_norms > 0.9).item()
        at_threshold = torch.sum(max_rel_norms > exclusive_threshold).item()
        
        print(f"Features with max relative norm > 0.8: {near_exclusive}")
        print(f"Features with max relative norm > 0.9: {very_near_exclusive}")
        print(f"Features with max relative norm > {exclusive_threshold}: {at_threshold}")
        
        total_decoder_norms = torch.sum(self.decoder_norms, dim=1)
        print(f"Total decoder norms - Min: {total_decoder_norms.min():.4f}, Max: {total_decoder_norms.max():.4f}")
        
        zero_norms = torch.sum(self.decoder_norms < 1e-6, dim=1)
        features_with_zero_norms = torch.sum(zero_norms > 0).item()
        print(f"Features with near-zero decoder norms: {features_with_zero_norms}")
        
        print("=" * 60)

    def _pair_exclusive_masks(self, model_i: int, model_j: int, thresh: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate exclusivity masks for a pair of models using the paper's approach.
        
        Parameters:
        -----------
        model_i, model_j : int
            Indices of the two models to compare
        thresh : float
            Threshold for exclusivity (default 0.05 as in the paper)
            
        Returns:
        --------
        mask_i, mask_j : torch.Tensor
            Boolean masks indicating features exclusive to model i and j respectively
        """
        # Get relative norms for this pair only
        pair_norms = self.decoder_norms[:, [model_i, model_j]]  # [latent_dim, 2]
        pair_total_norms = torch.sum(pair_norms, dim=1, keepdim=True)
        pair_total_norms = torch.clamp(pair_total_norms, min=1e-8)
        pair_relative_norms = pair_norms / pair_total_norms  # [latent_dim, 2]
        
        # r is the relative norm for model_i
        r = pair_relative_norms[:, 0]  # [latent_dim]
        
        # Apply thresholds as in the paper
        mask_i = r <= thresh  # exclusive to model i
        mask_j = r >= (1 - thresh)  # exclusive to model j
        
        return mask_i, mask_j
    
    def plot_all_pair_relative_norms(self, figsize: Tuple[int, int] = (15, 4)) -> plt.Figure:
        """
        Plot paper-style relative norm histograms for all model pairs.
        """
        n_pairs = self.n_models * (self.n_models - 1)  # ordered pairs
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_pairs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        plot_idx = 0
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i == j:
                    continue
                    
                if plot_idx >= len(axes):
                    break
                    
                ax = axes[plot_idx]
                
                # Get relative norms for this ordered pair
                pair_norms = self.decoder_norms[:, [i, j]]  # [latent_dim, 2]
                pair_total_norms = torch.sum(pair_norms, dim=1, keepdim=True)
                pair_total_norms = torch.clamp(pair_total_norms, min=1e-8)
                pair_relative_norms = pair_norms / pair_total_norms
                
                r = pair_relative_norms[:, 0].detach().cpu().numpy()  # relative norm for model i
                
                # Plot histogram
                ax.hist(r, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel(f'Relative norm for {self.model_names[i]}')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{self.model_names[i]} vs {self.model_names[j]}')
                ax.grid(True, alpha=0.3)
                
                # Add threshold lines
                ax.axvline(0.05, color='red', linestyle='--', alpha=0.7, 
                          label=f'{self.model_names[i]} exclusive')
                ax.axvline(0.95, color='magenta', linestyle='--', alpha=0.7, 
                          label=f'{self.model_names[j]} exclusive')
                ax.legend(fontsize=8)
                
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_pairwise_feature_analysis(self, figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
        """
        Comprehensive pairwise feature analysis dashboard.
        """
        n_pairs = len(self.feature_classifications_pairwise)
        n_cols = 3
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_pairs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (pair_name, classifications) in enumerate(self.feature_classifications_pairwise.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Extract model names from pair
            model_names = pair_name.split('_vs_')
            model_1, model_2 = model_names[0], model_names[1]
            
            # Get counts
            shared_count = len(classifications['shared'])
            exclusive_1_count = len(classifications['exclusive'][model_1])
            exclusive_2_count = len(classifications['exclusive'][model_2])
            
            # Create stacked bar chart
            categories = ['Features']
            shared_vals = [shared_count]
            excl_1_vals = [exclusive_1_count]
            excl_2_vals = [exclusive_2_count]
            
            width = 0.6
            ax.bar(categories, shared_vals, width, label=f'Shared ({shared_count})', color='#2E86AB')
            ax.bar(categories, excl_1_vals, width, bottom=shared_vals, 
                  label=f'{model_1} Exclusive ({exclusive_1_count})', color='#F24236')
            ax.bar(categories, excl_2_vals, width, 
                  bottom=[s + e1 for s, e1 in zip(shared_vals, excl_1_vals)], 
                  label=f'{model_2} Exclusive ({exclusive_2_count})', color='#F6AE2D')
            
            ax.set_title(f'{model_1} vs {model_2}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Features')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add percentage annotations
            total = shared_count + exclusive_1_count + exclusive_2_count
            if total > 0:
                shared_pct = shared_count / total * 100
                excl_1_pct = exclusive_1_count / total * 100
                excl_2_pct = exclusive_2_count / total * 100
                
                ax.text(0, shared_count/2, f'{shared_pct:.1f}%', ha='center', va='center', 
                       fontweight='bold', color='white')
                if exclusive_1_count > 0:
                    ax.text(0, shared_count + exclusive_1_count/2, f'{excl_1_pct:.1f}%', 
                           ha='center', va='center', fontweight='bold', color='white')
                if exclusive_2_count > 0:
                    ax.text(0, shared_count + exclusive_1_count + exclusive_2_count/2, 
                           f'{excl_2_pct:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        
        # Hide unused subplots
        for idx in range(len(self.feature_classifications_pairwise), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Pairwise Feature Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_triple_vs_pairwise_comparison(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compare 3-way vs pairwise analysis results.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: 3-way analysis pie chart
        ax = axes[0]
        shared_3way = len(self.feature_classifications_3way['shared'])
        exclusive_3way = [len(self.feature_classifications_3way['exclusive'][model]) 
                         for model in self.model_names]
        
        sizes = [shared_3way] + exclusive_3way
        labels = ['Shared'] + [f'{model} Exclusive' for model in self.model_names]
        colors = ['#2E86AB', '#F24236', '#F6AE2D', '#A23B72'][:len(sizes)]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('3-Way Analysis\n(All models together)', fontsize=14, fontweight='bold')
        
        # Plot 2: Pairwise shared features comparison
        ax = axes[1]
        pair_names = []
        pair_shared = []
        
        for pair_name, classifications in self.feature_classifications_pairwise.items():
            pair_names.append(pair_name.replace('_vs_', '\nvs\n'))
            pair_shared.append(len(classifications['shared']))
        
        bars = ax.bar(range(len(pair_names)), pair_shared, color='#2E86AB', alpha=0.7)
        ax.set_xlabel('Model Pairs')
        ax.set_ylabel('Number of Shared Features')
        ax.set_title('Pairwise Shared Features', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(pair_names)))
        ax.set_xticklabels(pair_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, pair_shared):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(value), ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Feature overlap heatmap
        ax = axes[2]
        overlap_matrix = np.zeros((self.n_models, self.n_models))
        
        for i in range(self.n_models):
            for j in range(self.n_models):
                if i == j:
                    model_name = self.model_names[i]
                    overlap_matrix[i, j] = len(self.feature_classifications_3way['exclusive'][model_name])
                else:
                    # Find shared features between models i and j
                    for pair_name, classifications in self.feature_classifications_pairwise.items():
                        if ((self.model_names[i] in pair_name and self.model_names[j] in pair_name) or
                            (self.model_names[j] in pair_name and self.model_names[i] in pair_name)):
                            overlap_matrix[i, j] = len(classifications['shared'])
                            break
        
        im = ax.imshow(overlap_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(self.n_models))
        ax.set_yticks(range(self.n_models))
        ax.set_xticklabels(self.model_names)
        ax.set_yticklabels(self.model_names)
        ax.set_title('Feature Overlap Matrix\n(Diagonal: Exclusive, Off-diagonal: Shared)', 
                    fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(self.n_models):
            for j in range(self.n_models):
                ax.text(j, i, f'{overlap_matrix[i, j]:.0f}', ha='center', va='center', 
                       color='white' if overlap_matrix[i, j] > overlap_matrix.max()/2 else 'black',
                       fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Plot 4: Summary statistics
        ax = axes[3]
        ax.axis('off')
        
        # Create summary text
        summary_text = f"""
ANALYSIS SUMMARY
{'='*30}

Total Features: {self.latent_dim}

3-WAY ANALYSIS:
Shared: {shared_3way} ({shared_3way/self.latent_dim*100:.1f}%)
"""
        
        for i, model in enumerate(self.model_names):
            count = exclusive_3way[i]
            pct = count/self.latent_dim*100
            summary_text += f"{model} Exclusive: {count} ({pct:.1f}%)\n"
        
        summary_text += "\nPAIRWISE ANALYSIS:\n"
        for pair_name, classifications in self.feature_classifications_pairwise.items():
            shared_count = len(classifications['shared'])
            pct = shared_count/self.latent_dim*100
            summary_text += f"{pair_name}: {shared_count} shared ({pct:.1f}%)\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig


def analyze_crosscoder(
        crosscoder,
        activations: torch.Tensor | None = None,
        model_names: list[str] = ('dice', 'pokemon', 'merged'),
        pair_exclusive_thresh: float = 0.05,
        show_pair_hist: bool = True):
    """
    Complete, one-shot analysis pipeline for a CrossCoder.

    Parameters
    ----------
    crosscoder : CrossCoder
        The trained cross-coder object (must expose W_dec / W_enc).
    activations : torch.Tensor | None
        Optional [n_samples, latent_dim] activations for density plots.
    model_names : list[str]
        Friendly names for the models in the same order as in W_dec.
    pair_exclusive_thresh : float
        r <= thresh → exclusive to model A
        r >= 1-thresh → exclusive to model B
    show_pair_hist : bool
        If True, will render the paper-style relative-norm histogram
        for every ordered pair of models.
    """
    from itertools import combinations
    
    # Initialize the enhanced analysis object
    analysis = CrossCoderAnalysis(crosscoder, model_names)

    # Text summary - triple analysis
    print("\n=== CROSSCODER ANALYSIS SUMMARY ===")
    print(f"Total latent dimensions : {analysis.latent_dim}")
    print(f"Number of models        : {analysis.n_models}")
    print(f"Number of activations   : {analysis.n_activations}")
    print(f"Model names             : {analysis.model_names}")

    print("\n=== TRIPLE ANALYSIS (all models together) ===")
    n_shared = len(analysis.feature_classifications_3way['shared'])
    print(f"Shared features         : {n_shared}")
    for m in analysis.model_names:
        n_excl = len(analysis.feature_classifications_3way['exclusive'][m])
        print(f"{m:>10} exclusive        : {n_excl}")

    # Text summary - pairwise (including new exclusivity masks)
    print("\n=== PAIRWISE ANALYSIS ===")
    for (i, mA), (j, mB) in combinations(enumerate(analysis.model_names), 2):
        mask_a, mask_b = analysis._pair_exclusive_masks(i, j, thresh=pair_exclusive_thresh)
        n_a, n_b = mask_a.sum().item(), mask_b.sum().item()
        n_shared_pair = analysis.latent_dim - n_a - n_b
        print(f"{mA} ↔ {mB}:   "
              f"shared={n_shared_pair:>5}, "
              f"{mA}-only={n_a:>5}, "
              f"{mB}-only={n_b:>5}")

    # Plots
    # 1) Full pairwise feature breakdown
    print("\n… generating comprehensive pairwise feature dashboard …")
    analysis.plot_pairwise_feature_analysis(figsize=(20, 15))
    plt.show()

    # 2) Optional paper-style relative-norm histograms
    if show_pair_hist:
        print("… generating paper-style relative-norm histograms …")
        analysis.plot_all_pair_relative_norms(figsize=(15, 4))
        plt.show()

    # 3) Triple vs pairwise bar-comparisons
    print("… generating triple-vs-pairwise comparison plot …")
    analysis.plot_triple_vs_pairwise_comparison(figsize=(15, 10))
    plt.show()

    # 4) 3D feature space visualization (if 3 models)
    if analysis.n_models == 3:
        print("… generating 3D feature space visualization …")
        fig_3d = analysis.plot_3d_feature_space(figsize=(12, 10))
        if fig_3d is not None:
            plt.show()

    # 6) Pairwise comparison matrix
    print("… generating pairwise comparison matrix …")
    analysis.plot_pairwise_comparison_matrix(figsize=(15, 10))
    plt.show()

    # 7) Cosine similarity across shared features
    print("… generating cosine-similarity plot for shared features …")
    fig_cos = analysis.plot_cosine_similarity_shared_features(figsize=(12, 8))
    if fig_cos is not None:
        plt.show()
    else:
        print("  (skipped - no shared features)")

    # 8) Feature-density plots (if we have activations)
    if activations is not None:
        print("… generating feature-density plot …")
        try:
            analysis.plot_feature_density(activations, figsize=(10, 6))
            plt.show()
        except Exception as err:
            print(f"  density plot failed: {err}")
    else:
        print("(skipping density plot - no activations provided)")

    # 9) Print comprehensive summary
    print("\n… printing comprehensive analysis summary …")
    analysis.print_summary()

    print("\n=== ANALYSIS COMPLETE ===")
    return analysis