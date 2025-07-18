import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, List
from itertools import combinations


class CrossCoderAnalysis:
    """Analysis functions for CrossCoder model diffing on ResNet models."""
    
    def __init__(self, crosscoder, model_names: List[str] = None):
        """
        Initialize the analysis with a crosscoder instance.
        
        Args:
            crosscoder: CrossCoder instance with W_dec and W_enc attributes
            model_names: List of model names (e.g., ['dice', 'pokemon', 'merged'])
        """
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
        
        # Classify features for all models (triple analysis)
        self.feature_classifications = self._classify_features()
        
        # NEW: Pairwise feature classifications
        self.pairwise_classifications = self._classify_features_pairwise()
        
    def _compute_relative_norms(self) -> torch.Tensor:
        """Compute relative decoder norms for feature classification."""
        # For each feature, compute relative norm for each model
        total_norms = torch.sum(self.decoder_norms, dim=1, keepdim=True)  # [latent_dim, 1]
        
        # Avoid division by zero
        total_norms = torch.clamp(total_norms, min=1e-8)
        
        relative_norms = self.decoder_norms / total_norms  # [latent_dim, n_models]
        return relative_norms
    
    def _classify_features(self, exclusive_threshold: float = 0.95) -> Dict:
        """
        Classify features as shared or exclusive based on relative decoder norms (triple analysis).
        
        Args:
            exclusive_threshold: Threshold for classifying features as exclusive
            
        Returns:
            Dictionary with feature classifications
        """
        classifications = {
            'shared': [],
            'exclusive': {model_name: [] for model_name in self.model_names}
        }
        
        for feature_idx in range(self.latent_dim):
            rel_norms = self.relative_norms[feature_idx]
            max_norm = torch.max(rel_norms)
            max_model_idx = torch.argmax(rel_norms)
            
            if max_norm > exclusive_threshold:
                # Feature is exclusive to the model with max norm
                model_name = self.model_names[max_model_idx]
                classifications['exclusive'][model_name].append(feature_idx)
            else:
                # Feature is shared
                classifications['shared'].append(feature_idx)
                
        return classifications
    
    def _classify_features_pairwise(self, exclusive_threshold: float = 0.95) -> Dict:
        """
        Classify features for each pair of models.
        
        Args:
            exclusive_threshold: Threshold for classifying features as exclusive
            
        Returns:
            Dictionary with pairwise feature classifications
        """
        pairwise_classifications = {}
        
        # Generate all pairwise combinations
        for (model1_idx, model1_name), (model2_idx, model2_name) in combinations(enumerate(self.model_names), 2):
            pair_key = f"{model1_name}_vs_{model2_name}"
            
            # Extract decoder norms for this pair
            pair_norms = self.decoder_norms[:, [model1_idx, model2_idx]]  # [latent_dim, 2]
            
            # Compute relative norms for this pair only
            pair_total_norms = torch.sum(pair_norms, dim=1, keepdim=True)  # [latent_dim, 1]
            pair_total_norms = torch.clamp(pair_total_norms, min=1e-8)
            pair_relative_norms = pair_norms / pair_total_norms  # [latent_dim, 2]
            
            # Classify features for this pair
            pair_classifications = {
                'shared': [],
                'exclusive': {model1_name: [], model2_name: []}
            }
            
            for feature_idx in range(self.latent_dim):
                rel_norms = pair_relative_norms[feature_idx]
                max_norm = torch.max(rel_norms)
                max_model_idx = torch.argmax(rel_norms)
                
                if max_norm > exclusive_threshold:
                    # Feature is exclusive to one model in this pair
                    if max_model_idx == 0:
                        pair_classifications['exclusive'][model1_name].append(feature_idx)
                    else:
                        pair_classifications['exclusive'][model2_name].append(feature_idx)
                else:
                    # Feature is shared between this pair
                    pair_classifications['shared'].append(feature_idx)
            
            pairwise_classifications[pair_key] = {
                'classifications': pair_classifications,
                'relative_norms': pair_relative_norms,
                'model_names': [model1_name, model2_name],
                'model_indices': [model1_idx, model2_idx]
            }
        
        return pairwise_classifications
    
    def plot_pairwise_decoder_norms_paper_style(self, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot pairwise relative decoder norms in the exact style of the crosscoder diffing paper.
        Each plot shows a single distribution where 0 = all first model, 1 = all second model.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        n_pairs = len(self.pairwise_classifications)
        
        # Create horizontal subplot layout
        fig, axes = plt.subplots(1, n_pairs, figsize=figsize)
        
        # If only one pair, ensure axes is iterable
        if n_pairs == 1:
            axes = [axes]
        
        pair_idx = 0
        for pair_key, pair_data in self.pairwise_classifications.items():
            model1_name, model2_name = pair_data['model_names']
            pair_relative_norms = pair_data['relative_norms']
            
            # Use model1's relative norm (0 = all model1, 1 = all model2)
            rel_norms = pair_relative_norms[:, 0].detach().cpu().numpy()
            
            # Create histogram with paper-style formatting
            axes[pair_idx].hist(rel_norms, bins=50, alpha=0.8, density=True, 
                              color='#2E86AB', edgecolor='none')
            
            # Paper-style labels and formatting
            axes[pair_idx].set_xlabel('Relative decoder norm', fontsize=12)
            axes[pair_idx].set_ylabel('Density', fontsize=12)
            axes[pair_idx].set_title(f'{model1_name} vs {model2_name}', fontsize=14, fontweight='bold')
            
            # Add model name annotations at the extremes
            axes[pair_idx].text(0.02, 0.98, model1_name, transform=axes[pair_idx].transAxes, 
                              verticalalignment='top', fontsize=12, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            axes[pair_idx].text(0.98, 0.98, model2_name, transform=axes[pair_idx].transAxes, 
                              verticalalignment='top', horizontalalignment='right', fontsize=12, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Clean grid
            #axesimport matplotlib.pyplot as plt
    
    def plot_pairwise_feature_analysis(self, figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
        """
        Plot comprehensive pairwise feature analysis.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        n_pairs = len(self.pairwise_classifications)
        
        # Create subplots: 4 rows x n_pairs columns
        fig, axes = plt.subplots(4, n_pairs, figsize=figsize)
        
        # If only one pair, ensure axes is 2D
        if n_pairs == 1:
            axes = axes.reshape(-1, 1)
        
        pair_idx = 0
        for pair_key, pair_data in self.pairwise_classifications.items():
            model1_name, model2_name = pair_data['model_names']
            model1_idx, model2_idx = pair_data['model_indices']
            pair_classifications = pair_data['classifications']
            pair_relative_norms = pair_data['relative_norms']
            
            # Plot 1: Bar chart of feature counts for this pair
            shared_count = len(pair_classifications['shared'])
            model1_exclusive = len(pair_classifications['exclusive'][model1_name])
            model2_exclusive = len(pair_classifications['exclusive'][model2_name])
            
            counts = [shared_count, model1_exclusive, model2_exclusive]
            labels = ['Shared', f'{model1_name}\nExclusive', f'{model2_name}\nExclusive']
            colors = ['#4CAF50', '#FF6B6B', '#4FC3F7']
            
            bars = axes[0, pair_idx].bar(range(3), counts, color=colors, alpha=0.7)
            axes[0, pair_idx].set_xlabel('Feature Type')
            axes[0, pair_idx].set_ylabel('Number of Features')
            axes[0, pair_idx].set_title(f'{model1_name} vs {model2_name}\nPairwise Feature Distribution')
            axes[0, pair_idx].set_xticks(range(3))
            axes[0, pair_idx].set_xticklabels(labels, rotation=45, ha='right')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[0, pair_idx].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                                    f'{count}', ha='center', va='bottom')
            
            # Plot 2: Relative decoder norms comparison for this pair
            rel_norms_model1 = pair_relative_norms[:, 0].detach().cpu().numpy()
            rel_norms_model2 = pair_relative_norms[:, 1].detach().cpu().numpy()
            
            axes[1, pair_idx].hist(rel_norms_model1, bins=30, alpha=0.7, 
                                label=model1_name, density=True, 
                                color='#FF6B6B', edgecolor='black', linewidth=0.5)
            axes[1, pair_idx].hist(rel_norms_model2, bins=30, alpha=0.7, 
                                label=model2_name, density=True,
                                color='#4FC3F7', edgecolor='black', linewidth=0.5)
            
            axes[1, pair_idx].set_xlabel('Relative Decoder Norm')
            axes[1, pair_idx].set_ylabel('Density')
            axes[1, pair_idx].set_title(f'{model1_name} vs {model2_name}\nPairwise Decoder Norms')
            axes[1, pair_idx].legend()
            axes[1, pair_idx].grid(True, alpha=0.3)
            
            # Plot 3: Pie chart for this pair
            sizes = [shared_count, model1_exclusive, model2_exclusive]
            labels_pie = ['Shared', f'{model1_name} Exclusive', f'{model2_name} Exclusive']
            colors_pie = ['#4CAF50', '#FF6B6B', '#4FC3F7']
            
            # Only show non-zero slices
            non_zero_sizes = []
            non_zero_labels = []
            non_zero_colors = []
            
            for size, label, color in zip(sizes, labels_pie, colors_pie):
                if size > 0:
                    non_zero_sizes.append(size)
                    non_zero_labels.append(label)
                    non_zero_colors.append(color)
            
            if len(non_zero_sizes) > 0:
                wedges, texts, autotexts = axes[2, pair_idx].pie(non_zero_sizes, labels=non_zero_labels, 
                                                                colors=non_zero_colors, autopct='%1.1f%%', 
                                                                startangle=90)
                # Make text more readable
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                axes[2, pair_idx].set_title(f'{model1_name} vs {model2_name}\nPairwise Distribution')
            else:
                axes[2, pair_idx].text(0.5, 0.5, 'No features found', 
                                    ha='center', va='center', transform=axes[2, pair_idx].transAxes,
                                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[2, pair_idx].set_title(f'{model1_name} vs {model2_name}\nPairwise Distribution')
            
            # Plot 4: Scatter plot of relative norms
            axes[3, pair_idx].scatter(rel_norms_model1, rel_norms_model2, alpha=0.6, s=20)
            axes[3, pair_idx].set_xlabel(f'{model1_name} Relative Norm')
            axes[3, pair_idx].set_ylabel(f'{model2_name} Relative Norm')
            axes[3, pair_idx].set_title(f'{model1_name} vs {model2_name}\nRelative Norm Scatter')
            axes[3, pair_idx].grid(True, alpha=0.3)
            
            # Add diagonal line for reference
            axes[3, pair_idx].plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1)
            
            pair_idx += 1
        
        plt.tight_layout()
        return fig
    
    def plot_triple_vs_pairwise_comparison(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compare triple analysis with pairwise analyses.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Triple analysis summary
        triple_shared = len(self.feature_classifications['shared'])
        triple_exclusive = [len(self.feature_classifications['exclusive'][model]) 
                           for model in self.model_names]
        
        counts = [triple_shared] + triple_exclusive
        labels = ['Shared'] + [f'{model}\nExclusive' for model in self.model_names]
        colors = ['#4CAF50', '#FF6B6B', '#4FC3F7', '#FFA726']
        
        bars = axes[0, 0].bar(range(len(counts)), counts, color=colors, alpha=0.7)
        axes[0, 0].set_xlabel('Feature Type')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Triple Analysis\n(All 3 Models Together)')
        axes[0, 0].set_xticks(range(len(counts)))
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                           f'{count}', ha='center', va='bottom')
        
        # Plot 2: Pairwise analysis summary
        pair_names = []
        pair_shared = []
        pair_exclusive_counts = []
        
        for pair_key, pair_data in self.pairwise_classifications.items():
            pair_classifications = pair_data['classifications']
            model1_name, model2_name = pair_data['model_names']
            
            pair_names.append(pair_key.replace('_vs_', ' vs '))
            pair_shared.append(len(pair_classifications['shared']))
            pair_exclusive_counts.append([
                len(pair_classifications['exclusive'][model1_name]),
                len(pair_classifications['exclusive'][model2_name])
            ])
        
        x = np.arange(len(pair_names))
        width = 0.25
        
        bars1 = axes[0, 1].bar(x - width, pair_shared, width, label='Shared', color='#4CAF50', alpha=0.7)
        bars2 = axes[0, 1].bar(x, [counts[0] for counts in pair_exclusive_counts], width, 
                              label='Model 1 Exclusive', color='#FF6B6B', alpha=0.7)
        bars3 = axes[0, 1].bar(x + width, [counts[1] for counts in pair_exclusive_counts], width,
                              label='Model 2 Exclusive', color='#4FC3F7', alpha=0.7)
        
        axes[0, 1].set_xlabel('Model Pairs')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Pairwise Analysis Summary')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(pair_names, rotation=45, ha='right')
        axes[0, 1].legend()
        
        # Plot 3: Comparison of shared features (triple vs pairwise)
        pairwise_shared_avg = np.mean(pair_shared)
        comparison_data = [triple_shared, pairwise_shared_avg]
        comparison_labels = ['Triple Analysis\n(All 3 Models)', 'Pairwise Average\n(2 Models)']
        
        bars = axes[1, 0].bar(range(2), comparison_data, color=['#4CAF50', '#2196F3'], alpha=0.7)
        axes[1, 0].set_xlabel('Analysis Type')
        axes[1, 0].set_ylabel('Number of Shared Features')
        axes[1, 0].set_title('Shared Features: Triple vs Pairwise')
        axes[1, 0].set_xticks(range(2))
        axes[1, 0].set_xticklabels(comparison_labels)
        
        # Add count labels on bars
        for bar, count in zip(bars, comparison_data):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(comparison_data),
                           f'{count:.1f}', ha='center', va='bottom')
        
        # Plot 4: Feature overlap analysis
        # Show how features classified as shared in triple analysis are distributed in pairwise
        triple_shared_features = set(self.feature_classifications['shared'])
        
        overlap_data = []
        overlap_labels = []
        
        for pair_key, pair_data in self.pairwise_classifications.items():
            pair_classifications = pair_data['classifications']
            pair_shared_features = set(pair_classifications['shared'])
            
            # Count how many triple-shared features are also pairwise-shared
            overlap = len(triple_shared_features.intersection(pair_shared_features))
            overlap_data.append(overlap)
            overlap_labels.append(pair_key.replace('_vs_', ' vs '))
        
        bars = axes[1, 1].bar(range(len(overlap_data)), overlap_data, color='#9C27B0', alpha=0.7)
        axes[1, 1].set_xlabel('Model Pairs')
        axes[1, 1].set_ylabel('Overlapping Shared Features')
        axes[1, 1].set_title('Triple-Shared Features\nFound in Pairwise Analysis')
        axes[1, 1].set_xticks(range(len(overlap_data)))
        axes[1, 1].set_xticklabels(overlap_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, overlap_data):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(overlap_data),
                           f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_cosine_similarity_shared_features(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot cosine similarity matrix and distribution for shared features.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        shared_indices = self.feature_classifications['shared']
        
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
        """
        Plot density (activation frequency) of shared and exclusive features.
        Replicates the style from the crosscoder diffing paper.
        
        Args:
            activations: Feature activations tensor [n_samples, latent_dim]
            figsize: Figure size
            activation_threshold: Threshold for considering a feature as "active"
            
        Returns:
            matplotlib Figure object
        """
        print(f"Activations shape: {activations.shape}")
        print(f"Activations stats: min={activations.min():.6f}, max={activations.max():.6f}, mean={activations.mean():.6f}")
        print(f"Non-zero activations: {(activations != 0).sum().item()} / {activations.numel()}")
        
        # Compute activation frequencies (density) using the threshold
        activation_freq = torch.mean((activations > activation_threshold).float(), dim=0)  # [latent_dim]
        activation_freq = activation_freq.detach().cpu().numpy()
        
        print(f"Feature density stats: min={activation_freq.min():.6f}, max={activation_freq.max():.6f}, mean={activation_freq.mean():.6f}")
        print(f"Features with non-zero density: {(activation_freq > 0).sum()} / {len(activation_freq)}")
        
        # Separate by feature type
        shared_indices = self.feature_classifications['shared']
        shared_densities = activation_freq[shared_indices] if shared_indices else []
        
        # Combine all exclusive features into one group (as in the paper)
        all_exclusive_indices = []
        for model_name in self.model_names:
            exclusive_indices = self.feature_classifications['exclusive'][model_name]
            all_exclusive_indices.extend(exclusive_indices)
        
        exclusive_densities = activation_freq[all_exclusive_indices] if all_exclusive_indices else []
        
        # Filter out zero densities for better visualization AND to avoid log scale issues
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
        
        # Determine if we should use log scale based on data range
        all_densities = shared_densities + exclusive_densities
        min_density = min(all_densities)
        max_density = max(all_densities)
        
        # Only use log scale if we have a reasonable range and all values are positive
        use_log_scale = (min_density > 0) and (max_density / min_density > 10)
        
        if use_log_scale:
            # Create log-spaced bins
            min_log = np.log10(max(min_density, 1e-8))  # Avoid log(0)
            max_log = np.log10(max_density)
            bins = np.logspace(min_log, max_log, 50)
        else:
            # Create linear-spaced bins
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
        
        # Style the plot to match the paper
        ax.set_xlabel('Feature density', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Feature density, shared and model-exclusive features', fontsize=14, fontweight='bold')
        
        # Only use log scale if appropriate
        if use_log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
            scale_note = " (log scale)"
        else:
            scale_note = " (linear scale)"
        
        ax.set_title(f'Feature density, shared and model-exclusive features{scale_note}', 
                    fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, which='both' if use_log_scale else 'major')
        
        # Add legend
        ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        
        # Set axis limits if needed
        if len(shared_densities) > 0 or len(exclusive_densities) > 0:
            if use_log_scale:
                ax.set_xlim(max(min_density * 0.5, 1e-8), max_density * 2)
            else:
                ax.set_xlim(min_density * 0.95, max_density * 1.05)
        
        # Add some statistics as text
        if len(shared_densities) > 0 and len(exclusive_densities) > 0:
            shared_median = np.median(shared_densities)
            exclusive_median = np.median(exclusive_densities)
            
            stats_text = f'Median density:\nShared: {shared_median:.6f}\nExclusive: {exclusive_median:.6f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_feature_density_log(self,
                             activations: torch.Tensor,
                             figsize: Tuple[int,int] = (10,6),
                             activation_threshold: float = 1e-6,
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

        shared_idx = self.feature_classifications['shared']
        excl_idx   = self.feature_classifications['exclusive']

        logf_shared = logf[shared_idx] if shared_idx else np.array([])
        logf_excl   = {
            m: logf[excl_idx[m]] if excl_idx[m] else np.array([])
            for m in self.model_names
        }

        all_vals = np

    # =============================================================
    #  1.  Utility – compute relative norms for an ordered pair
    # =============================================================
    def _pair_relative_norms(self,
                            idx_a: int,
                            idx_b: int,
                            eps: float = 1e-8) -> torch.Tensor:
        """
        Return a length‑latent_dim tensor r where
        
            r[i] = 0   -> feature i writes *only* to model A
            r[i] = 0.5 -> feature i writes equally to A and B
            r[i] = 1   -> feature i writes *only* to model B
            
        Parameters
        ----------
        idx_a, idx_b : int
            Indices of the two models you want to compare.
        eps : float
            Numerical guard to avoid division by zero.

        Notes
        -----
        This matches the definition used in §“Crosscoder model diffing recap”
        of the update post (peaks at 0, ½, 1):contentReference[oaicite:0]{index=0}.
        """
        pair_norms = self.decoder_norms[:, [idx_a, idx_b]]          # [latent_dim, 2]
        totals     = pair_norms.sum(dim=1, keepdim=True).clamp_min(eps)
        rel        = pair_norms[:, 1] / totals.squeeze(1)           # weight on model B
        return rel          # shape: [latent_dim]


    # =============================================================
    #  2.  Boolean mask for “exclusive” features in a pair
    # =============================================================
    def _pair_exclusive_masks(self,
                            idx_a: int,
                            idx_b: int,
                            thresh: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two boolean masks (len = latent_dim):
            mask_a → exclusive to model A   (relative ≤ thresh)
            mask_b → exclusive to model B   (relative ≥ 1‑thresh)
        Anything in between is treated as shared.
        """
        rel = self._pair_relative_norms(idx_a, idx_b)
        mask_b = rel >= (1.0 - thresh)
        mask_a = rel <= thresh
        return mask_a, mask_b
    
    # =============================================================
    #  3.  Paper‑style histogram for ONE pair
    # =============================================================
    def plot_relative_norm_hist(self,
                                idx_a: int,
                                idx_b: int,
                                bins: int = 60,
                                ax: plt.Axes | None = None) -> plt.Axes:
        """
        Draw the exact histogram style in the paper for a single ordered pair.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        rel = self._pair_relative_norms(idx_a, idx_b).detach().cpu().numpy()

        ax.hist(rel,
                bins=bins,
                color="#c48e67",               # matches paper’s tan color
                density=False)
        ax.set_yscale("log")
        ax.set_xlabel("Relative decoder norm", fontsize=12, weight="bold")
        ax.set_ylabel("Number of features",    fontsize=12, weight="bold")

        # Add nice x‑tick labels like in the figure
        ax.set_xticks([0, 0.5, 1])
        label_a = f"Fully in {self.model_names[idx_a]}"
        label_b = f"Fully in {self.model_names[idx_b]}"
        ax.set_xticklabels([label_a, "Shared features", label_b], fontsize=10)

        # Small vertical guides
        ax.axvline(0,   color="k", linewidth=0.5)
        ax.axvline(0.5, color="k", linewidth=0.5)
        ax.axvline(1,   color="k", linewidth=0.5)

        ax.set_title(f"{self.model_names[idx_a]} ↔ {self.model_names[idx_b]}", 
                    fontsize=13, weight="bold")
        return ax

    # =============================================================
    #  4.  Convenience: plot EVERY pair side‑by‑side
    # =============================================================
    def plot_all_pair_relative_norms(self,
                                    bins: int = 60,
                                    figsize: tuple[int, int] = (15, 4)):
        """
        Produce a row of histograms – one per ordered pair.
        """
        pairs = list(combinations(range(self.n_models), 2))
        fig, axes = plt.subplots(1, len(pairs), figsize=figsize,
                                squeeze=False)
        axes = axes[0]

        for ax, (i, j) in zip(axes, pairs):
            self.plot_relative_norm_hist(i, j, bins=bins, ax=ax)

        plt.tight_layout()
        return fig

def analyze_crosscoder(                       # ← signature unchanged for callers
        crosscoder,
        activations: torch.Tensor | None = None,
        model_names: list[str] = ('dice', 'pokemon', 'merged'),
        pair_exclusive_thresh: float = 0.05,  # ← NEW: threshold for exclusivity in a pair
        show_pair_hist: bool = True):         # ← NEW: toggle paper‑style histograms
    """
    Complete, one‑shot analysis pipeline for a CrossCoder.

    Parameters
    ----------
    crosscoder : CrossCoder
        The trained cross‑coder object (must expose W_dec / W_enc).
    activations : torch.Tensor | None
        Optional [n_samples, latent_dim] activations for density plots.
    model_names : list[str]
        Friendly names for the models in the same order as in W_dec.
    pair_exclusive_thresh : float
        r <= thresh  → exclusive to model A
        r >= 1‑thresh → exclusive to model B
    show_pair_hist : bool
        If True, will render the paper‑style relative‑norm histogram
        for every ordered pair of models.
    """
    # ──────────────────────────────────────────────────────────
    #  Initialise the analysis object
    # ──────────────────────────────────────────────────────────
    analysis = CrossCoderAnalysis(crosscoder, model_names)

    # ──────────────────────────────────────────────────────────
    #  Text summary ─ triple analysis
    # ──────────────────────────────────────────────────────────
    print("\n=== CROSSCODER ANALYSIS SUMMARY ===")
    print(f"Total latent dimensions : {analysis.latent_dim}")
    print(f"Number of models        : {analysis.n_models}")
    print(f"Number of activations   : {analysis.n_activations}")
    print(f"Model names             : {analysis.model_names}")

    print("\n=== TRIPLE ANALYSIS (all models together) ===")
    n_shared = len(analysis.feature_classifications['shared'])
    print(f"Shared features         : {n_shared}")
    for m in analysis.model_names:
        n_excl = len(analysis.feature_classifications['exclusive'][m])
        print(f"{m:>10} exclusive        : {n_excl}")

    # ──────────────────────────────────────────────────────────
    #  Text summary ─ pairwise (incl. new exclusivity masks)
    # ──────────────────────────────────────────────────────────
    print("\n=== PAIRWISE ANALYSIS ===")
    for (i, mA), (j, mB) in combinations(enumerate(analysis.model_names), 2):
        mask_a, mask_b = analysis._pair_exclusive_masks(i, j, thresh=pair_exclusive_thresh)
        n_a, n_b = mask_a.sum().item(), mask_b.sum().item()
        n_shared_pair = analysis.latent_dim - n_a - n_b
        print(f"{mA} ↔ {mB}:   "
              f"shared={n_shared_pair:>5}, "
              f"{mA}‑only={n_a:>5}, "
              f"{mB}‑only={n_b:>5}")

    # ──────────────────────────────────────────────────────────
    #  Plots
    # ──────────────────────────────────────────────────────────
    # 1) Full pairwise feature breakdown (old, still useful)
    print("\n… generating comprehensive pairwise feature dashboard …")
    analysis.plot_pairwise_feature_analysis(figsize=(20, 15))
    plt.show()

    # 2) Optional paper‑style relative‑norm histograms
    if show_pair_hist:
        print("… generating paper‑style relative‑norm histograms …")
        analysis.plot_all_pair_relative_norms(figsize=(15, 4))
        plt.show()

    # 3) Triple vs pairwise bar‑comparisons
    print("… generating triple‑vs‑pairwise comparison plot …")
    analysis.plot_triple_vs_pairwise_comparison(figsize=(15, 10))
    plt.show()

    # 4) Cosine similarity across shared features
    print("… generating cosine‑similarity plot for shared features …")
    fig_cos = analysis.plot_cosine_similarity_shared_features(figsize=(12, 8))
    if fig_cos is not None:
        plt.show()
    else:
        print("  (skipped ─ no shared features)")

    # 5) Feature‑density plots (if we have activations)
    if activations is not None:
        print("… generating feature‑density plot …")
        try:
            analysis.plot_feature_density(activations, figsize=(10, 6))
            plt.show()
        except Exception as err:
            print(f"  regular density plot failed: {err}\n  trying log version …")
            analysis.plot_feature_density_log(activations, figsize=(10, 6))
            plt.show()
    else:
        print("(skipping density plot ─ no activations provided)")

    print("\n=== ANALYSIS COMPLETE ===")
    return analysis


# Additional helper function for quick feature statistics
def print_feature_statistics(analysis):
    """
    Print detailed feature statistics from the analysis.
    
    Args:
        analysis: CrossCoderAnalysis instance
    """
    print("\n=== DETAILED FEATURE STATISTICS ===")
    
    # Triple analysis statistics
    print("\nTriple Analysis (All Models Together):")
    shared_features = analysis.feature_classifications['shared']
    print(f"  Shared features: {len(shared_features)}")
    if len(shared_features) > 0:
        print(f"  Shared feature indices (first 10): {shared_features[:10]}")
    
    total_exclusive = 0
    for model_name in analysis.model_names:
        exclusive_features = analysis.feature_classifications['exclusive'][model_name]
        total_exclusive += len(exclusive_features)
        print(f"  {model_name} exclusive features: {len(exclusive_features)}")
        if len(exclusive_features) > 0:
            print(f"    Indices (first 10): {exclusive_features[:10]}")
    
    print(f"  Total features accounted for: {len(shared_features) + total_exclusive}")
    
    # Pairwise analysis statistics
    print("\nPairwise Analysis Details:")
    for pair_key, pair_data in analysis.pairwise_classifications.items():
        pair_classifications = pair_data['classifications']
        model1_name, model2_name = pair_data['model_names']
        
        pair_shared = len(pair_classifications['shared'])
        model1_exclusive = len(pair_classifications['exclusive'][model1_name])
        model2_exclusive = len(pair_classifications['exclusive'][model2_name])
        
        print(f"\n  {pair_key}:")
        print(f"    Shared: {pair_shared}")
        print(f"    {model1_name} exclusive: {model1_exclusive}")
        print(f"    {model2_name} exclusive: {model2_exclusive}")
        print(f"    Total: {pair_shared + model1_exclusive + model2_exclusive}")


# Usage example with statistics
def analyze_crosscoder_with_stats(crosscoder, activations=None, model_names=['dice', 'pokemon', 'merged']):
    """
    Complete analysis pipeline with detailed statistics.
    
    Args:
        crosscoder: CrossCoder instance
        activations: Optional tensor of activations for density analysis
        model_names: List of model names
    """
    # Run main analysis
    analysis = analyze_crosscoder(crosscoder, activations, model_names)
    
    # Print detailed statistics
    print_feature_statistics(analysis)
    
    return analysis

# USAGE EXAMPLE
"""
# Given your setup:
interp = CrossCoder.CrossCoder(load_my_cross)

# Method 1: Quick analysis with all plots
model_names = ['dice', 'pokemon', 'merged']
analysis = analyze_crosscoder(interp, model_names=model_names)

# Method 2: Step-by-step analysis with diagnostics
analysis = CrossCoderAnalysis(interp, model_names)

# Print summary with diagnostic information
analysis.print_summary()

# If you get all shared features (like your current result), run diagnostics:
analysis.diagnose_feature_classification(num_examples=10)
analysis.try_different_thresholds([0.9, 0.8, 0.7, 0.6, 0.5])

# For balanced features (like your current case), try these specialized analyses:
most_unbalanced_features, variances = analysis.analyze_balanced_features(top_k=20)
fig_heatmap = analysis.plot_relative_norm_heatmap(figsize=(12, 8))
plt.show()

analysis.suggest_alternative_analysis()

# You can manually examine the most unbalanced features
print("\nMost unbalanced features (closest to being model-specific):")
for i, feature_idx in enumerate(most_unbalanced_features[:5]):
    print(f"Feature {feature_idx.item()}: variance = {variances[feature_idx]:.6f}")

# If you want to force categorization with very low threshold:
analysis.feature_classifications = analysis._classify_features(exclusive_threshold=0.4)
print("\nWith threshold 0.4:")
analysis.print_summary()

# You can also manually set a different threshold:
analysis.feature_classifications = analysis._classify_features(exclusive_threshold=0.8)
analysis.print_summary()

# Plot 1: Shared vs Exclusive Features Distribution
fig1 = analysis.plot_shared_exclusive_features(figsize=(15, 5))
plt.savefig('shared_exclusive_features.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Cosine Similarity of Shared Features
fig2 = analysis.plot_cosine_similarity_shared_features(figsize=(12, 8))
if fig2 is not None:
    plt.savefig('cosine_similarity_shared.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot 3: Feature Density Analysis (requires activations)
# You need to provide activations tensor from your models
# activations should be shape [n_samples, latent_dim]
# where n_samples is number of activation samples you want to analyze

# Example of getting activations (you'll need to adapt this to your data):
# with torch.no_grad():
#     # Get some sample activations from your encoder
#     sample_data = torch.randn(1000, 3, your_activation_dim)  # 1000 samples
#     activations = interp.W_enc.transpose(0, 2) @ sample_data.transpose(0, 1)
#     activations = activations.transpose(0, 1)  # [1000, latent_dim]
#     
#     fig3 = analysis.plot_feature_density(activations, figsize=(12, 6))
#     plt.savefig('feature_density.png', dpi=300, bbox_inches='tight')
#     plt.show()

# Access detailed results
shared_features = analysis.feature_classifications['shared']
dice_exclusive = analysis.feature_classifications['exclusive']['dice']
pokemon_exclusive = analysis.feature_classifications['exclusive']['pokemon']
merged_exclusive = analysis.feature_classifications['exclusive']['merged']

print(f"Found {len(shared_features)} shared features")
print(f"Found {len(dice_exclusive)} dice-exclusive features")
print(f"Found {len(pokemon_exclusive)} pokemon-exclusive features")
print(f"Found {len(merged_exclusive)} merged-exclusive features")

# Get relative decoder norms for further analysis
relative_norms = analysis.relative_norms  # [latent_dim, n_models]
decoder_norms = analysis.decoder_norms    # [latent_dim, n_models]

# You can also adjust the exclusive threshold if needed
# Default is 0.95, but you can create a new analysis with different threshold:
# analysis_strict = CrossCoderAnalysis(interp, model_names)
# analysis_strict.feature_classifications = analysis_strict._classify_features(exclusive_threshold=0.99)
"""

if __name__ == '__main__':
    from CrossCoderDataset import CrossCoderDataset
    import CrossCoder
    from get_dataloaders import get_dataloaders
    from utils import seed_run

    seed_run()
    LATENT_DIM=900
    LAMBDA_SPARSE=2
    BATCH_SIZE_CROSS = 64
    TRAINING_SIZE_CROSS   = 0.7
    VALIDATION_SIZE_CROSS = 0.1 # smaller validation because we just have to tune the latent_dim hyperparam
    TEST_SIZE_CROSS       = 0.2

    ACTIVATIONS_POKEMON_PATH = './activations_layer4/pokemon'
    ACTIVATIONS_DICE_PATH = './activations_layer4/dice'

    # Crosscoder dataset with Interpolation merging technique
    ACTIVATIONS_INTERPOLATED_PATH = './activations_layer4/interpolated'

    # Method 1: Quick analysis with all plots
    model_names = ['dice', 'pokemon', 'merged']
    ds = CrossCoderDataset(ACTIVATIONS_POKEMON_PATH, ACTIVATIONS_DICE_PATH, ACTIVATIONS_INTERPOLATED_PATH)
    n_acts = ds.get_n_activations()
    interp = CrossCoder.CrossCoder(LATENT_DIM, n_acts, LAMBDA_SPARSE)
    interp.load_state_dict(torch.load('./models/crosscoder/interpolated/model_weights.pth', weights_only=True))
    interp.eval()

    # split
    # dataset, batch_size, training_size, validation_size, test_size, shuffle_train=True
    train_loader, val_loader, test_loader = get_dataloaders(
        ds, BATCH_SIZE_CROSS, TRAINING_SIZE_CROSS, VALIDATION_SIZE_CROSS, TEST_SIZE_CROSS, shuffle_train=False
    )

    # TODO!!!!!! implementazione new features (differenza tra set)
    analyze_crosscoder(interp)
    # min, max = interp.W_enc.min(), interp.W_enc.max()
    # interp.W_enc.data = (interp.W_enc - min)/(max - min)


    # # analysis = analyze_crosscoder(interp, model_names=model_names)

    # # # Method 2: Step-by-step analysis with diagnostics
    # analysis = CrossCoderAnalysis(interp, model_names)


    # from torch.nn.functional import relu
    # print("\n" + "="*60)
    # print("PLOTTING E VISUALIZZAZIONE")
    # print("="*60)

    # min, max = interp.W_enc.min(), interp.W_enc.max()
    # interp.W_enc.data = (interp.W_enc - min)/(max - min)

    # # Usa tutti i dati disponibili
    # all_original_activations = []
    # for i, data in enumerate(test_loader):
    #     if i >= 10:  # Limita per performance
    #         break
    #     with torch.no_grad():
    #         acts = relu(interp.encode(data))
    #         all_original_activations.append(acts)
    
    # all_original_activations = torch.cat(all_original_activations, dim=0)

    # fig = analysis.plot_feature_density_log(all_original_activations,
    #                                 figsize=(12,6),
    #                                 activation_threshold=0.1)
    # plt.show()