import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, List
import seaborn as sns
from matplotlib.patches import Rectangle
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, List
import seaborn as sns
from matplotlib.patches import Rectangle

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
        
        # Classify features
        self.feature_classifications = self._classify_features()
        
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
        Classify features as shared or exclusive based on relative decoder norms.
        
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
    
    def plot_shared_exclusive_features(self, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot the distribution of shared vs exclusive features.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Bar chart of feature counts
        shared_count = len(self.feature_classifications['shared'])
        exclusive_counts = [len(self.feature_classifications['exclusive'][model]) 
                          for model in self.model_names]
        
        x_pos = np.arange(len(self.model_names) + 1)
        counts = [shared_count] + exclusive_counts
        labels = ['Shared'] + [f'{model}\nExclusive' for model in self.model_names]
        colors = ['gray'] + [plt.cm.Set3(i) for i in np.linspace(0, 1, self.n_models)]
        
        bars = axes[0].bar(x_pos, counts, color=colors, alpha=0.7)
        axes[0].set_xlabel('Feature Type')
        axes[0].set_ylabel('Number of Features')
        axes[0].set_title('Distribution of Shared vs Exclusive Features')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                        f'{count}', ha='center', va='bottom')
        
        # Plot 2: Relative decoder norms distribution
        for model_idx, model_name in enumerate(self.model_names):
            rel_norms_model = self.relative_norms[:, model_idx].detach().cpu().numpy()
            axes[1].hist(rel_norms_model, bins=50, alpha=0.6, label=model_name, density=True)
        
        axes[1].set_xlabel('Relative Decoder Norm')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution of Relative Decoder Norms')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Feature type pie chart
        sizes = [shared_count] + exclusive_counts
        labels_pie = ['Shared'] + [f'{model} Exclusive' for model in self.model_names]
        colors_pie = ['lightgray'] + [plt.cm.Set3(i) for i in np.linspace(0, 1, self.n_models)]
        
        wedges, texts, autotexts = axes[2].pie(sizes, labels=labels_pie, colors=colors_pie, 
                                              autopct='%1.1f%%', startangle=90)
        axes[2].set_title('Feature Type Distribution')
        
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
        
    def get_feature_summary(self) -> Dict:
        """
        Get a summary of the feature analysis.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_features': self.latent_dim,
            'shared_features': len(self.feature_classifications['shared']),
            'exclusive_features': {},
            'shared_percentage': len(self.feature_classifications['shared']) / self.latent_dim * 100
        }
        
        for model_name in self.model_names:
            count = len(self.feature_classifications['exclusive'][model_name])
            summary['exclusive_features'][model_name] = count
            summary[f'{model_name}_exclusive_percentage'] = count / self.latent_dim * 100
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of the analysis."""
        summary = self.get_feature_summary()
        
        print("=" * 50)
        print("CROSSCODER ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Features: {summary['total_features']}")
        print(f"Shared Features: {summary['shared_features']} ({summary['shared_percentage']:.1f}%)")
        print("\nExclusive Features:")
        for model_name in self.model_names:
            count = summary['exclusive_features'][model_name]
            percentage = summary[f'{model_name}_exclusive_percentage']
            print(f"  {model_name}: {count} ({percentage:.1f}%)")
        print("=" * 50)
        
        # Diagnostic information
        print("\nDIAGNOSTIC INFORMATION:")
        print(f"Decoder norms shape: {self.decoder_norms.shape}")
        print(f"Relative norms shape: {self.relative_norms.shape}")
        
        # Check relative norm statistics
        max_rel_norms = torch.max(self.relative_norms, dim=1)[0]
        print(f"Max relative norms - Min: {max_rel_norms.min():.4f}, Max: {max_rel_norms.max():.4f}, Mean: {max_rel_norms.mean():.4f}")
        
        # Check how many features are close to exclusive threshold
        exclusive_threshold = 0.90
        near_exclusive = torch.sum(max_rel_norms > 0.8).item()
        very_near_exclusive = torch.sum(max_rel_norms > 0.9).item()
        at_threshold = torch.sum(max_rel_norms > exclusive_threshold).item()
        
        print(f"Features with max relative norm > 0.8: {near_exclusive}")
        print(f"Features with max relative norm > 0.9: {very_near_exclusive}")
        print(f"Features with max relative norm > {exclusive_threshold}: {at_threshold}")
        
        # Check decoder norm statistics
        total_decoder_norms = torch.sum(self.decoder_norms, dim=1)
        print(f"Total decoder norms - Min: {total_decoder_norms.min():.4f}, Max: {total_decoder_norms.max():.4f}")
        
        # Check if any decoder norms are zero or very small
        zero_norms = torch.sum(self.decoder_norms < 1e-6, dim=1)
        features_with_zero_norms = torch.sum(zero_norms > 0).item()
        print(f"Features with near-zero decoder norms: {features_with_zero_norms}")
        
    def diagnose_feature_classification(self, num_examples: int = 10):
        """
        Diagnose why features might not be classified as exclusive.
        
        Args:
            num_examples: Number of example features to show
        """
        print("\n" + "=" * 60)
        print("DETAILED FEATURE CLASSIFICATION DIAGNOSIS")
        print("=" * 60)
        
        # Show examples of features with highest relative norms
        max_rel_norms, max_indices = torch.max(self.relative_norms, dim=1)
        sorted_indices = torch.argsort(max_rel_norms, descending=True)
        
        print(f"\nTop {num_examples} features by maximum relative norm:")
        for i in range(min(num_examples, len(sorted_indices))):
            feature_idx = sorted_indices[i].item()
            max_norm = max_rel_norms[feature_idx].item()
            max_model = max_indices[feature_idx].item()
            
            print(f"Feature {feature_idx}:")
            print(f"  Max relative norm: {max_norm:.4f} (model: {self.model_names[max_model]})")
            print(f"  Relative norms per model:")
            for j, model_name in enumerate(self.model_names):
                rel_norm = self.relative_norms[feature_idx, j].item()
                decoder_norm = self.decoder_norms[feature_idx, j].item()
                print(f"    {model_name}: {rel_norm:.4f} (decoder norm: {decoder_norm:.4f})")
            print(f"  Classification: {'Exclusive' if max_norm > 0.95 else 'Shared'}")
            print()
        
        # Show distribution of relative norms
        print("Relative norm distribution:")
        for i, model_name in enumerate(self.model_names):
            rel_norms = self.relative_norms[:, i]
            print(f"{model_name}: min={rel_norms.min():.4f}, max={rel_norms.max():.4f}, mean={rel_norms.mean():.4f}")
        
    def try_different_thresholds(self, thresholds: List[float] = [0.9, 0.8, 0.7, 0.6, 0.5]):
        """
        Try different exclusive thresholds to see classification results.
        
        Args:
            thresholds: List of thresholds to try
        """
        print("\n" + "=" * 60)
        print("TESTING DIFFERENT EXCLUSIVE THRESHOLDS")
        print("=" * 60)
        
        for threshold in thresholds:
            classifications = self._classify_features(exclusive_threshold=threshold)
            shared_count = len(classifications['shared'])
            
            print(f"\nThreshold {threshold}:")
            print(f"  Shared: {shared_count} ({shared_count/self.latent_dim*100:.1f}%)")
            for model_name in self.model_names:
                count = len(classifications['exclusive'][model_name])
                print(f"  {model_name} exclusive: {count} ({count/self.latent_dim*100:.1f}%)")
        
    def analyze_balanced_features(self, top_k: int = 20):
        """
        Analyze features when they're balanced across models (like your current situation).
        
        Args:
            top_k: Number of top features to analyze
        """
        print("\n" + "=" * 60)
        print("BALANCED FEATURES ANALYSIS")
        print("=" * 60)
        
        # Calculate variance in relative norms for each feature
        relative_norm_variance = torch.var(self.relative_norms, dim=1)
        
        # Find features with highest variance (most "unbalanced")
        most_unbalanced_indices = torch.argsort(relative_norm_variance, descending=True)[:top_k]
        
        print(f"\nTop {top_k} most unbalanced features:")
        print("(These are the closest to being model-specific)")
        print("-" * 60)
        
        for i, feature_idx in enumerate(most_unbalanced_indices):
            variance = relative_norm_variance[feature_idx].item()
            print(f"\nFeature {feature_idx.item()} (variance: {variance:.6f}):")
            
            # Show relative norms for each model
            for j, model_name in enumerate(self.model_names):
                rel_norm = self.relative_norms[feature_idx, j].item()
                decoder_norm = self.decoder_norms[feature_idx, j].item()
                print(f"  {model_name}: {rel_norm:.4f} (decoder: {decoder_norm:.4f})")
            
            # Find the "dominant" model (highest relative norm)
            dominant_model_idx = torch.argmax(self.relative_norms[feature_idx]).item()
            dominant_model = self.model_names[dominant_model_idx]
            dominant_norm = self.relative_norms[feature_idx, dominant_model_idx].item()
            print(f"  Dominant model: {dominant_model} ({dominant_norm:.4f})")
        
        # Statistics about the balance
        print(f"\nOverall balance statistics:")
        print(f"Mean relative norm variance: {relative_norm_variance.mean():.6f}")
        print(f"Max relative norm variance: {relative_norm_variance.max():.6f}")
        print(f"Min relative norm variance: {relative_norm_variance.min():.6f}")
        
        # Perfect balance would be 1/3 for each model
        perfect_balance = 1.0 / self.n_models
        print(f"Perfect balance point: {perfect_balance:.4f}")
        print(f"Actual mean relative norm: {self.relative_norms.mean():.4f}")
        
        return most_unbalanced_indices, relative_norm_variance
    
    def plot_relative_norm_heatmap(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot a heatmap of relative norms to visualize feature balance.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Heatmap of relative norms (sample of features)
        n_features_to_show = min(50, self.latent_dim)
        sample_indices = torch.randperm(self.latent_dim)[:n_features_to_show]
        sample_rel_norms = self.relative_norms[sample_indices].detach().cpu().numpy()
        
        im1 = axes[0, 0].imshow(sample_rel_norms.T, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'Relative Norms Heatmap\n(Random sample of {n_features_to_show} features)')
        axes[0, 0].set_xlabel('Feature Index')
        axes[0, 0].set_ylabel('Model')
        axes[0, 0].set_yticks(range(self.n_models))
        axes[0, 0].set_yticklabels(self.model_names)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Histogram of relative norms for each model
        for i, model_name in enumerate(self.model_names):
            rel_norms = self.relative_norms[:, i].detach().cpu().numpy()
            axes[0, 1].hist(rel_norms, bins=50, alpha=0.7, label=model_name, density=True)
        
        axes[0, 1].set_xlabel('Relative Norm')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Distribution of Relative Norms by Model')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Variance in relative norms
        rel_norm_variance = torch.var(self.relative_norms, dim=1).detach().cpu().numpy()
        axes[1, 0].hist(rel_norm_variance, bins=50, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Relative Norm Variance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Relative Norm Variance\n(Higher = More Unbalanced)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot: max relative norm vs variance
        max_rel_norms = torch.max(self.relative_norms, dim=1)[0].detach().cpu().numpy()
        axes[1, 1].scatter(max_rel_norms, rel_norm_variance, alpha=0.6)
        axes[1, 1].set_xlabel('Max Relative Norm')
        axes[1, 1].set_ylabel('Relative Norm Variance')
        axes[1, 1].set_title('Max Norm vs Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def suggest_alternative_analysis(self):
        """
        Suggest alternative analysis approaches when features are balanced.
        """
        print("\n" + "=" * 60)
        print("ALTERNATIVE ANALYSIS SUGGESTIONS")
        print("=" * 60)
        
        print("Since your crosscoder learned balanced features, consider these approaches:")
        print("\n1. COSINE SIMILARITY ANALYSIS:")
        print("   - Even if features have similar magnitudes, their directions might differ")
        print("   - Use plot_cosine_similarity_shared_features() to see if shared features")
        print("     have different orientations across models")
        
        print("\n2. GRADIENT-BASED ANALYSIS:")
        print("   - Look at which features activate most strongly for each model")
        print("   - Provide actual activations to use plot_feature_density()")
        
        print("\n3. FEATURE VARIANCE ANALYSIS:")
        print("   - Focus on features with highest relative norm variance")
        print("   - Use analyze_balanced_features() to find the most 'unbalanced' features")
        
        print("\n4. RETRAIN WITH STRONGER EXCLUSIVITY:")
        print("   - Your crosscoder might need stronger L1 penalty")
        print("   - Consider increasing the sparsity coefficient")
        print("   - Try the auxiliary loss approach from the paper")
        
        print("\n5. MANUAL THRESHOLDING:")
        print("   - Use a much lower threshold (e.g., 0.4) to artificially create categories")
        print("   - This will show you which features are 'most associated' with each model")
        
        # Calculate some useful statistics
        rel_norm_variance = torch.var(self.relative_norms, dim=1)
        most_unbalanced_count = torch.sum(rel_norm_variance > rel_norm_variance.quantile(0.9)).item()
        
        print(f"\n6. STATISTICAL INSIGHTS:")
        print(f"   - {most_unbalanced_count} features are in the top 10% of variance")
        print(f"   - These might be your 'most model-specific' features")
        print(f"   - Consider focusing analysis on these features")
        
        return rel_norm_variance


# Usage example function
def analyze_crosscoder(crosscoder, activations=None, model_names=['dice', 'pokemon', 'merged']):
    """
    Complete analysis pipeline for crosscoder.
    
    Args:
        crosscoder: CrossCoder instance
        activations: Optional tensor of activations for density analysis
        model_names: List of model names
    """
    # Initialize analysis
    analysis = CrossCoderAnalysis(crosscoder, model_names)
    
    # Print summary
    analysis.print_summary()
    
    # Plot shared/exclusive features
    print("\nGenerating shared/exclusive features plot...")
    fig1 = analysis.plot_shared_exclusive_features()
    plt.show()
    
    # Plot cosine similarity
    print("\nGenerating cosine similarity plot...")
    fig2 = analysis.plot_cosine_similarity_shared_features()
    if fig2 is not None:
        plt.show()
    
    # Plot feature density (if activations provided)
    if activations is not None:
        print("\nGenerating feature density plot...")
        fig3 = analysis.plot_feature_density(activations)
        plt.show()
    else:
        print("\nSkipping feature density plot (no activations provided)")
    
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

    ds = CrossCoderDataset('./activations_layer4')
    n_acts = ds.get_n_activations()
    interp = CrossCoder.CrossCoder(LATENT_DIM, n_acts, LAMBDA_SPARSE)
    interp.load_state_dict(torch.load('./models/crosscoder/interpolated/model_weights.pth', weights_only=True))
    interp.eval()

    # Method 1: Quick analysis with all plots
    model_names = ['dice', 'pokemon', 'merged']
    ds = CrossCoderDataset('./activations_layer4')
    n_acts = ds.get_n_activations()

    # split
    # dataset, batch_size, training_size, validation_size, test_size, shuffle_train=True
    train_loader, val_loader, test_loader = get_dataloaders(
        ds, BATCH_SIZE_CROSS, TRAINING_SIZE_CROSS, VALIDATION_SIZE_CROSS, TEST_SIZE_CROSS, shuffle_train=False
    )

    analysis = analyze_crosscoder(interp, model_names=model_names)

    # Method 2: Step-by-step analysis with diagnostics
    analysis = CrossCoderAnalysis(interp, model_names)

    # Print summary with diagnostic information
    analysis.print_summary()

    # If you get all shared features (like your current result), run diagnostics:
    # analysis.diagnose_feature_classification(num_examples=10)
    # analysis.try_different_thresholds([0.9, 0.8, 0.7, 0.6, 0.5])

    # You can also manually set a different threshold:
    analysis.feature_classifications = analysis._classify_features(exclusive_threshold=0.9)
    analysis.print_summary()

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
    # analysis.feature_classifications = analysis._classify_features(exclusive_threshold=0.4)
    # print("\nWith threshold 0.4:")
    # analysis.print_summary()

    # # You can also manually set a different threshold:
    # analysis.feature_classifications = analysis._classify_features(exclusive_threshold=0.8)
    # analysis.print_summary()



    # # Plot 1: Shared vs Exclusive Features Distribution
    # fig1 = analysis.plot_shared_exclusive_features(figsize=(15, 5))
    # plt.savefig('shared_exclusive_features.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # # Plot 2: Cosine Similarity of Shared Features
    # fig2 = analysis.plot_cosine_similarity_shared_features(figsize=(12, 8))
    # if fig2 is not None:
    #     plt.savefig('cosine_similarity_shared.png', dpi=300, bbox_inches='tight')
    #     plt.show()

    # Plot 3: Feature Density Analysis (requires activations)
    # You need to provide activations tensor from your models
    # activations should be shape [n_samples, latent_dim]
    # where n_samples is number of activation samples you want to analyze

    # Example of getting activations (you'll need to adapt this to your data):
    with torch.no_grad():
        import einops
        from torch.nn.functional import relu
        
        all_activations = []
        
        # Collect activations from multiple batches for better statistics
        for i, data in enumerate(train_loader):
            print(f"Batch {i}: {data.size()}")
            print(f"W_enc size: {interp.W_enc.size()}")
            
            # data should be shape [batch_size, n_models, n_activations]
            # interp.W_enc is shape [n_models, n_activations, latent_dim]
            
            activations = relu(interp.encode(data))
            
            all_activations.append(activations)
            
        
        # Concatenate all activations
        all_activations = torch.cat(all_activations, dim=0)
        print(all_activations.min(), all_activations.max())
        print(f"Final activations shape: {all_activations.shape}")
        
        # Plot feature density
        fig3 = analysis.plot_feature_density(all_activations, figsize=(12, 6))
        plt.savefig('feature_density.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Access detailed results
    shared_features = analysis.feature_classifications['shared']
    dice_exclusive = analysis.feature_classifications['exclusive']['dice']
    pokemon_exclusive = analysis.feature_classifications['exclusive']['pokemon']
    merged_exclusive = analysis.feature_classifications['exclusive']['merged']
    new_features = (set(merged_exclusive).difference(set(pokemon_exclusive))).difference(set(dice_exclusive))

    print(f"Found {len(shared_features)} shared features")
    print(f"Found {len(dice_exclusive)} dice-exclusive features")
    print(f"Found {len(pokemon_exclusive)} pokemon-exclusive features")
    print(f"Found {len(merged_exclusive)} merged-exclusive features")
    print(f"Found {len(new_features)} new features")

    # Get relative decoder norms for further analysis
    relative_norms = analysis.relative_norms  # [latent_dim, n_models]
    decoder_norms = analysis.decoder_norms    # [latent_dim, n_models]

    # You can also adjust the exclusive threshold if needed
    # Default is 0.95, but you can create a new analysis with different threshold:
    # analysis_strict = CrossCoderAnalysis(interp, model_names)
    # analysis_strict.feature_classifications = analysis_strict._classify_features(exclusive_threshold=0.99)
    # USAGE EXAMPLE