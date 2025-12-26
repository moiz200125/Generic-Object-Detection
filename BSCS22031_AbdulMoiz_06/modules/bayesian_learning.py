# modules/bayesian_learning.py
"""
Bayesian Parameter Learning for ED, SS cues
Finds parameters that best separate Object vs Background windows
"""
import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
try:
    from .parameter_learning import ParameterLearner
    from .ms_cue import MultiScaleSaliencyCue
    from .ss_cue import SuperpixelStraddlingCue
    from .ed_cue import EdgeDensityCue
except ImportError:
    from parameter_learning import ParameterLearner
    from ms_cue import MultiScaleSaliencyCue
    from ss_cue import SuperpixelStraddlingCue
    from ed_cue import EdgeDensityCue

class BayesianParameterLearner(ParameterLearner):
    """
    Bayesian learning for ED and SS parameters
    Uses histogram-based likelihood estimation and Bayesian selection
    """
    
    def __init__(self, voc_dataset_path, output_dir='bayesian_learned_params'):
        super().__init__(voc_dataset_path, output_dir)
        
        # For storing likelihood distributions
        self.likelihoods = {}
        
    def generate_training_samples(self, n_images=10, n_windows_per_image=1000):
        """
        Generate positive and negative samples for training
        
        Returns:
            Dictionary with image data and window samples
        """
        print("\n" + "="*60)
        print("GENERATING TRAINING SAMPLES")
        print("="*60)
        
        train_images = self.image_files[:min(n_images, len(self.image_files))]
        print(f"Using {len(train_images)} images")
        print(f"Generating {n_windows_per_image} windows per image")
        
        all_samples = {
            'positive': [],  # List of (image_idx, window, iou)
            'negative': [],  # List of (image_idx, window, iou)
            'image_data': [] # List of (image_path, image_array, gt_boxes)
        }
        
        for img_idx, img_file in enumerate(tqdm(train_images, desc="Processing images")):
            # Load image
            img_path = os.path.join(self.images_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
            
            # Get ground truth
            img_name = os.path.splitext(img_file)[0]
            gt_boxes = self.load_ground_truth(img_name)
            
            if not gt_boxes:
                continue
            
            # Store image data
            all_samples['image_data'].append({
                'path': img_path,
                'image': image,
                'gt_boxes': gt_boxes,
                'name': img_name
            })
            
            h, w = image.shape[:2]
            
            # Generate random windows
            for _ in range(n_windows_per_image):
                # Random window size (between 30px and min(200, w//2))
                win_w = np.random.randint(30, min(200, w//2))
                win_h = np.random.randint(30, min(200, h//2))
                x = np.random.randint(0, w - win_w)
                y = np.random.randint(0, h - win_h)
                
                window = (x, y, x + win_w, y + win_h)
                
                # Calculate IoU with all GT boxes
                best_iou = 0.0
                for gt_box in gt_boxes:
                    iou = self.calculate_iou(window, gt_box)
                    best_iou = max(best_iou, iou)
                
                # Classify as positive or negative
                if best_iou > 0.5:
                    all_samples['positive'].append((img_idx, window, best_iou))
                else:
                    all_samples['negative'].append((img_idx, window, best_iou))
        
        print(f"\nGenerated {len(all_samples['positive'])} positive samples (IoU > 0.5)")
        print(f"Generated {len(all_samples['negative'])} negative samples (IoU ≤ 0.5)")
        print(f"Total samples: {len(all_samples['positive']) + len(all_samples['negative'])}")
        
        return all_samples
    
    def calculate_histogram_likelihoods(self, scores_positive, scores_negative, n_bins=20):
        """
        Calculate likelihood distributions P(Score | Obj) and P(Score | Bg)
        
        Args:
            scores_positive: List of scores for positive samples
            scores_negative: List of scores for negative samples
            n_bins: Number of bins for histogram
            
        Returns:
            Dictionary with likelihood distributions
        """
        # Combine all scores to determine bin range
        all_scores = np.concatenate([scores_positive, scores_negative])
        score_min = np.min(all_scores)
        score_max = np.max(all_scores)
        
        # Add small epsilon to avoid edge issues
        epsilon = 1e-10
        score_min -= epsilon
        score_max += epsilon
        
        # Create bins
        bins = np.linspace(score_min, score_max, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate histograms (normalized to get probabilities)
        hist_pos, _ = np.histogram(scores_positive, bins=bins, density=True)
        hist_neg, _ = np.histogram(scores_negative, bins=bins, density=True)
        
        # Add small value to avoid zero probabilities
        hist_pos = hist_pos + 1e-10
        hist_neg = hist_neg + 1e-10
        
        # Normalize to sum to 1
        hist_pos = hist_pos / np.sum(hist_pos)
        hist_neg = hist_neg / np.sum(hist_neg)
        
        return {
            'bins': bins,
            'bin_centers': bin_centers,
            'p_score_given_obj': hist_pos,      # P(Score | Object)
            'p_score_given_bg': hist_neg,       # P(Score | Background)
            'n_bins': n_bins
        }
    
    def bayesian_selection_criterion(self, likelihoods):
        """
        Calculate Bayesian selection criterion (Eq 5 in paper)
        
        Essentially measures how well the parameter separates classes
        Higher value = better separation
        """
        p_obj = likelihoods['p_score_given_obj']
        p_bg = likelihoods['p_score_given_bg']
        
        # Avoid division by zero
        epsilon = 1e-10
        p_obj = p_obj + epsilon
        p_bg = p_bg + epsilon
        
        # Calculate KL divergence (measure of separation)
        # Higher KL divergence = better separation
        kl_divergence = np.sum(p_obj * np.log(p_obj / p_bg))
        
        # Alternative: Calculate probability of correct classification
        # Assuming equal prior P(Object) = P(Background) = 0.5
        p_correct = 0.5 * np.sum(np.maximum(p_obj - p_bg, 0)) + 0.5 * np.sum(np.maximum(p_bg - p_obj, 0))
        
        return {
            'kl_divergence': float(kl_divergence),
            'p_correct': float(p_correct),
            'mean_diff': float(np.mean(p_obj) - np.mean(p_bg))
        }
    
    def learn_ed_parameters_bayesian(self, n_images=10, n_windows_per_image=500):
        """
        Learn ED parameters using Bayesian approach
        
        Steps:
        1. Generate positive/negative samples
        2. Try different border_ratio values
        3. For each border_ratio, calculate ED scores
        4. Build likelihood distributions P(Score|Obj) and P(Score|Bg)
        5. Select border_ratio that maximizes separation
        """
        print("\n" + "="*60)
        print("BAYESIAN LEARNING: EDGE DENSITY PARAMETERS")
        print("="*60)
        
        # Generate training samples
        samples = self.generate_training_samples(n_images, n_windows_per_image)
        
        if len(samples['positive']) < 50 or len(samples['negative']) < 50:
            print("Warning: Not enough samples for learning")
            return {'border_ratio': 0.1, 'canny_low': 50, 'canny_high': 150}
        
        # Parameter range to search (θ_ED values)
        border_ratios = np.arange(0.05, 0.21, 0.01)  # 0.05 to 0.20 in steps of 0.01
        print(f"\nTesting {len(border_ratios)} border_ratio values: {border_ratios[:3]}...{border_ratios[-3:]}")
        
        # Store results for each parameter
        param_results = []
        
        for border_ratio in tqdm(border_ratios, desc="Testing border_ratios"):
            # Collect scores for this parameter
            positive_scores = []
            negative_scores = []
            
            # Process in batches to save memory
            batch_size = 1000
            n_batches = max(1, len(samples['positive']) // batch_size)
            
            for batch_idx in range(n_batches):
                # Get batch of positive samples
                pos_start = batch_idx * batch_size
                pos_end = min((batch_idx + 1) * batch_size, len(samples['positive']))
                pos_batch = samples['positive'][pos_start:pos_end]
                
                # Get batch of negative samples
                neg_start = batch_idx * batch_size
                neg_end = min((batch_idx + 1) * batch_size, len(samples['negative']))
                neg_batch = samples['negative'][neg_start:neg_end]
                
                # Process positive batch
                for img_idx, window, _ in pos_batch:
                    img_data = samples['image_data'][img_idx]
                    image = img_data['image']
                    
                    # Initialize ED cue with current border_ratio
                    ed = EdgeDensityCue(image, border_ratio=border_ratio)
                    
                    # Calculate ED score
                    score = ed.get_score(window)
                    positive_scores.append(score)
                
                # Process negative batch
                for img_idx, window, _ in neg_batch:
                    img_data = samples['image_data'][img_idx]
                    image = img_data['image']
                    
                    ed = EdgeDensityCue(image, border_ratio=border_ratio)
                    score = ed.get_score(window)
                    negative_scores.append(score)
            
            # Convert to numpy arrays
            positive_scores = np.array(positive_scores)
            negative_scores = np.array(negative_scores)
            
            # Skip if not enough scores
            if len(positive_scores) < 10 or len(negative_scores) < 10:
                continue
            
            # Calculate likelihood distributions
            likelihoods = self.calculate_histogram_likelihoods(
                positive_scores, negative_scores, n_bins=20
            )
            
            # Calculate Bayesian selection criterion
            bayesian_score = self.bayesian_selection_criterion(likelihoods)
            
            # Store results
            param_results.append({
                'border_ratio': float(border_ratio),
                'positive_mean': float(np.mean(positive_scores)),
                'negative_mean': float(np.mean(negative_scores)),
                'positive_std': float(np.std(positive_scores)),
                'negative_std': float(np.std(negative_scores)),
                'kl_divergence': bayesian_score['kl_divergence'],
                'p_correct': bayesian_score['p_correct'],
                'mean_diff': bayesian_score['mean_diff'],
                'likelihoods': likelihoods
            })
        
        if not param_results:
            print("Warning: No valid parameter results")
            return {'border_ratio': 0.1, 'canny_low': 50, 'canny_high': 150}
        
        # Find best parameter based on KL divergence
        best_idx = np.argmax([r['kl_divergence'] for r in param_results])
        best_result = param_results[best_idx]
        
        print(f"\nBest border_ratio: {best_result['border_ratio']:.3f}")
        print(f"KL Divergence: {best_result['kl_divergence']:.4f}")
        print(f"P(correct): {best_result['p_correct']:.4f}")
        print(f"Mean positive score: {best_result['positive_mean']:.4f}")
        print(f"Mean negative score: {best_result['negative_mean']:.4f}")
        print(f"Separation (mean diff): {best_result['mean_diff']:.4f}")
        
        # Plot results
        self._plot_ed_bayesian_results(param_results, best_result)
        
        # Plot likelihood distributions for best parameter
        self._plot_likelihood_distributions(
            best_result['likelihoods'],
            f"ED Likelihoods (border_ratio={best_result['border_ratio']:.3f})",
            os.path.join(self.output_dir, 'ed_likelihoods_best.png')
        )
        
        # Store results
        self.results['ed_params'] = {
            'border_ratio': best_result['border_ratio'],
            'canny_low': 50,
            'canny_high': 150,
            'use_perimeter': True,
            'bayesian_scores': {
                'kl_divergence': best_result['kl_divergence'],
                'p_correct': best_result['p_correct'],
                'mean_diff': best_result['mean_diff']
            }
        }
        
        print("\n" + "-"*60)
        print("ED BAYESIAN LEARNING COMPLETE")
        print("-"*60)
        
        return self.results['ed_params']
    
    def learn_ss_parameters_bayesian(self, n_images=10, n_windows_per_image=500):
        """
        Learn SS parameters using Bayesian approach
        
        Parameter to learn: n_segments (number of superpixels)
        """
        print("\n" + "="*60)
        print("BAYESIAN LEARNING: SUPERPIXELS STRADDLING PARAMETERS")
        print("="*60)
        
        # Generate training samples
        samples = self.generate_training_samples(n_images, n_windows_per_image)
        
        if len(samples['positive']) < 50 or len(samples['negative']) < 50:
            print("Warning: Not enough samples for learning")
            return {'n_segments': 100, 'algorithm': 'slic'}
        
        # Parameter range to search
        n_segments_options = [50, 75, 100, 125, 150, 175, 200]
        print(f"\nTesting {len(n_segments_options)} n_segments values: {n_segments_options}")
        
        param_results = []
        
        for n_segments in tqdm(n_segments_options, desc="Testing n_segments"):
            positive_scores = []
            negative_scores = []
            
            # We need to process each image separately for SS cue
            for img_data in samples['image_data']:
                image = img_data['image']
                
                # Initialize SS cue once per image (expensive operation)
                ss = SuperpixelStraddlingCue(image, n_segments=n_segments)
                ss.compute_superpixels()
                
                # Score positive samples for this image
                for img_idx, window, _ in samples['positive']:
                    if img_idx == samples['image_data'].index(img_data):
                        score = ss.get_score(window)
                        positive_scores.append(score)
                
                # Score negative samples for this image
                for img_idx, window, _ in samples['negative']:
                    if img_idx == samples['image_data'].index(img_data):
                        score = ss.get_score(window)
                        negative_scores.append(score)
            
            # Convert to arrays
            positive_scores = np.array(positive_scores)
            negative_scores = np.array(negative_scores)
            
            if len(positive_scores) < 10 or len(negative_scores) < 10:
                continue
            
            # Calculate likelihoods
            likelihoods = self.calculate_histogram_likelihoods(
                positive_scores, negative_scores, n_bins=20
            )
            
            # Calculate Bayesian score
            bayesian_score = self.bayesian_selection_criterion(likelihoods)
            
            param_results.append({
                'n_segments': int(n_segments),
                'positive_mean': float(np.mean(positive_scores)),
                'negative_mean': float(np.mean(negative_scores)),
                'positive_std': float(np.std(positive_scores)),
                'negative_std': float(np.std(negative_scores)),
                'kl_divergence': bayesian_score['kl_divergence'],
                'p_correct': bayesian_score['p_correct'],
                'mean_diff': bayesian_score['mean_diff'],
                'likelihoods': likelihoods
            })
        
        if not param_results:
            print("Warning: No valid parameter results")
            return {'n_segments': 100, 'algorithm': 'slic'}
        
        # Find best parameter
        best_idx = np.argmax([r['kl_divergence'] for r in param_results])
        best_result = param_results[best_idx]
        
        print(f"\nBest n_segments: {best_result['n_segments']}")
        print(f"KL Divergence: {best_result['kl_divergence']:.4f}")
        print(f"P(correct): {best_result['p_correct']:.4f}")
        print(f"Mean positive score: {best_result['positive_mean']:.4f}")
        print(f"Mean negative score: {best_result['negative_mean']:.4f}")
        print(f"Separation (mean diff): {best_result['mean_diff']:.4f}")
        
        # Plot results
        self._plot_ss_bayesian_results(param_results, best_result)
        
        # Plot likelihood distributions
        self._plot_likelihood_distributions(
            best_result['likelihoods'],
            f"SS Likelihoods (n_segments={best_result['n_segments']})",
            os.path.join(self.output_dir, 'ss_likelihoods_best.png')
        )
        
        # Store results
        self.results['ss_params'] = {
            'n_segments': best_result['n_segments'],
            'algorithm': 'slic',
            'compactness': 10,
            'bayesian_scores': {
                'kl_divergence': best_result['kl_divergence'],
                'p_correct': best_result['p_correct'],
                'mean_diff': best_result['mean_diff']
            }
        }
        
        print("\n" + "-"*60)
        print("SS BAYESIAN LEARNING COMPLETE")
        print("-"*60)
        
        return self.results['ss_params']
    
    def _plot_ed_bayesian_results(self, param_results, best_result):
        """Plot ED Bayesian learning results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        border_ratios = [r['border_ratio'] for r in param_results]
        kl_divs = [r['kl_divergence'] for r in param_results]
        p_corrects = [r['p_correct'] for r in param_results]
        mean_diffs = [r['mean_diff'] for r in param_results]
        pos_means = [r['positive_mean'] for r in param_results]
        neg_means = [r['negative_mean'] for r in param_results]
        
        # 1. KL Divergence vs border_ratio
        axes[0, 0].plot(border_ratios, kl_divs, 'b-', linewidth=2, marker='o')
        axes[0, 0].axvline(x=best_result['border_ratio'], color='r', linestyle='--',
                          label=f'Best: {best_result["border_ratio"]:.3f}')
        axes[0, 0].set_xlabel('Border Ratio (θ_ED)')
        axes[0, 0].set_ylabel('KL Divergence')
        axes[0, 0].set_title('KL Divergence vs Border Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. P(correct) vs border_ratio
        axes[0, 1].plot(border_ratios, p_corrects, 'g-', linewidth=2, marker='s')
        axes[0, 1].axvline(x=best_result['border_ratio'], color='r', linestyle='--')
        axes[0, 1].set_xlabel('Border Ratio (θ_ED)')
        axes[0, 1].set_ylabel('P(correct)')
        axes[0, 1].set_title('Classification Accuracy vs Border Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Mean scores vs border_ratio
        axes[1, 0].plot(border_ratios, pos_means, 'g-', linewidth=2, marker='^', label='Positive')
        axes[1, 0].plot(border_ratios, neg_means, 'r-', linewidth=2, marker='v', label='Negative')
        axes[1, 0].axvline(x=best_result['border_ratio'], color='b', linestyle='--')
        axes[1, 0].set_xlabel('Border Ratio (θ_ED)')
        axes[1, 0].set_ylabel('Mean Score')
        axes[1, 0].set_title('Mean Scores vs Border Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Separation (mean diff) vs border_ratio
        axes[1, 1].plot(border_ratios, mean_diffs, 'purple', linewidth=2, marker='D')
        axes[1, 1].axvline(x=best_result['border_ratio'], color='r', linestyle='--')
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Border Ratio (θ_ED)')
        axes[1, 1].set_ylabel('Mean Difference (Pos - Neg)')
        axes[1, 1].set_title('Separation vs Border Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('ED Bayesian Parameter Learning Results', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'ed_bayesian_learning.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"Saved ED Bayesian learning plot: {plot_path}")
    
    def _plot_ss_bayesian_results(self, param_results, best_result):
        """Plot SS Bayesian learning results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        n_segments = [r['n_segments'] for r in param_results]
        kl_divs = [r['kl_divergence'] for r in param_results]
        p_corrects = [r['p_correct'] for r in param_results]
        mean_diffs = [r['mean_diff'] for r in param_results]
        pos_means = [r['positive_mean'] for r in param_results]
        neg_means = [r['negative_mean'] for r in param_results]
        
        # 1. KL Divergence vs n_segments
        axes[0, 0].plot(n_segments, kl_divs, 'b-', linewidth=2, marker='o')
        axes[0, 0].axvline(x=best_result['n_segments'], color='r', linestyle='--',
                          label=f'Best: {best_result["n_segments"]}')
        axes[0, 0].set_xlabel('Number of Superpixels')
        axes[0, 0].set_ylabel('KL Divergence')
        axes[0, 0].set_title('KL Divergence vs n_segments')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. P(correct) vs n_segments
        axes[0, 1].plot(n_segments, p_corrects, 'g-', linewidth=2, marker='s')
        axes[0, 1].axvline(x=best_result['n_segments'], color='r', linestyle='--')
        axes[0, 1].set_xlabel('Number of Superpixels')
        axes[0, 1].set_ylabel('P(correct)')
        axes[0, 1].set_title('Classification Accuracy vs n_segments')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Mean scores vs n_segments
        axes[1, 0].plot(n_segments, pos_means, 'g-', linewidth=2, marker='^', label='Positive')
        axes[1, 0].plot(n_segments, neg_means, 'r-', linewidth=2, marker='v', label='Negative')
        axes[1, 0].axvline(x=best_result['n_segments'], color='b', linestyle='--')
        axes[1, 0].set_xlabel('Number of Superpixels')
        axes[1, 0].set_ylabel('Mean Score')
        axes[1, 0].set_title('Mean Scores vs n_segments')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Separation (mean diff) vs n_segments
        axes[1, 1].plot(n_segments, mean_diffs, 'purple', linewidth=2, marker='D')
        axes[1, 1].axvline(x=best_result['n_segments'], color='r', linestyle='--')
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Number of Superpixels')
        axes[1, 1].set_ylabel('Mean Difference (Pos - Neg)')
        axes[1, 1].set_title('Separation vs n_segments')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('SS Bayesian Parameter Learning Results', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'ss_bayesian_learning.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"Saved SS Bayesian learning plot: {plot_path}")
    
    def _plot_likelihood_distributions(self, likelihoods, title, save_path):
        """Plot likelihood distributions P(Score|Obj) and P(Score|Bg)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bin_centers = likelihoods['bin_centers']
        p_obj = likelihoods['p_score_given_obj']
        p_bg = likelihoods['p_score_given_bg']
        
        width = (bin_centers[1] - bin_centers[0]) * 0.4
        
        ax.bar(bin_centers - width/2, p_obj, width=width, 
               alpha=0.7, color='green', label='P(Score | Object)')
        ax.bar(bin_centers + width/2, p_bg, width=width,
               alpha=0.7, color='red', label='P(Score | Background)')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Saved likelihood plot: {save_path}")
    
    def run_bayesian_learning(self, n_images=10, n_windows=500):
        """
        Run complete Bayesian learning for all parameters
        """
        print("\n" + "="*70)
        print("COMPLETE BAYESIAN PARAMETER LEARNING")
        print("="*70)
        
        # 1. Learn ED parameters (Group A specific)
        print("\n[1/2] Learning ED parameters...")
        ed_params = self.learn_ed_parameters_bayesian(n_images, n_windows)
        
        # 2. Learn SS parameters
        print("\n[2/2] Learning SS parameters...")
        ss_params = self.learn_ss_parameters_bayesian(n_images, n_windows)
        
        # Combine results
        self.results = {
            'ed_params': ed_params,
            'ss_params': ss_params,
            'learning_method': 'bayesian',
            'n_training_images': n_images,
            'n_windows_per_image': n_windows
        }
        
        # Save results
        self.save_bayesian_results()
        
        print("\n" + "="*70)
        print("BAYESIAN LEARNING COMPLETE!")
        print("="*70)
        
        return self.results
    
    def save_bayesian_results(self):
        """Save Bayesian learning results"""
        # Convert numpy types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        python_results = convert_types(self.results)
        
        # Save to JSON
        output_file = os.path.join(self.output_dir, 'bayesian_parameters.json')
        with open(output_file, 'w') as f:
            json.dump(python_results, f, indent=2)
        
        print(f"\n✓ Saved Bayesian parameters to: {output_file}")
        
        # Create config file
        config = {
            # ED parameters
            'ed_border_ratio': python_results['ed_params']['border_ratio'],
            'ed_canny_low': python_results['ed_params']['canny_low'],
            'ed_canny_high': python_results['ed_params']['canny_high'],
            'ed_use_perimeter': True,
            
            # SS parameters
            'ss_n_segments': python_results['ss_params']['n_segments'],
            'ss_algorithm': 'slic',
            'ss_compactness': 10,
            
            # Other parameters (use defaults or previously learned)
            'window_scales': [0.5, 0.75, 1.0],
            'window_stride': 0.1,
            'top_k_windows': 10,
            'ms_threshold': 0.2,
            'ms_weight': 0.4,
            'ss_weight': 0.3,
            'ed_weight': 0.3,
        }
        
        config_file = os.path.join(self.output_dir, 'config_bayesian.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Created Bayesian config file: {config_file}")
        
        # Create summary
        self.create_bayesian_summary()
    
    def create_bayesian_summary(self):
        """Create summary of Bayesian learning results"""
        summary_file = os.path.join(self.output_dir, 'bayesian_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BAYESIAN PARAMETER LEARNING SUMMARY - GROUP A\n")
            f.write("="*70 + "\n\n")
            
            f.write("Learning Method: Bayesian (Histogram Likelihoods)\n")
            f.write("\n" + "-"*70 + "\n\n")
            
            f.write("1. EDGE DENSITY (ED) PARAMETERS:\n")
            ed_params = self.results['ed_params']
            f.write(f"   Border Ratio (θ_ED): {ed_params['border_ratio']:.3f}\n")
            f.write(f"   KL Divergence: {ed_params['bayesian_scores']['kl_divergence']:.4f}\n")
            f.write(f"   P(correct): {ed_params['bayesian_scores']['p_correct']:.4f}\n")
            f.write(f"   Mean Difference: {ed_params['bayesian_scores']['mean_diff']:.4f}\n")
            f.write("\n")
            
            f.write("2. SUPERPIXELS STRADDLING (SS) PARAMETERS:\n")
            ss_params = self.results['ss_params']
            f.write(f"   Number of Segments: {ss_params['n_segments']}\n")
            f.write(f"   KL Divergence: {ss_params['bayesian_scores']['kl_divergence']:.4f}\n")
            f.write(f"   P(correct): {ss_params['bayesian_scores']['p_correct']:.4f}\n")
            f.write(f"   Mean Difference: {ss_params['bayesian_scores']['mean_diff']:.4f}\n")
            f.write("\n")
            
            f.write("3. BAYESIAN LEARNING DETAILS:\n")
            f.write(f"   Training Images: {self.results['n_training_images']}\n")
            f.write(f"   Windows per Image: {self.results['n_windows_per_image']}\n")
            f.write(f"   Positive Sample Threshold: IoU > 0.5\n")
            f.write(f"   Negative Sample Threshold: IoU ≤ 0.5\n")
            f.write(f"   Histogram Bins: 20\n")
            f.write("\n")
            
            f.write("4. INTERPRETATION:\n")
            f.write("   - KL Divergence: Higher = better separation\n")
            f.write("   - P(correct): Probability of correct classification\n")
            f.write("   - Mean Difference: Positive mean - Negative mean\n")
            f.write("   - Values > 0 indicate good separation\n")
            f.write("\n")
            
            f.write("5. FILES GENERATED:\n")
            f.write("   - bayesian_parameters.json: All learned parameters\n")
            f.write("   - config_bayesian.json: Ready-to-use config\n")
            f.write("   - ed_bayesian_learning.png: ED learning curves\n")
            f.write("   - ss_bayesian_learning.png: SS learning curves\n")
            f.write("   - ed_likelihoods_best.png: Best ED likelihoods\n")
            f.write("   - ss_likelihoods_best.png: Best SS likelihoods\n")
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Created Bayesian summary: {summary_file}")

# Test function
def test_bayesian_learning():
    """Test Bayesian learning with synthetic data"""
    print("Testing Bayesian Parameter Learning...")
    
    # Create synthetic dataset
    test_dir = 'test_bayesian_data'
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'Annotations'), exist_ok=True)
    
    # Create 3 test images
    for i in range(3):
        img = np.ones((200, 300, 3), dtype=np.uint8) * 100
        
        # Add object
        x1, y1 = np.random.randint(30, 150), np.random.randint(30, 120)
        x2, y2 = x1 + np.random.randint(60, 100), y1 + np.random.randint(60, 80)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
        # Add texture
        for j in range(0, 300, 8):
            cv2.line(img, (j, 0), (j, 200), (150, 150, 150), 1)
        
        # Save image
        img_path = os.path.join(test_dir, 'JPEGImages', f'{i:06d}.jpg')
        cv2.imwrite(img_path, img)
        
        # Create annotation
        xml_content = f"""<?xml version="1.0"?>
<annotation>
    <filename>{i:06d}.jpg</filename>
    <size>
        <width>300</width>
        <height>200</height>
        <depth>3</depth>
    </size>
    <object>
        <name>object</name>
        <bndbox>
            <xmin>{x1}</xmin>
            <ymin>{y1}</ymin>
            <xmax>{x2}</xmax>
            <ymax>{y2}</ymax>
        </bndbox>
    </object>
</annotation>"""
        
        xml_path = os.path.join(test_dir, 'Annotations', f'{i:06d}.xml')
        with open(xml_path, 'w') as f:
            f.write(xml_content)
    
    print(f"Created synthetic dataset with 3 images in {test_dir}")
    
    # Run Bayesian learning
    learner = BayesianParameterLearner(test_dir, output_dir='bayesian_test_results')
    results = learner.run_bayesian_learning(n_images=3, n_windows=200)
    
    print("\n" + "="*70)
    print("BAYESIAN LEARNING TEST COMPLETE")
    print("="*70)
    print(f"Results saved in: bayesian_test_results/")
    print("="*70)

if __name__ == "__main__":
    test_bayesian_learning()