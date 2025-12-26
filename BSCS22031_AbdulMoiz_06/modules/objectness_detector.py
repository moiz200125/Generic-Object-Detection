# modules/objectness_detector.py
"""
Complete Objectness Detector combining MS + SS + ED cues
"""
import numpy as np
import cv2
import json
from datetime import datetime
import os

from .ms_cue import MultiScaleSaliencyCue
from .ss_cue import SuperpixelStraddlingCue
from .ed_cue import EdgeDensityCue
from .integral_image import IntegralImage

def calculate_iou(boxA, boxB):
    """Calculate IoU between two windows"""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes
    """
    if len(boxes) == 0:
        return []
    
    # Sort indices by score (descending)
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    
    while len(idxs) > 0:
        # Pick the current highest score box
        last = idxs.shape[0]
        i = idxs[0]
        pick.append(i)
        
        # Find IoU of this box with all others
        ious = []
        for j in range(1, last):
            iou = calculate_iou(boxes[i], boxes[idxs[j]])
            ious.append(iou)
        
        # Keep only boxes with IoU less than threshold
        idxs = np.delete(idxs, np.concatenate(([0], np.where(np.array(ious) > iou_threshold)[0] + 1)))
        
    return pick

class ObjectnessDetector:
    """
    Complete Objectness Detector for Group A
    Combines: MS + SS + ED cues
    """
    
    def __init__(self, config_path=None, use_learned_params=True):
        """
        Initialize with optional learned parameters
        
        Args:
            config_path: Path to config file
            use_learned_params: If True, use learned parameters from file
        """
        # Load learned parameters if available
        if use_learned_params:
            learned_path = 'learned_parameters/learned_parameters.json'
            if os.path.exists(learned_path):
                with open(learned_path, 'r') as f:
                    self.learned_params = json.load(f)
                print("✓ Loaded learned parameters")
            else:
                self.learned_params = {}
                print("⚠️ No learned parameters found, using defaults")


                
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Default parameters
        self.default_params = {
            # Window generation
            'window_scales': [0.5, 0.75, 1.0, 1.5, 2.0],
            'window_stride': 0.1,
            'top_k_windows': 10,
            
            # MS parameters
            'ms_threshold': 0.2,
            
            # SS parameters
            'ss_n_segments': 100,
            'ss_algorithm': 'slic',
            
            # ED parameters
            'ed_border_ratio': 0.1,
            'ed_canny_low': 50,
            'ed_canny_high': 150,
            'ed_use_perimeter': True,
            
            # Cue weights (sum should be 1.0)
            'ms_weight': 0.4,
            'ss_weight': 0.3,
            'ed_weight': 0.3,
        }
        
        # Merge with config
        self.params = {**self.default_params, **self.config}
        
        # Normalize weights to sum to 1.0
        total_weight = self.params['ms_weight'] + self.params['ss_weight'] + self.params['ed_weight']
        if total_weight > 0:
            self.params['ms_weight'] /= total_weight
            self.params['ss_weight'] /= total_weight
            self.params['ed_weight'] /= total_weight
        
        print("="*60)
        print("GROUP A OBJECTNESS DETECTOR")
        print("="*60)
        print(f"Cues: MS + SS + ED")
        print(f"Weights: MS={self.params['ms_weight']:.2f}, "
              f"SS={self.params['ss_weight']:.2f}, "
              f"ED={self.params['ed_weight']:.2f}")
        print("="*60)
    
    def generate_windows(self, image_shape):
        """
        Generate sliding windows at multiple scales
        """
        height, width = image_shape[:2]
        windows = []
        
        for scale in self.params['window_scales']:
            # Calculate window size
            window_width = int(width * scale)
            window_height = int(height * scale)
            
            # Skip if window is larger than image
            if window_width > width or window_height > height:
                continue
            
            # Calculate stride
            stride_x = max(1, int(width * self.params['window_stride']))
            stride_y = max(1, int(height * self.params['window_stride']))
            
            # Generate windows
            for y in range(0, height - window_height + 1, stride_y):
                for x in range(0, width - window_width + 1, stride_x):
                    windows.append((x, y, x + window_width, y + window_height))
        
        print(f"Generated {len(windows)} windows")
        return windows
    
    def process_image(self, image):
        """
        Process image using all three cues
        
        Returns:
            List of (window, total_score, ms_score, ss_score, ed_score) tuples
        """
        print(f"\nProcessing image: {image.shape}")
        
        # Initialize cues
        print("Initializing cues...")
        
        # 1. Multi-scale Saliency (MS)
        ms = MultiScaleSaliencyCue(image)
        ms_thresholds = {scale: self.params['ms_threshold'] for scale in ms.scales}
        ms.set_thresholds(ms_thresholds)
        
        # 2. Superpixels Straddling (SS)
        ss = SuperpixelStraddlingCue(
            image,
            algorithm=self.params['ss_algorithm'],
            n_segments=self.params['ss_n_segments']
        )
        
        # 3. Edge Density (ED) - Group A specific
        ed = EdgeDensityCue(
            image,
            border_ratio=self.params['ed_border_ratio'],
            canny_low=self.params['ed_canny_low'],
            canny_high=self.params['ed_canny_high']
        )
        
        # Compute features
        print("\nComputing features...")
        print("1. Multi-scale Saliency...")
        ms.compute_saliency_maps()
        
        print("2. Superpixels...")
        ss.compute_superpixels()
        
        print("3. Edge Map (already computed during initialization)")
        
        # Generate windows
        windows = self.generate_windows(image.shape)
        
        # Score windows
        print(f"\nScoring {len(windows)} windows...")
        window_scores = []
        
        for i, window in enumerate(windows):
            # Get individual cue scores
            ms_score = ms.get_score(window)
            ss_score = ss.get_score(window)
            ed_score = ed.get_score(window, use_perimeter=self.params['ed_use_perimeter'])
            
            # Combine scores with weights
            total_score = (
                self.params['ms_weight'] * ms_score +
                self.params['ss_weight'] * ss_score +
                self.params['ed_weight'] * ed_score
            )
            
            # Store all scores
            window_scores.append((window, total_score, ms_score, ss_score, ed_score))
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"  Scored {i + 1}/{len(windows)} windows...")
        
        # Sort by total score (descending)
        window_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K windows
        # Optional: Apply NMS if requested in params
        if self.params.get('apply_nms', True):
            # Extract boxes and scores
            boxes = [w[0] for w in window_scores]
            scores = [w[1] for w in window_scores]
            nms_threshold = self.params.get('nms_threshold', 0.3)
            
            pick_indices = non_max_suppression(boxes, scores, nms_threshold)
            filtered_scores = [window_scores[i] for i in pick_indices]
            window_scores = filtered_scores

        top_k = min(self.params['top_k_windows'], len(window_scores))
        return window_scores[:top_k], (ms, ss, ed)
    
    def visualize_results(self, image, top_windows, cues, save_path=None):
        """
        Create comprehensive visualization
        
        Args:
            image: Original image
            top_windows: List of (window, total, ms, ss, ed) tuples
            cues: Tuple of (ms_cue, ss_cue, ed_cue)
            save_path: Path to save visualization
        """
        ms_cue, ss_cue, ed_cue = cues
        
        # Create main visualization
        h, w = image.shape[:2]
        
        # Create a 2x2 grid visualization
        grid_h = h * 2
        grid_w = w * 2
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # 1. Original with top windows (top-left)
        vis_original = image.copy()
        colors = [(0, 255, 0), (0, 200, 100), (255, 255, 0)]
        
        for i, (window, total, ms_score, ss_score, ed_score) in enumerate(top_windows[:3]):
            x1, y1, x2, y2 = window
            color = colors[i % len(colors)]
            
            # Draw window
            cv2.rectangle(vis_original, (x1, y1), (x2, y2), color, 3)
            
            # Label with scores
            label = f"{i+1}: T={total:.3f}"
            cv2.putText(vis_original, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(vis_original, "Top Windows", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 2. MS saliency map (top-right)
        ms_vis = np.mean(list(ms_cue.saliency_maps.values()), axis=0)
        ms_vis = (ms_vis * 255).astype(np.uint8)
        ms_vis = cv2.applyColorMap(ms_vis, cv2.COLORMAP_JET)
        
        # Draw top window on MS map
        if top_windows:
            top_window = top_windows[0][0]
            x1, y1, x2, y2 = top_window
            cv2.rectangle(ms_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        cv2.putText(ms_vis, "MS Saliency", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 3. Superpixels (bottom-left)
        sp_vis = ss_cue.visualize_superpixels(top_windows[0][0] if top_windows else None)
        cv2.putText(sp_vis, "Superpixels (SS)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 4. Edge Density (bottom-right)
        ed_vis = ed_cue.visualize_edges(top_windows[0][0] if top_windows else None)
        
        # Place visualizations in grid
        grid[0:h, 0:w] = vis_original
        grid[0:h, w:w*2] = cv2.resize(ms_vis, (w, h))
        grid[h:h*2, 0:w] = cv2.resize(sp_vis, (w, h))
        grid[h:h*2, w:w*2] = cv2.resize(ed_vis, (w, h))
        
        # Add overall title
        cv2.putText(grid, "GROUP A: MS + SS + ED Objectness Detection", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Add weights
        weights_text = (f"Weights: MS={self.params['ms_weight']:.2f}, "
                       f"SS={self.params['ss_weight']:.2f}, "
                       f"ED={self.params['ed_weight']:.2f}")
        cv2.putText(grid, weights_text, (10, grid_h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, grid)
            print(f"✓ Saved comprehensive visualization to: {save_path}")
        
        return grid
    
    def save_results(self, image_name, top_windows, output_dir):
        """
        Save detailed results to text file
        """
        result_file = os.path.join(output_dir, f"{image_name}_results.txt")
        
        with open(result_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GROUP A OBJECTNESS DETECTION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Image: {image_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Top Windows: {len(top_windows)}\n\n")
            
            f.write(f"Parameters:\n")
            f.write(f"  MS threshold: {self.params['ms_threshold']}\n")
            f.write(f"  SS segments: {self.params['ss_n_segments']}\n")
            f.write(f"  ED border ratio: {self.params['ed_border_ratio']}\n")
            f.write(f"  Weights: MS={self.params['ms_weight']:.2f}, "
                   f"SS={self.params['ss_weight']:.2f}, "
                   f"ED={self.params['ed_weight']:.2f}\n\n")
            
            f.write("-"*60 + "\n\n")
            
            for i, (window, total, ms_score, ss_score, ed_score) in enumerate(top_windows):
                x1, y1, x2, y2 = window
                w, h = x2-x1, y2-y1
                
                f.write(f"Window {i+1}:\n")
                f.write(f"  Coordinates: [{x1}, {y1}, {x2}, {y2}]\n")
                f.write(f"  Dimensions: {w} x {h}\n")
                f.write(f"  Total Score: {total:.6f}\n")
                f.write(f"    - MS Score: {ms_score:.6f} ({self.params['ms_weight']*100:.0f}%)\n")
                f.write(f"    - SS Score: {ss_score:.6f} ({self.params['ss_weight']*100:.0f}%)\n")
                f.write(f"    - ED Score: {ed_score:.6f} ({self.params['ed_weight']*100:.0f}%)\n")
                
                # Show contribution breakdown
                ms_contrib = self.params['ms_weight'] * ms_score
                ss_contrib = self.params['ss_weight'] * ss_score
                ed_contrib = self.params['ed_weight'] * ed_score
                
                f.write(f"  Contributions:\n")
                f.write(f"    - MS: {ms_contrib:.6f} ({ms_contrib/total*100:.1f}% of total)\n")
                f.write(f"    - SS: {ss_contrib:.6f} ({ss_contrib/total*100:.1f}% of total)\n")
                f.write(f"    - ED: {ed_contrib:.6f} ({ed_contrib/total*100:.1f}% of total)\n")
                
                f.write("-"*40 + "\n\n")
        
        print(f"✓ Saved detailed results to: {result_file}")