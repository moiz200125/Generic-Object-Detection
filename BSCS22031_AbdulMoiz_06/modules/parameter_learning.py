# modules/parameter_learning.py
"""
Parameter Learning Module
Learns optimal parameters for MS, SS, and ED cues using PASCAL VOC dataset
"""
import os
import numpy as np
import cv2
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Import your cues
try:
    from .ms_cue import MultiScaleSaliencyCue
    from .ss_cue import SuperpixelStraddlingCue
    from .ed_cue import EdgeDensityCue
except ImportError:
    from ms_cue import MultiScaleSaliencyCue
    from ss_cue import SuperpixelStraddlingCue
    from ed_cue import EdgeDensityCue

class ParameterLearner:
    """
    Learns optimal parameters for Objectness cues
    """
    
    def __init__(self, voc_dataset_path, output_dir='learned_parameters'):
        """
        Initialize with PASCAL VOC dataset
        
        Args:
            voc_dataset_path: Path to VOC dataset (should have JPEGImages/ and Annotations/)
            output_dir: Directory to save learned parameters
        """
        self.voc_path = voc_dataset_path
        self.images_path = os.path.join(voc_dataset_path, "JPEGImages")
        self.annotations_path = os.path.join(voc_dataset_path, "Annotations")
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.images_path) 
                           if f.endswith('.jpg')]
        
        print(f"Found {len(self.image_files)} images in dataset")
        print(f"Dataset path: {voc_dataset_path}")
        
        # Cache for ground truth boxes
        self.gt_cache = {}
        
        # Results storage
        self.results = {
            'ms_thresholds': {},
            'ed_params': {},
            'ss_params': {},
            'weights': {}
        }
    
    def load_ground_truth(self, image_name):
        """
        Load ground truth bounding boxes from XML annotation
        
        Args:
            image_name: Name of image file without extension
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        if image_name in self.gt_cache:
            return self.gt_cache[image_name]
        
        xml_path = os.path.join(self.annotations_path, f"{image_name}.xml")
        
        if not os.path.exists(xml_path):
            print(f"Warning: No annotation for {image_name}")
            self.gt_cache[image_name] = []
            return []
        
        boxes = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                # Check if object is not difficult
                difficult = obj.find('difficult')
                if difficult is not None and difficult.text == '1':
                    continue
                
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    x1 = int(float(bndbox.find('xmin').text))
                    y1 = int(float(bndbox.find('ymin').text))
                    x2 = int(float(bndbox.find('xmax').text))
                    y2 = int(float(bndbox.find('ymax').text))
                    
                    # Ensure valid box
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))
        
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            boxes = []
        
        self.gt_cache[image_name] = boxes
        return boxes
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1, box2: (x1, y1, x2, y2)
            
        Returns:
            IoU score [0, 1]
        """
        # Calculate intersection coordinates
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])
        
        # Calculate intersection area
        inter_width = max(0, x2_i - x1_i)
        inter_height = max(0, y2_i - y1_i)
        inter_area = inter_width * inter_height
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def find_best_iou_with_gt(self, box, gt_boxes):
        """
        Find best IoU between a box and any ground truth box
        
        Args:
            box: Proposed box (x1, y1, x2, y2)
            gt_boxes: List of ground truth boxes
            
        Returns:
            Best IoU score
        """
        if not gt_boxes:
            return 0.0
        
        best_iou = 0.0
        for gt_box in gt_boxes:
            iou = self.calculate_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
        
        return best_iou
    
    def learn_ms_thresholds(self, n_images=10, scales=[16, 24, 32, 48, 64]):
        """
        Learn optimal thresholds for Multi-scale Saliency (MS)
        
        Steps:
        1. For each scale, generate saliency map
        2. Try different thresholds
        3. Binarize and get connected components as detections
        4. Calculate IoU with ground truth
        5. Pick threshold that maximizes IoU
        """
        print("\n" + "="*60)
        print("LEARNING MS THRESHOLDS")
        print("="*60)
        
        # Select training images
        train_images = self.image_files[:min(n_images, len(self.image_files))]
        print(f"Using {len(train_images)} images for training")
        
        # Threshold range to search
        threshold_range = np.arange(0.1, 0.91, 0.05)
        
        # Initialize results for each scale
        scale_results = {scale: {'thresholds': [], 'ious': []} for scale in scales}
        
        for img_file in tqdm(train_images, desc="Processing images"):
            # Load image
            img_path = os.path.join(self.images_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Could not load {img_file}")
                continue
            
            # Get ground truth
            img_name = os.path.splitext(img_file)[0]
            gt_boxes = self.load_ground_truth(img_name)
            
            if not gt_boxes:
                continue
            
            # Initialize MS cue
            ms = MultiScaleSaliencyCue(image)
            ms.compute_saliency_maps()
            
            # For each scale
            for scale in scales:
                if scale not in ms.saliency_maps:
                    continue
                
                saliency_map = ms.saliency_maps[scale]
                
                # Try each threshold
                for threshold in threshold_range:
                    # Binarize saliency map
                    binary_map = (saliency_map > threshold).astype(np.uint8) * 255
                    
                    # Find connected components (blobs)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        binary_map, connectivity=8
                    )
                    
                    total_iou = 0.0
                    detection_count = 0
                    
                    # Skip background (label 0)
                    for label in range(1, num_labels):
                        # Get blob bounding box
                        x = stats[label, cv2.CC_STAT_LEFT]
                        y = stats[label, cv2.CC_STAT_TOP]
                        w = stats[label, cv2.CC_STAT_WIDTH]
                        h = stats[label, cv2.CC_STAT_HEIGHT]
                        area = stats[label, cv2.CC_STAT_AREA]
                        
                        # Filter small blobs
                        if area < 100:  # Minimum area threshold
                            continue
                        
                        blob_box = (x, y, x + w, y + h)
                        
                        # Calculate best IoU with any GT box
                        best_iou = self.find_best_iou_with_gt(blob_box, gt_boxes)
                        
                        # Only count if IoU > 0.1 (meaningful detection)
                        if best_iou > 0.1:
                            total_iou += best_iou
                            detection_count += 1
                    
                    # Calculate average IoU
                    avg_iou = total_iou / max(detection_count, 1)
                    
                    # Store result
                    scale_results[scale]['thresholds'].append(threshold)
                    scale_results[scale]['ious'].append(avg_iou)
        
        # Find best threshold for each scale
        best_thresholds = {}
        
        for scale in scales:
            if not scale_results[scale]['thresholds']:
                best_thresholds[scale] = 0.5  # Default
                continue
            
            # Find threshold with maximum average IoU
            thresholds = np.array(scale_results[scale]['thresholds'])
            ious = np.array(scale_results[scale]['ious'])
            
            # Group by threshold and average
            unique_thresholds = np.unique(thresholds)
            avg_ious = []
            
            for t in unique_thresholds:
                mask = thresholds == t
                if np.any(mask):
                    avg_ious.append(np.mean(ious[mask]))
                else:
                    avg_ious.append(0)
            
            # Find best threshold
            best_idx = np.argmax(avg_ious)
            best_threshold = unique_thresholds[best_idx]
            best_iou = avg_ious[best_idx]
            
            best_thresholds[scale] = float(best_threshold)
            
            print(f"\nScale {scale}:")
            print(f"  Best threshold: {best_threshold:.3f}")
            print(f"  Average IoU: {best_iou:.4f}")
            
            # Plot threshold vs IoU
            self._plot_threshold_curve(unique_thresholds, avg_ious, 
                                      scale, best_threshold, best_iou)
        
        self.results['ms_thresholds'] = best_thresholds
        
        print("\n" + "-"*60)
        print("MS THRESHOLD LEARNING COMPLETE")
        print("-"*60)
        
        return best_thresholds
    
    def _plot_threshold_curve(self, thresholds, ious, scale, best_threshold, best_iou):
        """Plot threshold vs IoU curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, ious, 'b-', linewidth=2, marker='o')
        plt.axvline(x=best_threshold, color='r', linestyle='--', 
                   label=f'Best: {best_threshold:.3f}')
        plt.axhline(y=best_iou, color='g', linestyle='--', 
                   label=f'IoU: {best_iou:.3f}')
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Average IoU', fontsize=12)
        plt.title(f'MS Threshold Learning (Scale {scale})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'ms_threshold_scale_{scale}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"  Saved plot: {plot_path}")
    
    def learn_ed_parameters(self, n_images=10, n_windows_per_image=200):
        """
        Learn optimal parameters for Edge Density (ED) cue
        
        Steps:
        1. Generate positive samples (IoU > 0.5 with GT)
        2. Generate negative samples (IoU < 0.3 with GT)
        3. Try different border_ratio values
        4. Calculate ED scores for both positive and negative samples
        5. Choose parameters that best separate positive from negative
        """
        print("\n" + "="*60)
        print("LEARNING ED PARAMETERS")
        print("="*60)
        
        # Select training images
        train_images = self.image_files[:min(n_images, len(self.image_files))]
        print(f"Using {len(train_images)} images for training")
        
        # Parameter ranges to search
        border_ratios = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
        canny_lows = [30, 50, 70]
        canny_highs = [100, 150, 200]
        
        # Store scores for parameter combinations
        param_scores = []
        
        for img_file in tqdm(train_images, desc="Processing images"):
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
            
            h, w = image.shape[:2]
            
            # Generate random windows
            positive_scores = {br: [] for br in border_ratios}
            negative_scores = {br: [] for br in border_ratios}
            
            for _ in range(n_windows_per_image):
                # Generate random window
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
                    label = 'positive'
                elif best_iou < 0.3:
                    label = 'negative'
                else:
                    continue  # Skip ambiguous windows
                
                # Test different border ratios
                for border_ratio in border_ratios:
                    # Initialize ED cue with current parameters
                    ed = EdgeDensityCue(
                        image,
                        border_ratio=border_ratio,
                        canny_low=50,  # Use defaults for now
                        canny_high=150
                    )
                    
                    # Calculate ED score
                    score = ed.get_score(window)
                    
                    if label == 'positive':
                        positive_scores[border_ratio].append(score)
                    else:
                        negative_scores[border_ratio].append(score)
            
            # Calculate separation metric for this image
            for border_ratio in border_ratios:
                if positive_scores[border_ratio] and negative_scores[border_ratio]:
                    pos_mean = np.mean(positive_scores[border_ratio])
                    neg_mean = np.mean(negative_scores[border_ratio])
                    pos_std = np.std(positive_scores[border_ratio])
                    neg_std = np.std(negative_scores[border_ratio])
                    
                    # Separation metric: difference in means normalized by combined std
                    if pos_std + neg_std > 0:
                        separation = (pos_mean - neg_mean) / (pos_std + neg_std)
                    else:
                        separation = 0
                    
                    param_scores.append({
                        'border_ratio': border_ratio,
                        'separation': separation,
                        'pos_mean': pos_mean,
                        'neg_mean': neg_mean
                    })
        
        # Find best border ratio
        if not param_scores:
            print("Warning: No valid samples collected")
            best_border_ratio = 0.1  # Default
        else:
            # Average separation for each border ratio
            ratio_scores = {}
            for br in border_ratios:
                br_scores = [ps for ps in param_scores if ps['border_ratio'] == br]
                if br_scores:
                    avg_separation = np.mean([ps['separation'] for ps in br_scores])
                    ratio_scores[br] = avg_separation
            
            if ratio_scores:
                best_border_ratio = max(ratio_scores.items(), key=lambda x: x[1])[0]
                best_separation = ratio_scores[best_border_ratio]
            else:
                best_border_ratio = 0.1
                best_separation = 0
        
        print(f"\nBest border_ratio: {best_border_ratio:.3f}")
        print(f"Separation score: {best_separation:.4f}")
        
        # Plot separation vs border_ratio
        self._plot_ed_parameter_curve(border_ratios, ratio_scores, best_border_ratio)
        
        # Store results
        self.results['ed_params'] = {
            'border_ratio': float(best_border_ratio),
            'canny_low': 50,  # Could also learn these
            'canny_high': 150,
            'use_perimeter': True
        }
        
        print("\n" + "-"*60)
        print("ED PARAMETER LEARNING COMPLETE")
        print("-"*60)
        
        return self.results['ed_params']
    
    def _plot_ed_parameter_curve(self, border_ratios, ratio_scores, best_ratio):
        """Plot ED parameter learning curve"""
        if not ratio_scores:
            return
        
        x = []
        y = []
        for br in border_ratios:
            if br in ratio_scores:
                x.append(br)
                y.append(ratio_scores[br])
        
        if len(x) < 2:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', linewidth=2, marker='o')
        plt.axvline(x=best_ratio, color='r', linestyle='--', 
                   label=f'Best: {best_ratio:.3f}')
        
        plt.xlabel('Border Ratio', fontsize=12)
        plt.ylabel('Separation Score', fontsize=12)
        plt.title('ED Parameter Learning (Border Ratio)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'ed_parameter_learning.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"Saved plot: {plot_path}")
    
    def learn_ss_parameters(self, n_images=10):
        """
        Learn optimal parameters for Superpixels Straddling (SS)
        Main parameter: number of superpixels (n_segments)
        """
        print("\n" + "="*60)
        print("LEARNING SS PARAMETERS")
        print("="*60)
        
        train_images = self.image_files[:min(n_images, len(self.image_files))]
        print(f"Using {len(train_images)} images for training")
        
        # Test different numbers of superpixels
        n_segments_options = [50, 100, 150, 200, 250]
        
        segment_scores = {n: [] for n in n_segments_options}
        
        for img_file in tqdm(train_images, desc="Processing images"):
            img_path = os.path.join(self.images_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
            
            img_name = os.path.splitext(img_file)[0]
            gt_boxes = self.load_ground_truth(img_name)
            
            if not gt_boxes:
                continue
            
            h, w = image.shape[:2]
            
            # For each n_segments value
            for n_segments in n_segments_options:
                # Initialize SS cue
                ss = SuperpixelStraddlingCue(image, n_segments=n_segments)
                ss.compute_superpixels()
                
                # Score ground truth boxes
                gt_scores = []
                for gt_box in gt_boxes:
                    score = ss.get_score(gt_box)
                    gt_scores.append(score)
                
                if gt_scores:
                    avg_score = np.mean(gt_scores)
                    segment_scores[n_segments].append(avg_score)
        
        # Find best n_segments
        avg_scores = {}
        for n in n_segments_options:
            if segment_scores[n]:
                avg_scores[n] = np.mean(segment_scores[n])
            else:
                avg_scores[n] = 0
        
        if avg_scores:
            best_n_segments = max(avg_scores.items(), key=lambda x: x[1])[0]
            best_score = avg_scores[best_n_segments]
        else:
            best_n_segments = 100  # Default
            best_score = 0
        
        print(f"\nBest n_segments: {best_n_segments}")
        print(f"Average SS score on GT boxes: {best_score:.4f}")
        
        # Plot results
        self._plot_ss_parameter_curve(n_segments_options, avg_scores, best_n_segments)
        
        # Store results
        self.results['ss_params'] = {
            'n_segments': int(best_n_segments),
            'algorithm': 'slic',
            'compactness': 10
        }
        
        print("\n" + "-"*60)
        print("SS PARAMETER LEARNING COMPLETE")
        print("-"*60)
        
        return self.results['ss_params']
    
    def _plot_ss_parameter_curve(self, n_segments_options, avg_scores, best_n):
        """Plot SS parameter learning curve"""
        x = n_segments_options
        y = [avg_scores.get(n, 0) for n in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', linewidth=2, marker='o')
        plt.axvline(x=best_n, color='r', linestyle='--', 
                   label=f'Best: {best_n}')
        
        plt.xlabel('Number of Superpixels', fontsize=12)
        plt.ylabel('Average SS Score on GT Boxes', fontsize=12)
        plt.title('SS Parameter Learning (n_segments)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'ss_parameter_learning.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"Saved plot: {plot_path}")
    
    def learn_cue_weights(self, n_images=10, n_windows_per_image=100):
        """
        Learn optimal weights for combining MS, SS, and ED cues
        
        Steps:
        1. Generate windows and calculate all cue scores
        2. Classify as positive (IoU > 0.5) or negative (IoU < 0.3)
        3. Search weight combinations that maximize separation
        """
        print("\n" + "="*60)
        print("LEARNING CUE WEIGHTS")
        print("="*60)
        
        train_images = self.image_files[:min(n_images, len(self.image_files))]
        print(f"Using {len(train_images)} images for training")
        
        # Collect scores for all windows
        all_scores = []  # Each element: (ms_score, ss_score, ed_score, label)
        
        for img_file in tqdm(train_images, desc="Processing images"):
            img_path = os.path.join(self.images_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
            
            img_name = os.path.splitext(img_file)[0]
            gt_boxes = self.load_ground_truth(img_name)
            
            if not gt_boxes:
                continue
            
            h, w = image.shape[:2]
            
            # Initialize cues with learned parameters
            ms = MultiScaleSaliencyCue(image)
            if self.results['ms_thresholds']:
                ms.set_thresholds(self.results['ms_thresholds'])
            ms.compute_saliency_maps()
            
            ss = SuperpixelStraddlingCue(
                image, 
                n_segments=self.results.get('ss_params', {}).get('n_segments', 100)
            )
            ss.compute_superpixels()
            
            ed = EdgeDensityCue(
                image,
                border_ratio=self.results.get('ed_params', {}).get('border_ratio', 0.1)
            )
            
            # Generate and score windows
            for _ in range(n_windows_per_image):
                win_w = np.random.randint(30, min(200, w//2))
                win_h = np.random.randint(30, min(200, h//2))
                x = np.random.randint(0, w - win_w)
                y = np.random.randint(0, h - win_h)
                
                window = (x, y, x + win_w, y + win_h)
                
                # Calculate IoU
                best_iou = 0.0
                for gt_box in gt_boxes:
                    iou = self.calculate_iou(window, gt_box)
                    best_iou = max(best_iou, iou)
                
                # Classify
                if best_iou > 0.5:
                    label = 1  # Positive
                elif best_iou < 0.3:
                    label = 0  # Negative
                else:
                    continue  # Skip ambiguous
                
                # Get cue scores
                ms_score = ms.get_score(window)
                ss_score = ss.get_score(window)
                ed_score = ed.get_score(window)
                
                all_scores.append((ms_score, ss_score, ed_score, label))
        
        if len(all_scores) < 20:
            print("Warning: Not enough samples for weight learning")
            self.results['weights'] = {'ms': 0.4, 'ss': 0.3, 'ed': 0.3}
            return self.results['weights']
        
        # Convert to numpy arrays
        scores = np.array([s[:3] for s in all_scores])
        labels = np.array([s[3] for s in all_scores])
        
        # Grid search for weights
        best_weights = None
        best_separation = -np.inf
        
        # Search weight space (sums to 1)
        weight_step = 0.1
        for w1 in np.arange(0, 1.01, weight_step):
            for w2 in np.arange(0, 1 - w1 + 0.01, weight_step):
                w3 = 1 - w1 - w2
                
                if w3 < 0:
                    continue
                
                # Calculate combined scores
                combined = w1 * scores[:, 0] + w2 * scores[:, 1] + w3 * scores[:, 2]
                
                # Separate positive and negative
                pos_scores = combined[labels == 1]
                neg_scores = combined[labels == 0]
                
                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    # Calculate separation
                    pos_mean = np.mean(pos_scores)
                    neg_mean = np.mean(neg_scores)
                    pos_std = np.std(pos_scores)
                    neg_std = np.std(neg_scores)
                    
                    if pos_std + neg_std > 0:
                        separation = (pos_mean - neg_mean) / (pos_std + neg_std)
                    else:
                        separation = 0
                    
                    if separation > best_separation:
                        best_separation = separation
                        best_weights = (w1, w2, w3)
        
        if best_weights:
            print(f"\nBest weights found:")
            print(f"  MS weight: {best_weights[0]:.3f}")
            print(f"  SS weight: {best_weights[1]:.3f}")
            print(f"  ED weight: {best_weights[2]:.3f}")
            print(f"  Separation: {best_separation:.4f}")
            
            self.results['weights'] = {
                'ms': float(best_weights[0]),
                'ss': float(best_weights[1]),
                'ed': float(best_weights[2])
            }
        else:
            print("\nUsing default weights")
            self.results['weights'] = {'ms': 0.4, 'ss': 0.3, 'ed': 0.3}
        
        # Plot weight space
        self._plot_weight_space_analysis(scores, labels)
        
        print("\n" + "-"*60)
        print("WEIGHT LEARNING COMPLETE")
        print("-"*60)
        
        return self.results['weights']
    
    def _plot_weight_space_analysis(self, scores, labels):
        """Plot weight space analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Score distributions for each cue
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        cue_names = ['MS', 'SS', 'ED']
        colors = ['red', 'blue', 'green']
        
        for i in range(3):
            ax = axes[0, i]
            ax.hist(pos_scores[:, i], bins=20, alpha=0.7, label='Positive', color='green')
            ax.hist(neg_scores[:, i], bins=20, alpha=0.7, label='Negative', color='red')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{cue_names[i]} Cue Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Scatter plots of cue pairs
        pairs = [(0, 1), (0, 2), (1, 2)]
        pair_names = ['MS vs SS', 'MS vs ED', 'SS vs ED']
        
        for idx, (i, j) in enumerate(pairs):
            ax = axes[1, idx]
            
            # Positive samples
            ax.scatter(pos_scores[:, i], pos_scores[:, j], 
                      alpha=0.6, color='green', label='Positive', s=20)
            
            # Negative samples
            ax.scatter(neg_scores[:, i], neg_scores[:, j], 
                      alpha=0.6, color='red', label='Negative', s=20)
            
            ax.set_xlabel(cue_names[i])
            ax.set_ylabel(cue_names[j])
            ax.set_title(pair_names[idx])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'weight_analysis.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"Saved weight analysis plot: {plot_path}")
    
    def run_complete_learning(self, n_images=10):
        """
        Run complete parameter learning pipeline
        """
        print("\n" + "="*70)
        print("COMPLETE PARAMETER LEARNING PIPELINE")
        print("="*70)
        
        # 1. Learn MS thresholds
        ms_thresholds = self.learn_ms_thresholds(n_images=n_images)
        
        # 2. Learn ED parameters
        ed_params = self.learn_ed_parameters(n_images=n_images)
        
        # 3. Learn SS parameters
        ss_params = self.learn_ss_parameters(n_images=n_images)
        
        # 4. Learn cue weights
        weights = self.learn_cue_weights(n_images=n_images)
        
        # Combine all results
        self.results = {
            'ms_thresholds': ms_thresholds,
            'ed_params': ed_params,
            'ss_params': ss_params,
            'weights': weights
        }
        
        # Save results
        self.save_results()
        
        # Create summary
        self.create_learning_summary()
        
        return self.results
    
    def save_results(self):
        """Save learned parameters to JSON file"""
        # Convert numpy types to Python types
        def convert_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(item) for item in obj]
            else:
                return obj
        
        python_results = convert_to_python(self.results)
        
        # Save to JSON
        output_file = os.path.join(self.output_dir, 'learned_parameters.json')
        with open(output_file, 'w') as f:
            json.dump(python_results, f, indent=2)
        
        print(f"\n✓ Saved learned parameters to: {output_file}")
        
        # Also create a config file for direct use
        config = {
            # Window generation
            'window_scales': [0.5, 0.75, 1.0],
            'window_stride': 0.1,
            'top_k_windows': 10,
            
            # MS parameters
            'ms_threshold': python_results.get('ms_thresholds', {}).get('32', 0.5),
            'ms_thresholds': python_results.get('ms_thresholds', {}),
            
            # SS parameters
            'ss_n_segments': python_results.get('ss_params', {}).get('n_segments', 100),
            'ss_algorithm': 'slic',
            
            # ED parameters
            'ed_border_ratio': python_results.get('ed_params', {}).get('border_ratio', 0.1),
            'ed_canny_low': python_results.get('ed_params', {}).get('canny_low', 50),
            'ed_canny_high': python_results.get('ed_params', {}).get('canny_high', 150),
            'ed_use_perimeter': True,
            
            # Weights
            'ms_weight': python_results.get('weights', {}).get('ms', 0.4),
            'ss_weight': python_results.get('weights', {}).get('ss', 0.3),
            'ed_weight': python_results.get('weights', {}).get('ed', 0.3),
        }
        
        config_file = os.path.join(self.output_dir, 'config_group_a.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Created config file: {config_file}")
    
    def create_learning_summary(self):
        """Create a summary report of learning results"""
        summary_file = os.path.join(self.output_dir, 'learning_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PARAMETER LEARNING SUMMARY - GROUP A\n")
            f.write("="*70 + "\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  VOC Path: {self.voc_path}\n")
            f.write(f"  Images Used: {len(self.image_files[:10])}\n")
            f.write("\n" + "-"*70 + "\n\n")
            
            f.write("1. Multi-scale Saliency (MS) Thresholds:\n")
            f.write("   (Optimal threshold for each scale)\n")
            for scale, threshold in self.results['ms_thresholds'].items():
                f.write(f"   Scale {scale}: {threshold:.3f}\n")
            f.write("\n")
            
            f.write("2. Edge Density (ED) Parameters:\n")
            ed_params = self.results['ed_params']
            f.write(f"   Border Ratio: {ed_params['border_ratio']:.3f}\n")
            f.write(f"   Canny Low: {ed_params['canny_low']}\n")
            f.write(f"   Canny High: {ed_params['canny_high']}\n")
            f.write(f"   Use Perimeter: {ed_params['use_perimeter']}\n")
            f.write("\n")
            
            f.write("3. Superpixels Straddling (SS) Parameters:\n")
            ss_params = self.results['ss_params']
            f.write(f"   Number of Segments: {ss_params['n_segments']}\n")
            f.write(f"   Algorithm: {ss_params['algorithm']}\n")
            f.write(f"   Compactness: {ss_params['compactness']}\n")
            f.write("\n")
            
            f.write("4. Cue Weights:\n")
            weights = self.results['weights']
            f.write(f"   MS Weight: {weights['ms']:.3f}\n")
            f.write(f"   SS Weight: {weights['ss']:.3f}\n")
            f.write(f"   ED Weight: {weights['ed']:.3f}\n")
            f.write(f"   Sum: {sum(weights.values()):.3f}\n")
            f.write("\n" + "-"*70 + "\n\n")
            
            f.write("Usage Instructions:\n")
            f.write("1. Use 'config_group_a.json' with ObjectnessDetector\n")
            f.write("2. Or load 'learned_parameters.json' directly\n")
            f.write("3. All visualizations are saved in this directory\n")
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Created learning summary: {summary_file}")

# Simple test script
def test_parameter_learning():
    """Test parameter learning with a small dataset"""
    print("Testing Parameter Learning...")
    
    # Create a synthetic dataset if VOC not available
    if not os.path.exists('data/VOCdevkit'):
        print("VOC dataset not found. Creating synthetic test...")
        
        # Create synthetic test directory
        test_dir = 'test_learning_data'
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'Annotations'), exist_ok=True)
        
        # Create 5 test images
        for i in range(5):
            # Create image with object
            img = np.ones((300, 400, 3), dtype=np.uint8) * 100
            
            # Add object at random position
            x1 = np.random.randint(50, 250)
            y1 = np.random.randint(50, 200)
            x2 = x1 + np.random.randint(80, 150)
            y2 = y1 + np.random.randint(80, 120)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)
            
            # Add texture
            for j in range(0, 400, 10):
                cv2.line(img, (j, 0), (j, 300), (150, 150, 150), 1)
            
            # Save image
            img_path = os.path.join(test_dir, 'JPEGImages', f'{i:06d}.jpg')
            cv2.imwrite(img_path, img)
            
            # Create XML annotation
            xml_content = f"""<?xml version="1.0"?>
<annotation>
    <filename>{i:06d}.jpg</filename>
    <size>
        <width>400</width>
        <height>300</height>
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
        
        print(f"Created synthetic dataset with 5 images in {test_dir}")
        dataset_path = test_dir
    else:
        dataset_path = 'data/VOCdevkit/VOC2007'
    
    # Initialize learner
    learner = ParameterLearner(dataset_path, output_dir='learned_params_test')
    
    # Run learning on 5 images
    results = learner.run_complete_learning(n_images=5)
    
    print("\n" + "="*70)
    print("PARAMETER LEARNING TEST COMPLETE")
    print("="*70)
    print(f"Results saved in: learned_params_test/")
    print(f"Check 'learned_parameters.json' for learned parameters")
    print("="*70)

if __name__ == "__main__":
    test_parameter_learning()