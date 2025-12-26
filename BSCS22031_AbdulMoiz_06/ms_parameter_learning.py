# ms_parameter_learning.py
import numpy as np
import cv2
import os
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from modules.ms_cue import MultiScaleSaliencyCue

class MSParameterLearner:
    """
    Learn optimal thresholds for MS cue using PASCAL VOC dataset
    """
    
    def __init__(self, voc_dataset_path):
        """
        Initialize with path to PASCAL VOC dataset
        
        Args:
            voc_dataset_path: Path to VOC dataset (should have JPEGImages/ and Annotations/)
        """
        self.voc_path = voc_dataset_path
        self.images_path = os.path.join(voc_dataset_path, "JPEGImages")
        self.annotations_path = os.path.join(voc_dataset_path, "Annotations")
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.images_path) 
                           if f.endswith('.jpg')]
        
        print(f"Found {len(self.image_files)} images in dataset")
        
        # Ground truth cache
        self.gt_cache = {}
    
    def _load_ground_truth(self, image_name):
        """
        Load ground truth bounding boxes from XML annotation
        
        Args:
            image_name: Name of image file (without extension)
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        if image_name in self.gt_cache:
            return self.gt_cache[image_name]
        
        xml_path = os.path.join(self.annotations_path, f"{image_name}.xml")
        
        if not os.path.exists(xml_path):
            return []
        
        # Parse XML to get bounding boxes
        # Simplified parsing - you might want to use xml.etree.ElementTree
        boxes = []
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                x1 = int(float(bndbox.find('xmin').text))
                y1 = int(float(bndbox.find('ymin').text))
                x2 = int(float(bndbox.find('xmax').text))
                y2 = int(float(bndbox.find('ymax').text))
                boxes.append((x1, y1, x2, y2))
        except:
            print(f"Warning: Could not parse {xml_path}")
        
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
        inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def find_best_thresholds(self, n_images=50, n_windows_per_image=100):
        """
        Find optimal thresholds for each scale that maximize IoU
        
        Args:
            n_images: Number of images to use for learning
            n_windows_per_image: Number of windows to sample per image
            
        Returns:
            Dictionary of optimal thresholds for each scale
        """
        print("Learning MS thresholds...")
        print("-" * 50)
        
        # Limit number of images for faster learning
        image_subset = self.image_files[:min(n_images, len(self.image_files))]
        
        # Define threshold range to search
        threshold_range = np.arange(0.1, 0.95, 0.05)
        
        # Initialize results dictionary
        scale_results = {}
        scales = [16, 24, 32, 48, 64]
        
        for scale in scales:
            print(f"\nLearning threshold for scale {scale}:")
            
            best_threshold = 0.5
            best_avg_iou = 0
            
            # Try each threshold
            for threshold in threshold_range:
                ious = []
                
                for img_file in image_subset:
                    # Load image
                    img_path = os.path.join(self.images_path, img_file)
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        continue
                    
                    # Get image name without extension
                    img_name = os.path.splitext(img_file)[0]
                    
                    # Load ground truth
                    gt_boxes = self._load_ground_truth(img_name)
                    
                    if not gt_boxes:
                        continue
                    
                    # Initialize MS cue with this image
                    ms = MultiScaleSaliencyCue(image)
                    ms.compute_saliency_maps()
                    
                    # Set current threshold for all scales (we're testing one scale at a time)
                    test_thresholds = {s: 0.5 for s in scales}
                    test_thresholds[scale] = threshold
                    ms.set_thresholds(test_thresholds)
                    
                    # Generate random windows
                    height, width = image.shape[:2]
                    
                    # Sample windows
                    window_ious = []
                    for _ in range(n_windows_per_image):
                        # Generate random window
                        w = np.random.randint(30, min(200, width//2))
                        h = np.random.randint(30, min(200, height//2))
                        x = np.random.randint(0, width - w)
                        y = np.random.randint(0, height - h)
                        
                        window = (x, y, x + w, y + h)
                        
                        # Calculate MS score
                        ms_score = ms.get_score(window)
                        
                        # Find best IoU with any ground truth box
                        best_iou = 0
                        for gt_box in gt_boxes:
                            iou = self.calculate_iou(window, gt_box)
                            best_iou = max(best_iou, iou)
                        
                        window_ious.append((ms_score, best_iou))
                    
                    # For this threshold, we want to see if high MS scores correlate with high IoU
                    # We'll use average IoU of top 10% windows as metric
                    if window_ious:
                        scores, ious_list = zip(*window_ious)
                        # Sort by MS score
                        sorted_indices = np.argsort(scores)[::-1]  # Descending
                        top_count = max(1, len(sorted_indices) // 10)  # Top 10%
                        top_avg_iou = np.mean([ious_list[i] for i in sorted_indices[:top_count]])
                        ious.append(top_avg_iou)
                
                # Calculate average IoU across all images for this threshold
                if ious:
                    avg_iou = np.mean(ious)
                    
                    print(f"  Threshold {threshold:.2f}: Avg IoU = {avg_iou:.4f}")
                    
                    if avg_iou > best_avg_iou:
                        best_avg_iou = avg_iou
                        best_threshold = threshold
            
            scale_results[scale] = best_threshold
            print(f"  → Best threshold for scale {scale}: {best_threshold:.2f} (IoU: {best_avg_iou:.4f})")
        
        print("\n" + "="*50)
        print("LEARNING COMPLETE!")
        print("Optimal thresholds:")
        for scale, threshold in scale_results.items():
            print(f"  Scale {scale}: {threshold:.3f}")
        print("="*50)
        
        return scale_results
    
    def visualize_threshold_effect(self, image_file):
        """
        Visualize how different thresholds affect saliency detection
        
        Args:
            image_file: Name of image file to visualize
        """
        img_path = os.path.join(self.images_path, image_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Could not load image: {image_file}")
            return
        
        # Get image name without extension
        img_name = os.path.splitext(image_file)[0]
        gt_boxes = self._load_ground_truth(img_name)
        
        # Initialize MS cue
        ms = MultiScaleSaliencyCue(image)
        ms.compute_saliency_maps()
        
        # Test different thresholds
        thresholds = [0.3, 0.5, 0.7]
        
        fig, axes = plt.subplots(2, len(thresholds) + 1, figsize=(15, 8))
        
        # Show original image with ground truth
        img_with_gt = image.copy()
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_with_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[0, 0].imshow(cv2.cvtColor(img_with_gt, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original with GT")
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(np.mean(list(ms.saliency_maps.values()), axis=0), cmap='hot')
        axes[1, 0].set_title("Saliency Map (average)")
        axes[1, 0].axis('off')
        
        # Show binary maps for different thresholds
        for i, threshold in enumerate(thresholds):
            # Set threshold
            ms.set_thresholds({s: threshold for s in ms.scales})
            
            # Get combined binary map (average across scales)
            binary_maps = []
            for scale in ms.scales:
                binary_maps.append(ms.binary_maps[scale])
            
            combined_binary = np.mean(binary_maps, axis=0)
            
            # Find contours in binary map
            binary_vis = (combined_binary > 0.5).astype(np.uint8) * 255
            
            # Draw on image
            img_with_detections = image.copy()
            contours, _ = cv2.findContours(binary_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img_with_detections, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            axes[0, i+1].imshow(cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB))
            axes[0, i+1].set_title(f"Detections (thresh={threshold})")
            axes[0, i+1].axis('off')
            
            axes[1, i+1].imshow(combined_binary, cmap='gray')
            axes[1, i+1].set_title(f"Binary Map (thresh={threshold})")
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'ms_threshold_effect_{img_name}.png', dpi=150, bbox_inches='tight')
        plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize learner with path to your VOC dataset
    # Download from: https://www.kaggle.com/datasets/zaraks/pascal-voc-2007
    learner = MSParameterLearner("path/to/your/VOC2007")
    
    # Find optimal thresholds
    optimal_thresholds = learner.find_best_thresholds(n_images=20, n_windows_per_image=50)
    
    # Save thresholds to file
    import json
    with open('ms_optimal_thresholds.json', 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    print("\nOptimal thresholds saved to 'ms_optimal_thresholds.json'")
    
    # Visualize on a sample image
    if learner.image_files:
        sample_image = learner.image_files[0]
        learner.visualize_threshold_effect(sample_image)