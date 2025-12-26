# main.py
"""
MAIN EXECUTION SCRIPT FOR OBJECTNESS DETECTOR
Usage: python main.py --input <folder> --output <folder>
"""
import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add current directory to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from modules directory
try:
    from modules.ms_cue import MultiScaleSaliencyCue
    from modules.ss_cue import SuperpixelStraddlingCue
    from modules.ed_cue import EdgeDensityCue
    from modules.integral_image import IntegralImage
    print("✓ Successfully imported modules")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Trying alternative import...")
    # Try direct import
    import importlib.util
    
    # Import integral_image
    spec = importlib.util.spec_from_file_location(
        "integral_image", 
        os.path.join("modules", "integral_image.py")
    )
    integral_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integral_module)
    IntegralImage = integral_module.IntegralImage
    
    # Import ms_cue
    spec = importlib.util.spec_from_file_location(
        "ms_cue", 
        os.path.join("modules", "ms_cue.py")
    )
    ms_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms_module)
    MultiScaleSaliencyCue = ms_module.MultiScaleSaliencyCue

    # Import ed_cue
    spec = importlib.util.spec_from_file_location(
        "ed_cue", 
        os.path.join("modules", "ed_cue.py")
    )
    ed_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ed_module)
    EdgeDensityCue = ed_module.EdgeDensityCue
    
    print("✓ Modules loaded via alternative method")

class ObjectnessDetector:
    """Main class for Objectness Detection"""
    
    def __init__(self, config_path=None):
        """
        Initialize detector with configuration
        """
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                # Convert scale keys to int
                if 'ms_thresholds' in self.config:
                    self.config['ms_thresholds'] = {int(k): v for k, v in self.config['ms_thresholds'].items()}
        
        # Initialize cues
        self.ms_cue = None
        
        # Default parameters
        self.default_params = {
            'window_scales': [0.5, 0.75, 1.0, 1.5, 2.0],
            'window_stride': 0.1,  # 10% of image size
            'top_k_windows': 10,
             'ms_threshold': 0.2,
            # SS parameters
            'ss_algorithm': 'slic',
            'ss_n_segments': 100,
            'ss_compactness': 10,
            'ss_weight': 0.3,  # Weight for SS in final score
            'ms_weight': 0.4,  # Weight for MS
            'ed_weight': 0.3,  
        }
        
        # Merge with config
        self.params = {**self.default_params, **self.config}
        # Initialize cues
        self.ms_cue = None
        self.ss_cue = None
        self.ed_cue = None
        
    
        print("Objectness Detector Initialized")
    
    def generate_windows(self, image_shape):
        """
        Generate sliding windows at multiple scales
        """
        height, width = image_shape[:2]
        windows = []
        
        for scale in self.params['window_scales']:
            # Calculate window size for this scale
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
        print(f"\nProcessing image: {image.shape}")
    
        # Initialize cues
        self.ms_cue = MultiScaleSaliencyCue(image)
        self.ss_cue = SuperpixelStraddlingCue(
            image,
            algorithm=self.params['ss_algorithm'],
            n_segments=self.params['ss_n_segments'],
            compactness=self.params['ss_compactness']
        )
        self.ed_cue = EdgeDensityCue(
            image,
            border_ratio=0.1  # Default parameter
        )
        
        # Set MS threshold if available
        if 'ms_threshold' in self.params:
            thresholds = {scale: self.params['ms_threshold'] for scale in self.ms_cue.scales}
            self.ms_cue.set_thresholds(thresholds)
        
        # Compute features
        print("Computing Multi-scale Saliency...")
        self.ms_cue.compute_saliency_maps()
        
        print("Computing Superpixels...")
        self.ss_cue.compute_superpixels()
        
        # Generate windows
        windows = self.generate_windows(image.shape)
        
        # Score windows
        print(f"Scoring {len(windows)} windows...")
        window_scores = []
        
        for i, window in enumerate(windows):
            # Get MS score
            ms_score = self.ms_cue.get_score(window)
            
            # Get SS score
            ss_score = self.ss_cue.get_score(window)

            # Get ED score
            ed_score = self.ed_cue.get_score(window)
            
            # Combine scores (simple weighted sum for now)
            # You'll add ED score later
            total_score = (
                self.params['ms_weight'] * ms_score +
                self.params['ss_weight'] * ss_score +
                self.params['ed_weight'] * ed_score
            )
            
            window_scores.append((window, total_score, ms_score, ss_score, ed_score))
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"  Scored {i + 1}/{len(windows)} windows...")
        
        # Sort by total score
        window_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply Non-Maximum Suppression (NMS)
        print("Applying Non-Maximum Suppression...")
        nms_windows = self.non_max_suppression(window_scores, iou_threshold=0.3)
        
        # Return top K windows
        top_k = min(self.params['top_k_windows'], len(nms_windows))
        return nms_windows[:top_k]
    
    def non_max_suppression(self, window_scores, iou_threshold=0.3):
        """
        Apply Non-Maximum Suppression to filter overlapping windows
        """
        if not window_scores:
            return []
            
        # Extract boxes and scores
        boxes = []
        for item in window_scores:
            boxes.append(item[0])  # window (x1, y1, x2, y2)
            
        keep_indices = []
        indices = list(range(len(boxes)))
        
        while indices:
            # Pick the box with highest score (first in sorted list)
            current = indices.pop(0)
            keep_indices.append(current)
            
            # Compare with remaining boxes
            remaining_indices = []
            for idx in indices:
                # Calculate IoU
                box1 = boxes[current]
                box2 = boxes[idx]
                
                # Intersection
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                
                if inter_area == 0:
                    remaining_indices.append(idx)
                    continue
                
                # Union
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = area1 + area2 - inter_area
                
                iou = inter_area / union_area
                
                # If IoU < threshold, keep it (not too overlapping)
                if iou < iou_threshold:
                    remaining_indices.append(idx)
            
            indices = remaining_indices
            
        # Return filtered scores
        return [window_scores[i] for i in keep_indices]
    
    def visualize_results(self, image, top_windows, save_path=None):
        """Update visualization to show all cues"""
        # Create multiple visualizations
        
        # 1. Main visualization with top windows
        vis_main = image.copy()
        colors = [(0, 255, 0), (0, 200, 100), (255, 255, 0), (255, 165, 0), (255, 0, 0)]
        
        for i, (window, total_score, ms_score, ss_score, ed_score) in enumerate(top_windows):
            x1, y1, x2, y2 = window
            color = colors[i % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(vis_main, (x1, y1), (x2, y2), color, 2)
            
            # Draw scores
            label = f"{i+1}: T={total_score:.2f}, M={ms_score:.2f}, S={ss_score:.2f}, E={ed_score:.2f}"
            cv2.putText(vis_main, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(vis_main, "Objectness (MS + SS)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 2. Superpixel visualization for top window
        if top_windows:
            top_window = top_windows[0][0]
            vis_superpixels = self.ss_cue.visualize_superpixels(top_window)
            
            # Combine visualizations side by side
            h, w = image.shape[:2]
            combined = np.zeros((h, w*2, 3), dtype=np.uint8)
            combined[:, :w] = vis_main
            combined[:, w:] = vis_superpixels
            
            # Add labels
            cv2.putText(combined, "Detection Results", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Superpixels (Green=Top Window)", (w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            vis = combined
        else:
            vis = vis_main
        
        if save_path:
            cv2.imwrite(save_path, vis)
            print(f"✓ Saved visualization to: {save_path}")
        
        return vis
    
    def save_results(self, image_name, top_windows, output_dir):
        """
        Save results to text file
        """
        # Create output file path
        result_file = os.path.join(output_dir, f"{image_name}_results.txt")
        
        with open(result_file, 'w') as f:
            f.write(f"Objectness Detection Results for: {image_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of top windows: {len(top_windows)}\n")
            f.write("-" * 50 + "\n\n")
            
            for i, (window, total_score, ms_score, ss_score, ed_score) in enumerate(top_windows):
                x1, y1, x2, y2 = window
                f.write(f"Window {i+1}:\n")
                f.write(f"  Coordinates: [{x1}, {y1}, {x2}, {y2}]\n")
                f.write(f"  Dimensions: {x2-x1} x {y2-y1}\n")
                f.write(f"  Total Score: {total_score:.6f}\n")
                f.write(f"  MS Score:    {ms_score:.6f}\n")
                f.write(f"  SS Score:    {ss_score:.6f}\n")
                f.write(f"  ED Score:    {ed_score:.6f}\n")
                f.write("-" * 30 + "\n")
        
        print(f"✓ Saved results to: {result_file}")

def main():
    """Main execution function"""
    
    # Parse command line arguments
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Objectness Detector - Task 1 (MS Cue)')
    
    # Input argument (allow both --input and --input_folder)
    parser.add_argument('--input', help='Input folder containing images')
    parser.add_argument('--input_folder', help='Input folder containing images (Assignment spec)')
    
    # Output argument (allow both --output and --output_folder)
    parser.add_argument('--output', help='Output folder for results')
    parser.add_argument('--output_folder', help='Output folder for results (Assignment spec)')
    
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images to process')
    
    args = parser.parse_args()
    
    # Normalize arguments
    if args.input_folder and not args.input:
        args.input = args.input_folder
    elif not args.input and not args.input_folder:
        parser.error("One of --input or --input_folder is required")
        
    if args.output_folder and not args.output:
        args.output = args.output_folder
    elif not args.output and not args.output_folder:
        parser.error("One of --output or --output_folder is required")
    
    # Validate arguments
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'text'), exist_ok=True)
    
    # Initialize detector
    detector = ObjectnessDetector(args.config)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(args.input).glob(f'*{ext}')))
        image_files.extend(list(Path(args.input).glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images in '{args.input}'")
    
    # Limit if specified
    if args.limit > 0:
        image_files = image_files[:args.limit]
        print(f"Limiting to {len(image_files)} images")
    
    # Process each image
    processed_count = 0
    
    for img_path in image_files:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {img_path.name}")
            print(f"{'='*60}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Resize if too large (for faster processing)
            max_dim = 500
            h, w = image.shape[:2]
            if h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
                print(f"Resized from {w}x{h} to {new_w}x{new_h}")
            
            # Process image
            top_windows = detector.process_image(image)
            
            # Visualize results
            vis_path = os.path.join(args.output, 'images', f"{img_path.stem}_result.jpg")
            detector.visualize_results(image, top_windows, vis_path)
            
            # Save text results
            text_dir = os.path.join(args.output, 'text')
            detector.save_results(img_path.stem, top_windows, text_dir)
            
            processed_count += 1
            print(f"✓ Completed: {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {processed_count}/{len(image_files)}")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"  - Visualizations: {args.output}/images/")
    print(f"  - Text results: {args.output}/text/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()