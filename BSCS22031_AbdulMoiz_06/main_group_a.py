# main_group_a.py
"""
Main execution script for Group A (MS + SS + ED)
"""
import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import json

# Add modules path
sys.path.append('modules')

try:
    from modules.objectness_detector import ObjectnessDetector
    print("✓ ObjectnessDetector imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating ObjectnessDetector inline...")
    
    # Define inline if import fails
    # ... (copy the ObjectnessDetector class from above)

def create_config_file():
    """Create default configuration file"""
    config = {
        # Window generation
        "window_scales": [0.5, 0.75, 1.0],
        "window_stride": 0.1,
        "top_k_windows": 10,
        
        # MS parameters
        "ms_threshold": 0.2,
        
        # SS parameters
        "ss_n_segments": 100,
        "ss_algorithm": "slic",
        
        # ED parameters
        "ed_border_ratio": 0.1,
        "ed_canny_low": 50,
        "ed_canny_high": 150,
        "ed_use_perimeter": True,
        
        # Cue weights
        "ms_weight": 0.4,
        "ss_weight": 0.3,
        "ed_weight": 0.3,
    }
    
    with open('config_group_a.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created config_group_a.json")
    return config

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Group A Objectness Detector (MS + SS + ED)'
    )
    parser.add_argument('--input', required=True, help='Input image or folder')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--config', default='config_group_a.json', help='Config file')
    parser.add_argument('--limit', type=int, default=3, help='Max images to process')
    parser.add_argument('--create-config', action='store_true', help='Create default config')
    
    args = parser.parse_args()
    
    # Create config file if requested
    if args.create_config:
        create_config_file()
        return
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'text'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'visualizations'), exist_ok=True)
    
    # Get image files
    if os.path.isfile(args.input):
        image_files = [Path(args.input)]
    else:
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(Path(args.input).glob(f'*{ext}')))
            image_files.extend(list(Path(args.input).glob(f'*{ext.upper()}')))
    
    image_files = image_files[:args.limit]
    print(f"Found {len(image_files)} images to process")
    
    # Initialize detector
    detector = ObjectnessDetector(args.config)
    
    # Process each image
    processed = 0
    
    for img_path in image_files:
        try:
            print(f"\n{'='*70}")
            print(f"PROCESSING: {img_path.name}")
            print(f"{'='*70}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Resize if too large
            h, w = image.shape[:2]
            if h > 800 or w > 800:
                scale = 800 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
                print(f"Resized from {w}x{h} to {new_w}x{new_h}")
            
            # Process image
            top_windows, cues = detector.process_image(image)
            
            if not top_windows:
                print("Warning: No windows scored")
                continue
            
            # Save results
            text_dir = os.path.join(args.output, 'text')
            detector.save_results(img_path.stem, top_windows, text_dir)
            
            # Create visualization
            vis_path = os.path.join(args.output, 'visualizations', f"{img_path.stem}_result.jpg")
            detector.visualize_results(image, top_windows, cues, vis_path)
            
            # Print summary
            print(f"\n✓ Processing complete for {img_path.name}")
            print(f"  Top window: {top_windows[0][0]}")
            print(f"  Total score: {top_windows[0][1]:.4f}")
            print(f"  MS: {top_windows[0][2]:.4f}, SS: {top_windows[0][3]:.4f}, ED: {top_windows[0][4]:.4f}")
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Images processed: {processed}/{len(image_files)}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"  - Text results: {args.output}/text/")
    print(f"  - Visualizations: {args.output}/visualizations/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()