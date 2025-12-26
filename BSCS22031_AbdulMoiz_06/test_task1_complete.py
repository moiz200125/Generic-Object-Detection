# test_task1_complete.py
"""
COMPLETE TEST SCRIPT FOR TASK 1 (Multi-scale Saliency)
Run this to test everything from basic to advanced
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add modules to path
sys.path.append('modules')

# Import our implementation
from ms_cue import MultiScaleSaliencyCue
from integral_image import IntegralImage

def test_basic_functionality():
    """Test 1: Basic functionality with synthetic image"""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    # Create synthetic test image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Create a clear object (red circle) on textured background
    # Background texture (low frequency)
    for i in range(0, 256, 15):
        cv2.line(img, (0, i), (256, i), (100, 100, 100), 2)
        cv2.line(img, (i, 0), (i, 256), (100, 100, 100), 2)
    
    # Object (different from background)
    cv2.circle(img, (128, 128), 60, (0, 0, 255), -1)  # Red circle
    cv2.circle(img, (128, 128), 30, (0, 255, 255), -1)  # Yellow inner circle
    
    # Add some high-frequency noise to object
    noise = np.random.randint(-20, 20, (120, 120, 3), dtype=np.int16)
    img[68:188, 68:188] = np.clip(img[68:188, 68:188].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Test MS Cue
    print("Initializing MS Cue...")
    ms = MultiScaleSaliencyCue(img)
    
    print("Computing saliency maps...")
    ms.compute_saliency_maps()
    
    # Test different windows
    test_windows = [
        (100, 100, 156, 156),  # On object (should have high score)
        (20, 20, 60, 60),      # On background (low score)
        (0, 0, 256, 256),      # Full image
        (200, 200, 240, 240),  # Corner (background)
    ]
    
    print("\nWindow Scores:")
    for i, window in enumerate(test_windows):
        score = ms.get_score(window)
        x1, y1, x2, y2 = window
        print(f"  Window {i+1} [{x1},{y1},{x2},{y2}]: {score:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Draw windows
    img_windows = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    for i, window in enumerate(test_windows):
        x1, y1, x2, y2 = window
        cv2.rectangle(img_windows, (x1, y1), (x2, y2), colors[i], 2)
        score = ms.get_score(window)
        cv2.putText(img_windows, f"{score:.3f}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
    
    axes[0, 1].imshow(cv2.cvtColor(img_windows, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Test Windows with Scores")
    axes[0, 1].axis('off')
    
    # Show saliency maps for different scales
    scales_to_show = [16, 32, 64]
    for idx, scale in enumerate(scales_to_show):
        if scale in ms.saliency_maps:
            axes[0, 2+idx].imshow(ms.saliency_maps[scale], cmap='hot')
            axes[0, 2+idx].set_title(f"Saliency (Scale {scale})")
            axes[0, 2+idx].axis('off')
    
    # Show binary maps
    for idx, scale in enumerate(scales_to_show):
        if scale in ms.binary_maps:
            axes[1, idx].imshow(ms.binary_maps[scale], cmap='gray')
            axes[1, idx].set_title(f"Binary (Scale {scale})")
            axes[1, idx].axis('off')
    
    # Combined saliency
    combined = np.mean(list(ms.saliency_maps.values()), axis=0)
    axes[1, 2].imshow(combined, cmap='hot')
    axes[1, 2].set_title("Combined Saliency")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/test1_basic.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to: results/test1_basic.png")
    
    plt.show()
    return ms

def test_with_voc_image(voc_path):
    """Test 2: With actual VOC image"""
    print("\n" + "="*60)
    print("TEST 2: Real VOC Image")
    print("="*60)
    
    # Find a VOC image
    jpeg_path = os.path.join(voc_path, "JPEGImages")
    image_files = [f for f in os.listdir(jpeg_path) if f.endswith('.jpg')]
    
    if not image_files:
        print("No VOC images found. Please download dataset first.")
        return None
    
    # Load first image
    sample_image = image_files[0]
    img_path = os.path.join(jpeg_path, sample_image)
    
    print(f"Loading: {sample_image}")
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to load {img_path}")
        return None
    
    print(f"Image shape: {img.shape}")
    
    # Resize if too large (for faster testing)
    if img.shape[0] > 500 or img.shape[1] > 500:
        scale = 500 / max(img.shape[:2])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size)
        print(f"Resized to: {img.shape}")
    
    # Run MS cue
    ms = MultiScaleSaliencyCue(img)
    ms.compute_saliency_maps()
    
    # Generate sliding windows
    height, width = img.shape[:2]
    windows = []
    stride = 80
    window_size = 120
    
    for y in range(0, height - window_size, stride):
        for x in range(0, width - window_size, stride):
            windows.append((x, y, x + window_size, y + window_size))
    
    print(f"Generated {len(windows)} windows")
    
    # Score all windows
    scores = []
    for window in windows[:100]:  # Limit to 100 for speed
        score = ms.get_score(window)
        scores.append(score)
    
    # Find best window
    if scores:
        best_idx = np.argmax(scores)
        best_window = windows[best_idx]
        best_score = scores[best_idx]
        
        print(f"\nBest window: {best_window}")
        print(f"Best score: {best_score:.4f}")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original with best window
        img_vis = img.copy()
        x1, y1, x2, y2 = best_window
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_vis, f"Score: {best_score:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        axes[0].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Best Window (Score: {best_score:.3f})")
        axes[0].axis('off')
        
        # Combined saliency
        combined = np.mean(list(ms.saliency_maps.values()), axis=0)
        axes[1].imshow(combined, cmap='hot')
        axes[1].set_title("Combined Saliency Map")
        axes[1].axis('off')
        
        # Heatmap of window scores
        score_map = np.zeros((height, width))
        for (x1, y1, x2, y2), score in zip(windows[:100], scores):
            score_map[y1:y2, x1:x2] = score
        
        axes[2].imshow(score_map, cmap='hot')
        axes[2].set_title("Window Score Heatmap")
        axes[2].axis('off')
        
        plt.colorbar(axes[2].imshow(score_map, cmap='hot'), ax=axes[2])
        
        plt.tight_layout()
        plt.savefig('results/test2_voc.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to: results/test2_voc.png")
        
        plt.show()
    
    return ms

def test_performance():
    """Test 3: Performance testing"""
    print("\n" + "="*60)
    print("TEST 3: Performance Testing")
    print("="*60)
    
    # Create different sized images
    sizes = [(128, 128), (256, 256), (512, 512)]
    
    results = []
    for h, w in sizes:
        # Create random image
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        print(f"\nTesting {w}x{h} image...")
        
        # Time MS computation
        import time
        start = time.time()
        
        ms = MultiScaleSaliencyCue(img)
        ms.compute_saliency_maps()
        
        end = time.time()
        elapsed = end - start
        
        # Time window scoring
        windows = [(0, 0, 100, 100)]  # Single window for testing
        score_start = time.time()
        
        for window in windows:
            score = ms.get_score(window)
        
        score_end = time.time()
        score_time = score_end - score_start
        
        results.append({
            'size': f"{w}x{h}",
            'computation_time': elapsed,
            'scoring_time': score_time,
            'fps': 1.0 / elapsed if elapsed > 0 else 0
        })
        
        print(f"  Computation: {elapsed:.2f}s")
        print(f"  Scoring (per window): {score_time*1000:.2f}ms")
        print(f"  FPS: {1.0/elapsed:.2f}" if elapsed > 0 else "  FPS: N/A")
    
    # Print summary
    print("\n" + "-"*50)
    print("PERFORMANCE SUMMARY:")
    print("-"*50)
    for r in results:
        print(f"{r['size']}: Compute={r['computation_time']:.2f}s, "
              f"Score={r['scoring_time']*1000:.2f}ms, FPS={r['fps']:.2f}")
    
    return results

def test_integral_image_correctness():
    """Test 4: Verify Integral Image calculations"""
    print("\n" + "="*60)
    print("TEST 4: Integral Image Verification")
    print("="*60)
    
    # Create random matrix
    np.random.seed(42)
    test_matrix = np.random.rand(100, 100)
    
    # Create integral image
    ii = IntegralImage(test_matrix)
    
    # Test random windows
    n_tests = 10
    errors = []
    
    for i in range(n_tests):
        # Random window
        x1 = np.random.randint(0, 90)
        y1 = np.random.randint(0, 90)
        x2 = np.random.randint(x1 + 10, 100)
        y2 = np.random.randint(y1 + 10, 100)
        
        # Manual sum
        manual_sum = np.sum(test_matrix[y1:y2, x1:x2])
        
        # Integral image sum
        ii_sum = ii.rectangle_sum(x1, y1, x2, y2)
        
        # Calculate error
        error = abs(manual_sum - ii_sum)
        errors.append(error)
        
        print(f"Test {i+1}: Window [{x1},{y1},{x2},{y2}]")
        print(f"  Manual: {manual_sum:.6f}")
        print(f"  Integral: {ii_sum:.6f}")
        print(f"  Error: {error:.6f}")
        print(f"  {'✓ PASS' if error < 1e-6 else '✗ FAIL'}")
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"\nAverage error: {avg_error:.6f}")
    print(f"Maximum error: {max_error:.6f}")
    
    if max_error < 1e-6:
        print("✓ All Integral Image tests passed!")
    else:
        print("✗ Some Integral Image tests failed.")
    
    return max_error < 1e-6

def run_all_tests():
    """Run all tests"""
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("="*70)
    print("COMPLETE TASK 1 TEST SUITE")
    print("="*70)
    
    # Test 1: Basic functionality
    ms1 = test_basic_functionality()
    
    # Test 2: With VOC image (if available)
    voc_path = "data/VOCdevkit/VOC2007"
    if os.path.exists(voc_path):
        ms2 = test_with_voc_image(voc_path)
    else:
        print("\nSkipping VOC test - dataset not found.")
        print("Download dataset using instructions in README.")
    
    # Test 3: Performance
    perf_results = test_performance()
    
    # Test 4: Integral Image correctness
    ii_correct = test_integral_image_correctness()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("1. Basic Functionality: ✓ COMPLETE")
    print("2. VOC Image Test: " + ("✓ COMPLETE" if os.path.exists(voc_path) else "⏸ SKIPPED"))
    print("3. Performance Tests: ✓ COMPLETE")
    print("4. Integral Image: " + ("✓ CORRECT" if ii_correct else "✗ ISSUES"))
    print("\nAll visualizations saved to 'results/' folder")
    print("="*70)

if __name__ == "__main__":
    run_all_tests()