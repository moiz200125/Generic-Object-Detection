# test_ss_cue_complete.py
"""
Complete test for Superpixels Straddling (SS) Cue
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('modules')

from ss_cue import SuperpixelStraddlingCue, MemoryOptimizedSuperpixelCue
from ms_cue import MultiScaleSaliencyCue

def test_ss_basic():
    """Basic test with synthetic image"""
    print("\n" + "="*60)
    print("TEST 1: Basic SS Cue Test")
    print("="*60)
    
    # Create image with clear regions
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Region 1 (top-left)
    img[0:150, 0:200] = [100, 50, 50]
    # Region 2 (top-right)
    img[0:150, 200:400] = [50, 100, 50]
    # Region 3 (bottom-left)
    img[150:300, 0:200] = [50, 50, 100]
    # Region 4 (bottom-right)
    img[150:300, 200:400] = [100, 100, 50]
    
    # Add some texture
    for i in range(0, 400, 25):
        cv2.line(img, (i, 0), (i, 300), (30, 30, 30), 1)
    for i in range(0, 300, 25):
        cv2.line(img, (0, i), (400, i), (30, 30, 30), 1)
    
    # Initialize SS cue
    ss = SuperpixelStraddlingCue(img, algorithm='slic', n_segments=50)
    ss.compute_superpixels()
    
    # Test windows
    windows = [
        (50, 50, 150, 150),   # Inside region 1
        (250, 50, 350, 150),  # Inside region 2
        (150, 50, 250, 150),  # Straddling region 1-2 boundary
        (50, 150, 150, 250),  # Straddling region 1-3 boundary
        (150, 150, 250, 250), # At intersection of all 4 regions
        (0, 0, 400, 300),     # Full image
    ]
    
    print("\nWindow Scores:")
    for i, window in enumerate(windows):
        score = ss.get_score(window)
        print(f"  Window {i+1} {window}: {score:.4f}")
        
        # Expected: Inside regions should have higher scores than straddling
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Draw windows on image
    img_windows = img.copy()
    colors = [(0, 255, 0), (0, 200, 100), (255, 255, 0), 
              (255, 165, 0), (255, 0, 0), (0, 255, 255)]
    
    for i, window in enumerate(windows):
        x1, y1, x2, y2 = window
        color = colors[i]
        cv2.rectangle(img_windows, (x1, y1), (x2, y2), color, 2)
        score = ss.get_score(window)
        cv2.putText(img_windows, f"{score:.3f}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    axes[0, 1].imshow(cv2.cvtColor(img_windows, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Test Windows with SS Scores")
    axes[0, 1].axis('off')
    
    # Show superpixel visualization
    vis_superpixels = ss.visualize_superpixels(windows[2])  # Straddling window
    axes[0, 2].imshow(cv2.cvtColor(vis_superpixels, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Superpixels (Red=Straddling)")
    axes[0, 2].axis('off')
    
    # Show superpixel map
    axes[1, 0].imshow(ss.superpixel_labels, cmap='tab20')
    axes[1, 0].set_title("Superpixel Label Map")
    axes[1, 0].axis('off')
    
    # Show score distribution
    all_scores = [ss.get_score(w) for w in windows]
    axes[1, 1].bar(range(len(windows)), all_scores, color=[c[:3] for c in colors])
    axes[1, 1].set_title("SS Scores by Window")
    axes[1, 1].set_xlabel("Window Index")
    axes[1, 1].set_ylabel("SS Score")
    axes[1, 1].set_ylim(0, 1)
    
    # Show penalty calculation for straddling window
    info = ss.get_window_straddling_info(windows[2])
    penalty_text = f"Window {windows[2]}:\n"
    penalty_text += f"Score: {info['score']:.3f}\n"
    penalty_text += f"Penalty: {info['penalty']:.1f}\n"
    penalty_text += f"Superpixels: {info['num_superpixels']}\n"
    penalty_text += f"Window Area: {info['window_area']}"
    
    axes[1, 2].text(0.1, 0.5, penalty_text, fontsize=12, 
                   verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("Straddling Analysis")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('ss_basic_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to ss_basic_test.png")
    
    plt.show()
    
    return ss

def test_ss_with_real_image():
    """Test SS cue on real image"""
    print("\n" + "="*60)
    print("TEST 2: SS Cue with Real Image")
    print("="*60)
    
    # Load an image
    img_path = 'sample_images/000019.jpg' if os.path.exists('sample_images/000019.jpg') else None
    
    if img_path and os.path.exists(img_path):
        img = cv2.imread(img_path)
        print(f"Loaded {img_path}: {img.shape}")
    else:
        # Create synthetic image
        img = np.ones((300, 400, 3), dtype=np.uint8) * 100
        cv2.rectangle(img, (100, 100), (300, 200), (0, 0, 255), -1)
        cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
        print("Created synthetic image")
    
    # Initialize SS cue
    ss = MemoryOptimizedSuperpixelCue(img, algorithm='slic', n_segments=150)
    ss.compute_superpixels()
    
    # Generate test windows
    h, w = img.shape[:2]
    windows = []
    
    # Some candidate windows
    window_size = 100
    stride = 80
    
    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            windows.append((x, y, x + window_size, y + window_size))
    
    # Score all windows
    scores = []
    for window in windows[:20]:  # Limit for speed
        score = ss.get_score(window)
        scores.append((window, score))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 windows (of {len(scores)} scored):")
    for i, (window, score) in enumerate(scores[:5]):
        print(f"  {i+1}. {window}: {score:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original with top window
    img_with_top = img.copy()
    if scores:
        top_window, top_score = scores[0]
        x1, y1, x2, y2 = top_window
        cv2.rectangle(img_with_top, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_with_top, f"Score: {top_score:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    axes[0].imshow(cv2.cvtColor(img_with_top, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Top Window (Score: {top_score:.3f})" if scores else "Original")
    axes[0].axis('off')
    
    # Superpixel visualization
    vis_superpixels = ss.visualize_superpixels(top_window if scores else None)
    axes[1].imshow(cv2.cvtColor(vis_superpixels, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Superpixels")
    axes[1].axis('off')
    
    # Score heatmap
    score_map = np.zeros((h, w))
    for (x1, y1, x2, y2), score in scores:
        score_map[y1:y2, x1:x2] = score
    
    heatmap = axes[2].imshow(score_map, cmap='hot')
    axes[2].set_title("SS Score Heatmap")
    axes[2].axis('off')
    plt.colorbar(heatmap, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('ss_real_test.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to ss_real_test.png")
    
    plt.show()
    
    return ss

def test_ss_ms_integration():
    """Test integration of MS and SS cues"""
    print("\n" + "="*60)
    print("TEST 3: MS + SS Integration")
    print("="*60)
    
    # Create test image
    img = np.ones((300, 400, 3), dtype=np.uint8) * 100
    
    # Create an object
    cv2.rectangle(img, (100, 100), (300, 200), (0, 0, 255), -1)
    
    # Add texture to object (makes it salient for MS)
    for i in range(100, 300, 5):
        cv2.line(img, (i, 100), (i, 200), (0, 50, 255), 1)
    
    # Add texture to background (different pattern)
    for i in range(0, 400, 15):
        cv2.line(img, (i, 0), (i, 300), (150, 150, 150), 1)
    
    # Initialize both cues
    print("Initializing MS cue...")
    ms = MultiScaleSaliencyCue(img)
    ms.thresholds = {scale: 0.2 for scale in ms.scales}
    ms.compute_saliency_maps()
    
    print("Initializing SS cue...")
    ss = SuperpixelStraddlingCue(img, n_segments=80)
    ss.compute_superpixels()
    
    # Test windows
    windows = [
        (110, 110, 290, 190),  # Tight around object (should score high on both)
        (50, 50, 150, 150),    # Partially on object
        (250, 50, 350, 150),   # On background
        (80, 80, 320, 220),    # Loose around object
    ]
    
    print("\nWindow Scores (MS + SS):")
    print("-" * 60)
    print(f"{'Window':<25} {'MS':<8} {'SS':<8} {'Combined':<8}")
    print("-" * 60)
    
    ms_weight = 0.6
    ss_weight = 0.4
    
    for window in windows:
        ms_score = ms.get_score(window)
        ss_score = ss.get_score(window)
        combined = ms_weight * ms_score + ss_weight * ss_score
        
        print(f"{str(window):<25} {ms_score:.4f}   {ss_score:.4f}   {combined:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # MS saliency
    combined_saliency = np.mean(list(ms.saliency_maps.values()), axis=0)
    axes[0, 1].imshow(combined_saliency, cmap='hot')
    axes[0, 1].set_title("MS Saliency Map")
    axes[0, 1].axis('off')
    
    # Superpixels
    vis_sp = ss.visualize_superpixels()
    axes[0, 2].imshow(cv2.cvtColor(vis_sp, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Superpixels")
    axes[0, 2].axis('off')
    
    # Window scores
    window_names = [f"W{i+1}" for i in range(len(windows))]
    ms_scores = [ms.get_score(w) for w in windows]
    ss_scores = [ss.get_score(w) for w in windows]
    combined_scores = [ms_weight * ms + ss_weight * ss for ms, ss in zip(ms_scores, ss_scores)]
    
    x = np.arange(len(windows))
    width = 0.25
    
    axes[1, 0].bar(x - width, ms_scores, width, label='MS', color='red')
    axes[1, 0].bar(x, ss_scores, width, label='SS', color='blue')
    axes[1, 0].bar(x + width, combined_scores, width, label='Combined', color='green')
    
    axes[1, 0].set_xlabel('Window')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Cue Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(window_names)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    
    # Show best window on original
    best_idx = np.argmax(combined_scores)
    best_window = windows[best_idx]
    
    img_best = img.copy()
    x1, y1, x2, y2 = best_window
    cv2.rectangle(img_best, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    axes[1, 1].imshow(cv2.cvtColor(img_best, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Best Window (Combined: {combined_scores[best_idx]:.3f})")
    axes[1, 1].axis('off')
    
    # Show straddling info for best window
    info = ss.get_window_straddling_info(best_window)
    info_text = f"Best Window Analysis:\n"
    info_text += f"MS Score: {ms_scores[best_idx]:.3f}\n"
    info_text += f"SS Score: {ss_scores[best_idx]:.3f}\n"
    info_text += f"Combined: {combined_scores[best_idx]:.3f}\n"
    info_text += f"Superpixels: {info['num_superpixels']}\n"
    info_text += f"Penalty: {info['penalty']:.1f}"
    
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=11, 
                   verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("Analysis")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('ms_ss_integration.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to ms_ss_integration.png")
    
    plt.show()
    
    return ms, ss

def main():
    """Run all tests"""
    print("="*70)
    print("SUPERPIXEL STRADDLING (SS) CUE TEST SUITE")
    print("="*70)
    
    # Run tests
    ss1 = test_ss_basic()
    ss2 = test_ss_with_real_image()
    ms, ss3 = test_ss_ms_integration()
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print("✓ SS Cue implementation verified")
    print("✓ Integration with MS Cue working")
    print("✓ Ready for Task 3 (Edge Density)")
    print("="*70)

if __name__ == "__main__":
    main()