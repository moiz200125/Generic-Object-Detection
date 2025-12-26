# test_ms_cue.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.ms_cue import MultiScaleSaliencyCue

def test_ms_cue():
    """Test the Multi-scale Saliency implementation"""
    
    print("Testing Multi-scale Saliency Cue...")
    
    # Create a synthetic test image with a salient object
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Create background with some texture (lower frequency)
    for i in range(0, 200, 10):
        cv2.line(img, (0, i), (200, i), (100, 100, 100), 1)
        cv2.line(img, (i, 0), (i, 200), (100, 100, 100), 1)
    
    # Create a salient object (different from background)
    cv2.circle(img, (100, 100), 40, (200, 50, 50), -1)  # Blue circle
    cv2.circle(img, (100, 100), 20, (50, 200, 50), -1)  # Green inner circle
    
    # Add some high-frequency noise to make it more unique
    noise = np.random.randint(0, 30, (40, 40, 3), dtype=np.uint8)
    img[80:120, 80:120] = cv2.add(img[80:120, 80:120], noise)
    
    # Initialize and compute MS cue
    print("Initializing MS Cue...")
    ms = MultiScaleSaliencyCue(img)
    ms.compute_saliency_maps()
    
    # Test some windows
    test_windows = [
        (80, 80, 120, 120),    # On the object (should have high score)
        (20, 20, 60, 60),      # On background (should have low score)
        (0, 0, 200, 200),      # Entire image
        (140, 140, 180, 180),  # Empty background
    ]
    
    print("\nTesting window scores:")
    print("-" * 50)
    
    scores = {}
    for i, window in enumerate(test_windows):
        score = ms.get_score(window)
        scores[f"Window {i+1}"] = score
        x1, y1, x2, y2 = window
        print(f"Window {i+1}: [{x1}, {y1}, {x2}, {y2}] -> MS Score: {score:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Draw test windows on original image
    img_with_windows = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    for i, window in enumerate(test_windows):
        x1, y1, x2, y2 = window
        cv2.rectangle(img_with_windows, (x1, y1), (x2, y2), colors[i], 2)
        # Add score text
        cv2.putText(img_with_windows, f"{scores[f'Window {i+1}']:.3f}", 
                   (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
    
    axes[0, 1].imshow(cv2.cvtColor(img_with_windows, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Test Windows with Scores")
    axes[0, 1].axis('off')
    
    # Show saliency maps for each scale
    scales = ms.scales
    for i, scale in enumerate(scales[:3]):  # Show first 3 scales
        saliency_map = ms.saliency_maps[scale]
        axes[0, 2+i].imshow(saliency_map, cmap='hot')
        axes[0, 2+i].set_title(f"Saliency Map (Scale {scale})")
        axes[0, 2+i].axis('off')
    
    # Show binary maps for each scale
    for i, scale in enumerate(scales[:3]):  # Show first 3 scales
        binary_map = ms.binary_maps[scale]
        axes[1, i].imshow(binary_map, cmap='gray')
        axes[1, i].set_title(f"Binary Map (Scale {scale})")
        axes[1, i].axis('off')
    
    # Show combined visualization
    vis = ms.get_saliency_visualization()
    axes[1, 3].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title("Multi-scale Visualization")
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('ms_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Verify that Integral Images are working
    print("\nVerifying Integral Image calculations:")
    print("-" * 50)
    
    # Test with a specific window and scale
    test_window = test_windows[0]
    test_scale = scales[0]
    
    # Manual calculation for verification
    binary_map = ms.binary_maps[test_scale]
    x1, y1, x2, y2 = test_window
    manual_sum = np.sum(binary_map[y1:y2, x1:x2])
    manual_area = (x2 - x1) * (y2 - y1)
    manual_density = manual_sum / manual_area if manual_area > 0 else 0
    
    # Integral image calculation
    integral_density = ms.get_window_density(test_window, test_scale)
    
    print(f"Test Window: {test_window}")
    print(f"Scale: {test_scale}")
    print(f"Manual calculation - Sum: {manual_sum:.2f}, Area: {manual_area}, Density: {manual_density:.4f}")
    print(f"Integral Image calculation - Density: {integral_density:.4f}")
    print(f"Match: {np.abs(manual_density - integral_density) < 0.01}")
    
    return ms

def test_with_real_image():
    """Test MS cue with a real image"""
    print("\n" + "="*60)
    print("Testing with real image...")
    print("="*60)
    
    # Create or load a real image
    # Option 1: Create synthetic image with clear object
    img = np.ones((300, 300, 3), dtype=np.uint8) * 100
    
    # Add a salient object (red square with texture)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), -1)
    
    # Add texture to the object
    for i in range(100, 200, 5):
        cv2.line(img, (100, i), (200, i), (0, 100, 255), 1)
    
    # Add texture to background (different frequency)
    for i in range(0, 300, 15):
        cv2.line(img, (i, 0), (i, 300), (150, 150, 150), 1)
    
    # Initialize MS cue
    ms = MultiScaleSaliencyCue(img)
    ms.compute_saliency_maps()
    
    # Test sliding window approach
    print("\nTesting sliding window detection...")
    stride = 50
    best_score = -1
    best_window = None
    
    for y in range(0, img.shape[0] - 60, stride):
        for x in range(0, img.shape[1] - 60, stride):
            window = (x, y, x + 60, y + 60)
            score = ms.get_score(window)
            
            if score > best_score:
                best_score = score
                best_window = window
    
    print(f"Best window: {best_window}")
    print(f"Best MS score: {best_score:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    
    # Draw best window
    x1, y1, x2, y2 = best_window
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      linewidth=2, edgecolor='green', 
                                      facecolor='none'))
    plt.text(x1, y1-5, f"Score: {best_score:.3f}", 
             color='green', fontsize=10, fontweight='bold')
    
    plt.subplot(1, 2, 2)
    # Show combined saliency map (average of all scales)
    combined_saliency = np.mean(list(ms.saliency_maps.values()), axis=0)
    plt.imshow(combined_saliency, cmap='hot')
    plt.title("Combined Saliency Map")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('ms_real_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return ms

if __name__ == "__main__":
    print("="*60)
    print("MULTI-SCALE SALIENCY (MS) CUE IMPLEMENTATION TEST")
    print("="*60)
    
    # Run tests
    ms1 = test_ms_cue()
    ms2 = test_with_real_image()
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)