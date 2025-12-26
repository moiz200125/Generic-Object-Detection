# test_pipeline.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.integral_image import IntegralImage
from modules.ed_cue import EdgeDensityCue

def test_on_real_image():
    """Test your implementation on a real image"""
    
    # Option 1: Create a synthetic image for testing
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(img, (100, 100), 30, (200, 200, 200), -1)
    
    # Option 2: Load a real image (if you have one)
    # img = cv2.imread('test_image.jpg')
    
    # Test Edge Density
    print("Testing Edge Density...")
    ed = EdgeDensityCue(img)
    
    # Generate some test windows
    windows = []
    for i in range(0, 150, 25):
        for j in range(0, 150, 25):
            windows.append((i, j, i+50, j+50))
    
    # Score each window
    scores = []
    for window in windows:
        score = ed.get_score(window)
        scores.append(score)
    
    # Find best window
    best_idx = np.argmax(scores)
    best_window = windows[best_idx]
    best_score = scores[best_idx]
    
    # Visualize
    vis = img.copy()
    x1, y1, x2, y2 = best_window
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"Score: {best_score:.3f}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show edge map
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(132)
    plt.title("Edge Map")
    plt.imshow(ed.edge_map, cmap='gray')
    
    plt.subplot(133)
    plt.title(f"Best Window (Score: {best_score:.3f})")
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.savefig('ed_test_results.png')
    plt.show()
    
    print(f"✅ Best window: {best_window} with score: {best_score:.4f}")
    return True

if __name__ == "__main__":
    test_on_real_image()