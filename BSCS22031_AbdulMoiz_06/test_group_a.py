# test_group_a.py
"""
Quick test for Group A complete pipeline
"""
import cv2
import numpy as np
import sys
import os

sys.path.append('modules')

# Test all cues together
def test_complete_pipeline():
    print("\n" + "="*70)
    print("GROUP A COMPLETE PIPELINE TEST (MS + SS + ED)")
    print("="*70)
    
    # Create test image
    img = np.ones((300, 400, 3), dtype=np.uint8) * 100
    
    # Add a clear object with edges
    cv2.rectangle(img, (100, 100), (300, 200), (0, 0, 255), -1)
    
    # Add texture inside object (for MS saliency)
    for i in range(100, 300, 8):
        cv2.line(img, (i, 110), (i, 190), (0, 50, 255), 1)
    
    # Add texture to background (different pattern)
    for i in range(0, 400, 15):
        cv2.line(img, (i, 0), (i, 300), (150, 150, 150), 1)
    
    cv2.imwrite('test_object.jpg', img)
    print("Created test_object.jpg")
    
    # Test individual cues first
    print("\n1. Testing Edge Density (ED) cue...")
    from ed_cue import EdgeDensityCue
    ed = EdgeDensityCue(img, border_ratio=0.1)
    
    test_windows = [
        (90, 90, 310, 210),   # Around object (high edges on border)
        (110, 110, 290, 190), # Inside object (low edges)
        (50, 50, 150, 150),   # Background (medium edges)
    ]
    
    for i, window in enumerate(test_windows):
        score = ed.get_score(window)
        print(f"  Window {i+1} {window}: ED={score:.4f}")
    
    print("\n2. Testing all cues together...")
    
    # Create simple detector
    from ms_cue import MultiScaleSaliencyCue
    from ss_cue import SuperpixelStraddlingCue
    
    ms = MultiScaleSaliencyCue(img)
    ms.thresholds = {scale: 0.2 for scale in ms.scales}
    ms.compute_saliency_maps()
    
    ss = SuperpixelStraddlingCue(img, n_segments=80)
    ss.compute_superpixels()
    
    # Define weights
    weights = {'ms': 0.4, 'ss': 0.3, 'ed': 0.3}
    
    print(f"\nWindow Scores with weights {weights}:")
    print("-"*60)
    
    for i, window in enumerate(test_windows):
        ms_score = ms.get_score(window)
        ss_score = ss.get_score(window)
        ed_score = ed.get_score(window)
        
        total = (weights['ms'] * ms_score + 
                weights['ss'] * ss_score + 
                weights['ed'] * ed_score)
        
        print(f"\nWindow {i+1} {window}:")
        print(f"  MS:  {ms_score:.4f} × {weights['ms']:.1f} = {weights['ms']*ms_score:.4f}")
        print(f"  SS:  {ss_score:.4f} × {weights['ss']:.1f} = {weights['ss']*ss_score:.4f}")
        print(f"  ED:  {ed_score:.4f} × {weights['ed']:.1f} = {weights['ed']*ed_score:.4f}")
        print(f"  TOTAL: {total:.4f}")
        
        # Expected behavior
        if i == 0:
            print(f"  ✓ Should be HIGHEST (around object)")
        elif i == 1:
            print(f"  ✓ Should be MEDIUM (inside object)")
        elif i == 2:
            print(f"  ✓ Should be LOW (background)")
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    
    # ED visualization
    ed_vis = ed.visualize_edges(test_windows[0])
    cv2.imwrite('ed_visualization.jpg', ed_vis)
    print("  ✓ Saved ed_visualization.jpg")
    
    # Combined visualization
    h, w = img.shape[:2]
    combined = np.zeros((h, w*3, 3), dtype=np.uint8)
    
    # Original with windows
    img_windows = img.copy()
    for i, window in enumerate(test_windows):
        x1, y1, x2, y2 = window
        color = [(0, 255, 0), (255, 255, 0), (255, 0, 0)][i]
        cv2.rectangle(img_windows, (x1, y1), (x2, y2), color, 2)
    
    combined[:, :w] = img_windows
    
    # MS saliency
    ms_saliency = np.mean(list(ms.saliency_maps.values()), axis=0)
    ms_saliency = (ms_saliency * 255).astype(np.uint8)
    ms_saliency = cv2.applyColorMap(ms_saliency, cv2.COLORMAP_JET)
    combined[:, w:w*2] = ms_saliency
    
    # ED edges
    ed_edges = (ed.edge_map * 255).astype(np.uint8)
    ed_edges = cv2.cvtColor(ed_edges, cv2.COLOR_GRAY2BGR)
    combined[:, w*2:w*3] = ed_edges
    
    cv2.putText(combined, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "MS Saliency", (w+10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "ED Edges", (w*2+10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite('combined_cues.jpg', combined)
    print("  ✓ Saved combined_cues.jpg")
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print("Generated files:")
    print("  - test_object.jpg")
    print("  - ed_visualization.jpg")
    print("  - combined_cues.jpg")
    print("="*70)

if __name__ == "__main__":
    test_complete_pipeline()