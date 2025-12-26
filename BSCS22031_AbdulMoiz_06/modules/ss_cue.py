# modules/ss_cue_fixed.py
"""
FIXED Superpixels Straddling (SS) Cue Implementation
"""
import numpy as np
import cv2
from skimage.segmentation import felzenszwalb, slic
import warnings
warnings.filterwarnings('ignore')

try:
    from .integral_image import IntegralImage
except ImportError:
    from integral_image import IntegralImage

class SuperpixelStraddlingCue:
    """FIXED VERSION - Proper SS scoring"""
    
    def __init__(self, image, algorithm='slic', n_segments=100, compactness=10):
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image_rgb.shape[:2]
        
        self.algorithm = algorithm
        self.n_segments = n_segments
        self.compactness = compactness
        
        # Superpixel data
        self.superpixel_labels = None
        self.superpixel_areas = {}  # Total area of each superpixel
        self.superpixel_integrals = {}  # Integral images
        
        # Cache
        self.cache = {}
    
    def compute_superpixels(self):
        """Compute superpixel segmentation"""
        print("Computing superpixels...")
        
        if self.algorithm == 'slic':
            self.superpixel_labels = slic(
                self.image_rgb, 
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=1
            )
        else:
            self.superpixel_labels = felzenszwalb(
                self.image_rgb,
                scale=100,
                sigma=0.5,
                min_size=50
            )
        
        # Ensure labels are consecutive from 0
        unique_labels = np.unique(self.superpixel_labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        
        # Remap labels
        remapped = np.zeros_like(self.superpixel_labels)
        for old_label in unique_labels:
            remapped[self.superpixel_labels == old_label] = label_map[old_label]
        
        self.superpixel_labels = remapped
        unique_labels = np.unique(self.superpixel_labels)
        
        print(f"  Generated {len(unique_labels)} superpixels")
        
        # Create integral images
        self._create_integral_images()
        
        return self.superpixel_labels
    
    def _create_integral_images(self):
        """Create integral image for each superpixel"""
        unique_labels = np.unique(self.superpixel_labels)
        
        print(f"Creating integral images for {len(unique_labels)} superpixels...")
        
        for label in unique_labels:
            # Create binary mask
            mask = (self.superpixel_labels == label).astype(np.float32)
            
            # Store total area
            total_area = np.sum(mask)
            self.superpixel_areas[label] = total_area
            
            # Create integral image
            self.superpixel_integrals[label] = IntegralImage(mask)
        
        print("✓ Integral images created")
    
    def get_superpixel_area_in_window(self, label, window):
        """Get area of superpixel inside window"""
        if label not in self.superpixel_integrals:
            return 0
        
        x1, y1, x2, y2 = window
        
        # Clip to image bounds
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height))
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        integral = self.superpixel_integrals[label]
        return integral.rectangle_sum(x1, y1, x2, y2)
    
    def get_intersecting_superpixels(self, window):
        """Get superpixels that intersect with window"""
        if self.superpixel_labels is None:
            self.compute_superpixels()
        
        x1, y1, x2, y2 = window
        
        # Clip to image bounds
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height))
        
        if x2 <= x1 or y2 <= y1:
            return []
        
        # Extract labels from window region
        window_labels = self.superpixel_labels[y1:y2, x1:x2]
        return np.unique(window_labels)
    
    def get_score(self, window):
        """
        Calculate SS score: SS(w) = 1 - Penalty / WindowArea
        Penalty = Σ min(Area_in, Area_out)
        
        Higher score = less straddling = better window
        """
        cache_key = tuple(window)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.superpixel_labels is None:
            self.compute_superpixels()
        
        x1, y1, x2, y2 = window
        
        # Clip to image bounds
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height))
        
        if x2 <= x1 or y2 <= y1:
            self.cache[cache_key] = 0.0
            return 0.0
        
        window_area = (x2 - x1) * (y2 - y1)
        if window_area == 0:
            self.cache[cache_key] = 0.0
            return 0.0
        
        # Get intersecting superpixels
        intersecting_labels = self.get_intersecting_superpixels(window)
        
        # Calculate penalty
        penalty = 0.0
        
        for label in intersecting_labels:
            # Area of superpixel inside window
            area_in = self.get_superpixel_area_in_window(label, window)
            
            # Total area of superpixel
            total_area = self.superpixel_areas[label]
            
            # Area outside window
            area_out = total_area - area_in
            
            # Add min(area_in, area_out) to penalty
            penalty += min(area_in, area_out)
        
        # **CRITICAL FIX: The formula from your assignment**
        # SS(w) = 1 - Penalty / WindowArea
        ss_score = 1.0 - (penalty / window_area)
        
        # Clamp to [0, 1]
        ss_score = max(0.0, min(1.0, ss_score))
        
        # DEBUG: Print if score seems wrong
        if ss_score > 0.9 and window_area > self.height * self.width * 0.5:
            print(f"  DEBUG: Large window {window} has high SS score: {ss_score:.4f}")
            print(f"         Penalty: {penalty:.1f}, WindowArea: {window_area}")
        
        self.cache[cache_key] = ss_score
        return ss_score
    
    def visualize_superpixels(self, window=None):
        """Visualize superpixels"""
        if self.superpixel_labels is None:
            self.compute_superpixels()
        
        # Create color visualization
        from skimage.color import label2rgb
        colored = label2rgb(
            self.superpixel_labels, 
            self.image_rgb, 
            kind='avg',
            bg_label=-1
        )
        
        vis = (colored * 255).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        
        if window is not None:
            x1, y1, x2, y2 = window
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate and display score
            score = self.get_score(window)
            cv2.putText(vis, f"SS: {score:.3f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis

# Simple test to verify SS cue
def quick_ss_test():
    """Quick test to verify SS cue is working correctly"""
    print("\n" + "="*60)
    print("QUICK SS CUE VERIFICATION")
    print("="*60)
    
    # Create a simple test image with clear regions
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # Create 4 distinct regions
    img[0:100, 0:150] = [100, 0, 0]    # Red region
    img[0:100, 150:300] = [0, 100, 0]  # Green region
    img[100:200, 0:150] = [0, 0, 100]  # Blue region
    img[100:200, 150:300] = [100, 100, 0]  # Yellow region
    
    # Initialize SS cue with few superpixels
    ss = SuperpixelStraddlingCue(img, algorithm='slic', n_segments=4)
    ss.compute_superpixels()
    
    print(f"Unique superpixels: {np.unique(ss.superpixel_labels)}")
    
    # Test windows
    windows = [
        (25, 25, 125, 75),    # Inside red region (should have HIGH score)
        (175, 25, 275, 75),   # Inside green region (should have HIGH score)
        (125, 25, 225, 75),   # Straddling red-green (should have LOW score)
        (25, 125, 125, 175),  # Inside blue region
        (125, 125, 225, 175), # Straddling all 4 regions (should have LOWEST)
        (0, 0, 300, 200),     # Full image (should have LOW score)
    ]
    
    print("\nWindow Scores (correct SS behavior):")
    print("HIGH score = window inside single superpixel")
    print("LOW score = window straddles multiple superpixels")
    print("-" * 50)
    
    for i, window in enumerate(windows):
        score = ss.get_score(window)
        x1, y1, x2, y2 = window
        area = (x2-x1) * (y2-y1)
        
        # Get penalty for debugging
        intersecting = ss.get_intersecting_superpixels(window)
        penalty = 0
        for label in intersecting:
            area_in = ss.get_superpixel_area_in_window(label, window)
            total = ss.superpixel_areas[label]
            area_out = total - area_in
            penalty += min(area_in, area_out)
        
        print(f"Window {i+1} {window}:")
        print(f"  Area: {area}, Superpixels: {len(intersecting)}")
        print(f"  Penalty: {penalty:.1f}, SS Score: {score:.4f}")
        
        # Expected behavior
        if len(intersecting) == 1:
            print(f"  ✓ Good: Inside single superpixel")
        elif score < 0.3:
            print(f"  ✓ Good: Low score for straddling")
        elif score > 0.7 and area > 10000:  # Large window with high score
            print(f"  ⚠️ Warning: Large straddling window has high score")
        print()
    
    # Visualize
    cv2.imwrite('ss_quick_test.jpg', ss.visualize_superpixels(windows[2]))
    print("✓ Saved visualization to ss_quick_test.jpg")
    
    return ss

if __name__ == "__main__":
    quick_ss_test()