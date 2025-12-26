# modules/ed_cue.py
"""
Edge Density (ED) Cue Implementation - Group A
Measures edge density near window borders
"""
import numpy as np
import cv2

try:
    from .integral_image import IntegralImage
except ImportError:
    from integral_image import IntegralImage

class EdgeDensityCue:
    """
    Edge Density (ED) Cue
    Measures edge density near window borders
    Higher score = more edges near boundaries = likely an object
    """
    
    def __init__(self, image, border_ratio=0.1, canny_low=50, canny_high=150):
        """
        Initialize ED cue
        
        Args:
            image: Input image (BGR format)
            border_ratio: Ratio of window to consider as border (θ_ED in assignment)
                        Default 0.1 = 10% of window dimensions
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
        """
        self.image = image
        self.border_ratio = border_ratio
        self.canny_low = canny_low
        self.canny_high = canny_high
        
        # Convert to grayscale
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image
        
        self.height, self.width = self.gray.shape
        
        # Compute edge map
        self.edge_map = self._compute_edge_map()
        
        # Create integral image of edge map
        self.edge_integral = IntegralImage(self.edge_map)
        
        print(f"Edge Density Cue initialized (border_ratio={border_ratio})")
    
    def _compute_edge_map(self):
        """
        Compute Canny edge map
        Returns binary edge map where edge pixels = 1, others = 0
        """
        print("Computing edge map...")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 1.4)
        
        # Compute edge map using Canny
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Convert to float [0, 1]
        edge_map = edges.astype(np.float32) / 255.0
        
        # Count edges for debugging
        edge_pixels = np.sum(edge_map)
        total_pixels = edge_map.size
        print(f"  Edge pixels: {edge_pixels}/{total_pixels} ({edge_pixels/total_pixels*100:.2f}%)")
        
        return edge_map
    
    def _compute_canny_manual(self):
        """
        Manual Canny edge detector implementation for extra marks
        Steps: 1. Gaussian blur, 2. Gradient computation, 3. Non-max suppression, 4. Hysteresis
        """
        print("Computing Canny edges manually...")
        
        # 1. Gaussian blur
        kernel_size = 5
        sigma = 1.4
        kernel = self._gaussian_kernel(kernel_size, sigma)
        blurred = cv2.filter2D(self.gray, -1, kernel)
        
        # 2. Compute gradients using Sobel
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        direction = np.mod(direction, 180)
        
        # 3. Non-maximum suppression
        suppressed = self._non_max_suppression(magnitude, direction)
        
        # 4. Hysteresis thresholding
        edges = self._hysteresis_threshold(suppressed, self.canny_low, self.canny_high)
        
        return edges.astype(np.float32) / 255.0
    
    def _gaussian_kernel(self, size, sigma):
        """Create Gaussian kernel"""
        ax = np.linspace(-(size-1)/2., (size-1)/2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        return kernel / kernel.sum()
    
    def _non_max_suppression(self, magnitude, direction):
        """Apply non-maximum suppression"""
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                angle = direction[i, j]
                
                # Quantize angle to 0, 45, 90, 135 degrees
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q1 = magnitude[i, j-1]
                    q2 = magnitude[i, j+1]
                elif 22.5 <= angle < 67.5:
                    q1 = magnitude[i-1, j+1]
                    q2 = magnitude[i+1, j-1]
                elif 67.5 <= angle < 112.5:
                    q1 = magnitude[i-1, j]
                    q2 = magnitude[i+1, j]
                else:  # 112.5 <= angle < 157.5
                    q1 = magnitude[i-1, j-1]
                    q2 = magnitude[i+1, j+1]
                
                if magnitude[i, j] >= q1 and magnitude[i, j] >= q2:
                    suppressed[i, j] = magnitude[i, j]
        
        return suppressed
    
    def _hysteresis_threshold(self, image, low, high):
        """Apply hysteresis thresholding"""
        strong = 255
        weak = 50
        
        # Create strong and weak edges
        strong_edges = (image >= high)
        weak_edges = (image >= low) & (image < high)
        
        # Initialize output
        edges = np.zeros_like(image, dtype=np.uint8)
        edges[strong_edges] = strong
        edges[weak_edges] = weak
        
        # Track weak edges connected to strong edges
        h, w = edges.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                if edges[i, j] == weak:
                    # Check 8-connected neighborhood
                    if np.any(edges[i-1:i+2, j-1:j+2] == strong):
                        edges[i, j] = strong
                    else:
                        edges[i, j] = 0
        
        # Convert to binary
        edges = (edges == strong).astype(np.uint8) * 255
        
        return edges
    
    def get_target_region(self, window):
        """
        Calculate target region (border area) for a window
        
        Args:
            window: (x1, y1, x2, y2)
            
        Returns:
            (outer_window, inner_window) coordinates
        """
        x1, y1, x2, y2 = window
        w = x2 - x1
        h = y2 - y1
        
        # Calculate border thickness
        border_w = int(w * self.border_ratio)
        border_h = int(h * self.border_ratio)
        
        # Outer window is the original window
        outer_window = (x1, y1, x2, y2)
        
        # Inner window (shrunk by border_ratio)
        inner_x1 = x1 + border_w
        inner_y1 = y1 + border_h
        inner_x2 = x2 - border_w
        inner_y2 = y2 - border_h
        
        # Ensure inner window is valid (non-negative dimensions)
        if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
            # If window too small, use entire window as target region
            inner_window = outer_window
        else:
            inner_window = (inner_x1, inner_y1, inner_x2, inner_y2)
        
        return outer_window, inner_window
    
    def get_edge_sum_in_region(self, window):
        """
        Get sum of edge pixels in a window region using integral image
        
        Args:
            window: (x1, y1, x2, y2)
            
        Returns:
            Sum of edge pixels in the region
        """
        x1, y1, x2, y2 = window
        
        # Clip to image bounds
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height))
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        # Use integral image for O(1) calculation
        return self.edge_integral.rectangle_sum(x1, y1, x2, y2)
    
    def get_score(self, window, use_perimeter=True):
        """
        Calculate ED score for a window
        
        Args:
            window: (x1, y1, x2, y2)
            use_perimeter: If True, use perimeter normalization (2W+2H)
                          If False, use target region area normalization
                          
        Returns:
            ED score [0, 1], higher = more edges near borders
        """
        x1, y1, x2, y2 = window
        w = x2 - x1
        h = y2 - y1
        
        if w <= 0 or h <= 0:
            return 0.0
        
        # Get target region (border area)
        outer_window, inner_window = self.get_target_region(window)
        
        # Calculate edge sum in target region
        outer_sum = self.get_edge_sum_in_region(outer_window)
        inner_sum = self.get_edge_sum_in_region(inner_window)
        target_sum = outer_sum - inner_sum
        
        # Calculate target region area
        outer_area = (outer_window[2] - outer_window[0]) * (outer_window[3] - outer_window[1])
        inner_area = (inner_window[2] - inner_window[0]) * (inner_window[3] - inner_window[1])
        target_area = outer_area - inner_area
        
        if target_area <= 0:
            return 0.0
        
        if use_perimeter:
            # Normalize by perimeter (2W + 2H) as in assignment
            perimeter = 2 * w + 2 * h
            if perimeter > 0:
                ed_score = target_sum / perimeter
            else:
                ed_score = 0.0
        else:
            # Alternative: Normalize by target region area
            ed_score = target_sum / target_area
        
        # Clamp to [0, 1]
        ed_score = max(0.0, min(1.0, ed_score))
        
        return ed_score
    
    def get_detailed_score(self, window):
        """
        Get detailed ED score information
        
        Returns:
            Dictionary with score breakdown
        """
        x1, y1, x2, y2 = window
        w = x2 - x1
        h = y2 - y1
        
        # Get target region
        outer_window, inner_window = self.get_target_region(window)
        
        # Calculate edge sums
        outer_sum = self.get_edge_sum_in_region(outer_window)
        inner_sum = self.get_edge_sum_in_region(inner_window)
        target_sum = outer_sum - inner_sum
        
        # Calculate areas
        outer_area = (outer_window[2] - outer_window[0]) * (outer_window[3] - outer_window[1])
        inner_area = (inner_window[2] - inner_window[0]) * (inner_window[3] - inner_window[1])
        target_area = outer_area - inner_area
        
        # Calculate scores with different normalizations
        perimeter = 2 * w + 2 * h
        score_perimeter = target_sum / perimeter if perimeter > 0 else 0
        score_area = target_sum / target_area if target_area > 0 else 0
        
        return {
            'window': window,
            'dimensions': (w, h),
            'border_ratio': self.border_ratio,
            'border_thickness': (int(w * self.border_ratio), int(h * self.border_ratio)),
            'outer_window': outer_window,
            'inner_window': inner_window,
            'edge_sums': {
                'outer': float(outer_sum),
                'inner': float(inner_sum),
                'target': float(target_sum)
            },
            'areas': {
                'outer': int(outer_area),
                'inner': int(inner_area),
                'target': int(target_area)
            },
            'perimeter': float(perimeter),
            'scores': {
                'perimeter_normalized': float(score_perimeter),
                'area_normalized': float(score_area)
            }
        }
    
    def visualize_edges(self, window=None):
        """
        Visualize edge map with optional window overlay
        
        Args:
            window: Optional window to highlight
            
        Returns:
            Visualization image
        """
        # Convert edge map to 3-channel for visualization
        edge_vis = (self.edge_map * 255).astype(np.uint8)
        edge_vis = cv2.cvtColor(edge_vis, cv2.COLOR_GRAY2BGR)
        
        # Overlay on original image (semi-transparent)
        overlay = self.image.copy()
        if len(overlay.shape) == 3:
            # Create red edges
            edge_red = np.zeros_like(overlay)
            edge_red[:, :, 2] = self.edge_map * 255  # Red channel
            
            # Blend with original
            alpha = 0.7
            overlay = cv2.addWeighted(overlay, 1-alpha, edge_red, alpha, 0)
        else:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        
        if window is not None:
            x1, y1, x2, y2 = window
            
            # Draw window
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(edge_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw target region (border)
            outer, inner = self.get_target_region(window)
            ox1, oy1, ox2, oy2 = outer
            ix1, iy1, ix2, iy2 = inner
            
            # Draw inner rectangle
            cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (255, 0, 0), 1)
            cv2.rectangle(edge_vis, (ix1, iy1), (ix2, iy2), (255, 0, 0), 1)
            
            # Calculate and display score
            score = self.get_score(window)
            cv2.putText(overlay, f"ED: {score:.3f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show target region area
            target_area = (ox2-ox1)*(oy2-oy1) - (ix2-ix1)*(iy2-iy1)
            cv2.putText(overlay, f"Target Area: {target_area}", (x1, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Create side-by-side visualization
        h, w = overlay.shape[:2]
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = overlay
        combined[:, w:] = edge_vis
        
        # Add labels
        cv2.putText(combined, "Original + Edges", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Edge Map", (w+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if window is not None:
            cv2.putText(combined, f"Green: Window, Blue: Inner Region", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return combined

# Test function
def test_ed_cue():
    """Test Edge Density cue"""
    print("\n" + "="*60)
    print("EDGE DENSITY (ED) CUE TEST - Group A")
    print("="*60)
    
    # Create test image with clear edges
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add a square with clear edges
    cv2.rectangle(img, (100, 100), (300, 200), (255, 255, 255), -1)
    
    # Add some noise inside the square (less edges)
    for i in range(100, 300, 10):
        cv2.line(img, (i, 110), (i, 190), (200, 200, 200), 1)
    
    # Add texture to background (more edges)
    for i in range(0, 400, 5):
        cv2.line(img, (i, 0), (i, 300), (50, 50, 50), 1)
    
    # Initialize ED cue
    print("Initializing ED cue...")
    ed = EdgeDensityCue(img, border_ratio=0.1)
    
    # Test different windows
    windows = [
        (90, 90, 310, 210),   # Around the square (should have HIGH edge density on borders)
        (110, 110, 290, 190), # Inside the square (should have LOW edge density)
        (50, 50, 150, 150),   # On background texture (MEDIUM edge density)
        (0, 0, 400, 300),     # Full image (LOW edge density on borders)
    ]
    
    print("\nWindow ED Scores (perimeter normalization):")
    print("Higher = more edges near window borders")
    print("-" * 60)
    
    for i, window in enumerate(windows):
        score = ed.get_score(window, use_perimeter=True)
        detailed = ed.get_detailed_score(window)
        
        print(f"\nWindow {i+1} {window}:")
        print(f"  ED Score: {score:.4f}")
        print(f"  Edge sum in target region: {detailed['edge_sums']['target']:.1f}")
        print(f"  Target area: {detailed['areas']['target']}")
        print(f"  Perimeter: {detailed['perimeter']:.1f}")
        
        # Expected behavior
        if i == 0:  # Around square
            print(f"  ✓ Expected: HIGH (edges on square borders)")
        elif i == 1:  # Inside square
            print(f"  ✓ Expected: LOW (few edges on inner borders)")
    
    # Visualize
    print("\nCreating visualizations...")
    vis = ed.visualize_edges(windows[0])
    cv2.imwrite('ed_test_result.jpg', vis)
    print("✓ Saved ed_test_result.jpg")
    
    # Test with different border ratios
    print("\n" + "-"*60)
    print("Testing different border ratios:")
    print("-"*60)
    
    test_window = windows[0]
    for ratio in [0.05, 0.1, 0.15, 0.2]:
        ed_test = EdgeDensityCue(img, border_ratio=ratio)
        score = ed_test.get_score(test_window)
        print(f"Border ratio {ratio:.2f}: ED Score = {score:.4f}")
    
    return ed

if __name__ == "__main__":
    test_ed_cue()