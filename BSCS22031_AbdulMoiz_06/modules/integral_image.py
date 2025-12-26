# modules/integral_image.py
import numpy as np

class IntegralImage:
    """Core class for O(1) rectangle sum calculations"""
    
    def __init__(self, image):
        """
        Compute integral image (summed area table)
        
        Args:
            image: 2D numpy array (height, width)
        """
        if len(image.shape) > 2:
            raise ValueError("IntegralImage requires 2D array")
        
        # Compute cumulative sums
        self.integral = np.cumsum(np.cumsum(image, axis=0), axis=1)
        
        # Pad with zeros for easier indexing
        self.integral = np.pad(self.integral, ((1, 0), (1, 0)), mode='constant')
        self.height, self.width = image.shape
    
    def rectangle_sum(self, x1, y1, x2, y2):
        """
        Calculate sum in rectangle [x1, x2) x [y1, y2) in O(1)
        
        Note: Coordinates are inclusive-exclusive [x1, x2)
        Returns 0 if rectangle is invalid
        """
        # Convert to integral image coordinates (1-indexed due to padding)
        x1, x2 = x1 + 1, x2 + 1
        y1, y2 = y1 + 1, y2 + 1
        
        # Ensure bounds
        x1 = max(0, min(x1, self.width))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height))
        y2 = max(0, min(y2, self.height))
        
        if x1 >= x2 or y1 >= y2:
            return 0
        
        # Standard formula: D - B - C + A
        D = self.integral[y2, x2]
        B = self.integral[y2, x1]
        C = self.integral[y1, x2]
        A = self.integral[y1, x1]
        
        return D - B - C + A
    
    def window_density(self, x1, y1, x2, y2):
        """Helper: density = sum / area"""
        area = (x2 - x1) * (y2 - y1)
        if area == 0:
            return 0
        return self.rectangle_sum(x1, y1, x2, y2) / area

# No need for relative imports here