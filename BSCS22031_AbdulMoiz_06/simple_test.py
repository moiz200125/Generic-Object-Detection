# simple_test.py
"""
Simple test without import issues - run this first!
"""
import cv2
import numpy as np
import os

# Create synthetic image for testing
def create_test_image():
    """Create a simple test image with a clear object"""
    img = np.ones((300, 400, 3), dtype=np.uint8) * 100  # Gray background
    
    # Add background texture (lines)
    for i in range(0, 400, 20):
        cv2.line(img, (i, 0), (i, 300), (150, 150, 150), 1)
    for i in range(0, 300, 20):
        cv2.line(img, (0, i), (400, i), (150, 150, 150), 1)
    
    # Add a red circle (object)
    cv2.circle(img, (200, 150), 60, (0, 0, 255), -1)
    cv2.circle(img, (200, 150), 30, (0, 255, 255), -1)
    
    return img

# Copy integral_image code directly into test
class IntegralImage:
    def __init__(self, image):
        self.integral = np.cumsum(np.cumsum(image, axis=0), axis=1)
        self.integral = np.pad(self.integral, ((1, 0), (1, 0)), mode='constant')
        self.height, self.width = image.shape
    
    def rectangle_sum(self, x1, y1, x2, y2):
        x1, x2 = x1 + 1, x2 + 1
        y1, y2 = y1 + 1, y2 + 1
        
        x1 = max(0, min(x1, self.width))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height))
        y2 = max(0, min(y2, self.height))
        
        if x1 >= x2 or y1 >= y2:
            return 0
        
        D = self.integral[y2, x2]
        B = self.integral[y2, x1]
        C = self.integral[y1, x2]
        A = self.integral[y1, x1]
        
        return D - B - C + A

# Copy simplified MS cue directly into test
class SimpleMSCue:
    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.image.shape
        
    def spectral_residual(self, img):
        """Simple spectral residual"""
        fft = np.fft.fft2(img)
        amplitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Avoid log(0)
        amplitude[amplitude == 0] = 1e-10
        
        log_amplitude = np.log(amplitude)
        local_avg = cv2.blur(log_amplitude, (3, 3))
        residual = log_amplitude - local_avg
        
        reconstructed = np.exp(residual) * np.exp(1j * phase)
        saliency = np.abs(np.fft.ifft2(reconstructed))
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
        
        return saliency
    
    def compute_saliency(self):
        """Compute saliency at a single scale for simplicity"""
        # Use a single scale for testing
        scale = 32
        h, w = self.image.shape
        
        # Resize
        if h > w:
            new_h = scale
            new_w = int(w * (scale / h))
        else:
            new_w = scale
            new_h = int(h * (scale / w))
        
        resized = cv2.resize(self.image, (new_w, new_h))
        
        # Pad to square
        if new_w < scale or new_h < scale:
            pad_h = (scale - new_h) // 2
            pad_w = (scale - new_w) // 2
            resized = cv2.copyMakeBorder(resized, pad_h, scale-new_h-pad_h, 
                                        pad_w, scale-new_w-pad_w, cv2.BORDER_CONSTANT, 0)
        
        # Compute spectral residual
        saliency = self.spectral_residual(resized)
        
        # Remove padding
        if new_w < scale or new_h < scale:
            saliency = saliency[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
        
        # Resize back
        saliency = cv2.resize(saliency, (w, h))
        
        # Binarize
        binary = (saliency > 0.5).astype(np.float32)
        
        # Create integral image
        self.integral_saliency = IntegralImage(binary)
        self.saliency_map = saliency
        self.binary_map = binary
        
        return saliency, binary
    
    def get_window_score(self, window):
        """Score a window"""
        x1, y1, x2, y2 = window
        area = (x2 - x1) * (y2 - y1)
        
        if area == 0:
            return 0
        
        salient_sum = self.integral_saliency.rectangle_sum(x1, y1, x2, y2)
        return salient_sum / area

# Main test
def run_simple_test():
    print("="*60)
    print("SIMPLE TEST - No Import Issues")
    print("="*60)
    
    # Create test image
    test_img = create_test_image()
    
    # Save it
    cv2.imwrite('test_image.jpg', test_img)
    print("Created test_image.jpg")
    
    # Initialize cue
    ms = SimpleMSCue(test_img)
    
    # Compute saliency
    print("Computing saliency...")
    saliency, binary = ms.compute_saliency()
    
    # Test some windows
    windows = [
        (170, 90, 230, 150),   # On the object
        (50, 50, 100, 100),    # On background
        (0, 0, 400, 300),      # Full image
    ]
    
    print("\nWindow Scores:")
    for i, window in enumerate(windows):
        score = ms.get_window_score(window)
        print(f"  Window {i+1} {window}: {score:.4f}")
    
    # Visualize
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Draw windows on original
    img_with_windows = test_img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, window in enumerate(windows):
        x1, y1, x2, y2 = window
        cv2.rectangle(img_with_windows, (x1, y1), (x2, y2), colors[i], 2)
        score = ms.get_window_score(window)
        cv2.putText(img_with_windows, f"{score:.3f}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
    
    axes[1].imshow(cv2.cvtColor(img_with_windows, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Windows with Scores")
    axes[1].axis('off')
    
    axes[2].imshow(saliency, cmap='hot')
    axes[2].set_title("Saliency Map")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_test_results.png', dpi=150)
    print("\n✓ Saved visualization to: simple_test_results.png")
    plt.show()
    
    print("\n" + "="*60)
    print("SIMPLE TEST COMPLETE!")
    print("="*60)
    return True

if __name__ == "__main__":
    run_simple_test()