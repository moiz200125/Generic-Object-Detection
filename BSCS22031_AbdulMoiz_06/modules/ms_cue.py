# modules/ms_cue.py
import numpy as np
import cv2

# Remove relative import, use direct import
try:
    # Try relative import first
    from .integral_image import IntegralImage
except ImportError:
    # Fall back to absolute import
    from integral_image import IntegralImage

class MultiScaleSaliencyCue:
    """
    Multi-scale Saliency (MS) Cue
    Detects uniqueness compared to global image statistics using Spectral Residual
    """
    
    def __init__(self, image):
        """
        Initialize with an RGB/BGR image
        
        Args:
            image: Input image (H, W, 3) in BGR format (OpenCV default)
        """
        # Convert BGR to RGB for consistent processing
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image_rgb.shape[:2]
        
        # Store the saliency maps and binary maps for each scale
        self.saliency_maps = {}  # Scale -> Saliency map
        self.binary_maps = {}    # Scale -> Binary map (thresholded)
        self.integral_images = {} # Scale -> IntegralImage of binary map
        
        # Define scales as per assignment: {16, 24, 32, 48, 64}
        self.scales = [16, 24, 32, 48, 64]
        
        # Threshold for binarization (will be learned, default 0.5)
        self.thresholds = {scale: 0.5 for scale in self.scales}
        
        # Weights for combining scales (from paper)
        self.weights = {
            16: 0.25,
            24: 0.25,
            32: 0.20,
            48: 0.15,
            64: 0.15
        }
    
    def _spectral_residual(self, image_channel):
        """
        Compute Spectral Residual Saliency for a single channel
        """
        # Step 1: 2D Fast Fourier Transform
        fft = np.fft.fft2(image_channel)
        
        # Step 2: Get magnitude (amplitude) and phase
        amplitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Avoid log(0)
        amplitude[amplitude == 0] = 1e-10
        
        # Step 3: Log Spectrum
        log_amplitude = np.log(amplitude)
        
        # Step 4: Local average of log spectrum (3x3 box filter)
        local_avg = cv2.blur(log_amplitude, (3, 3))
        
        # Step 5: Spectral Residual
        residual = log_amplitude - local_avg
        
        # Step 6: Reconstruction
        reconstructed = np.exp(residual) * np.exp(1j * phase)
        
        # Inverse FFT
        saliency = np.abs(np.fft.ifft2(reconstructed))
        
        # Step 7: Square and Gaussian smoothing
        saliency = saliency ** 2
        saliency = cv2.GaussianBlur(saliency, (3, 3), sigmaX=0.5, sigmaY=0.5)
        
        # Normalize to [0, 1]
        if saliency.max() > 0:
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        return saliency
    
    def _compute_single_scale_saliency(self, scale):
        """
        Compute saliency map for a single scale
        """
        channel_saliencies = []
        
        # Process each channel (R, G, B)
        for channel_idx in range(3):
            channel = self.image_rgb[:, :, channel_idx].astype(np.float32)
            
            # Calculate resize factor
            h, w = channel.shape
            if h > w:
                new_h = scale
                new_w = int(w * (scale / h))
            else:
                new_w = scale
                new_h = int(h * (scale / w))
            
            # Resize to target dimensions
            resized_channel = cv2.resize(channel, (new_w, new_h))
            
            # Pad if needed
            if new_w < scale or new_h < scale:
                pad_h = (scale - new_h) // 2
                pad_w = (scale - new_w) // 2
                resized_channel = cv2.copyMakeBorder(
                    resized_channel, 
                    pad_h, scale - new_h - pad_h,
                    pad_w, scale - new_w - pad_w,
                    cv2.BORDER_CONSTANT, value=0
                )
            
            # Compute spectral residual
            saliency = self._spectral_residual(resized_channel)
            
            # Remove padding if we added it
            if new_w < scale or new_h < scale:
                saliency = saliency[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
            
            # Resize back to original dimensions
            saliency = cv2.resize(saliency, (self.width, self.height))
            channel_saliencies.append(saliency)
        
        # Average across channels
        return np.mean(channel_saliencies, axis=0)
    
    def compute_saliency_maps(self):
        """
        Compute saliency maps for all scales and create binary maps
        """
        print("Computing Multi-scale Saliency...")
        
        for scale in self.scales:
            print(f"  Processing scale: {scale}x{scale}")
            
            # Compute saliency for this scale
            saliency_map = self._compute_single_scale_saliency(scale)
            self.saliency_maps[scale] = saliency_map
            
            # Threshold to create binary map
            threshold = self.thresholds[scale]
            binary_map = (saliency_map > threshold).astype(np.float32)
            self.binary_maps[scale] = binary_map
            
            # Create integral image of binary map
            self.integral_images[scale] = IntegralImage(binary_map)
        
        print("✓ Multi-scale saliency computation complete")
    
    def set_thresholds(self, thresholds_dict):
        """Set learned thresholds for each scale"""
        for scale, threshold in thresholds_dict.items():
            if scale in self.thresholds:
                self.thresholds[scale] = threshold
    
    def get_window_density(self, window, scale):
        """Compute density of salient pixels in a window"""
        if scale not in self.integral_images:
            raise ValueError(f"Scale {scale} not computed. Run compute_saliency_maps() first.")
        
        x1, y1, x2, y2 = window
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Get integral image for this scale
        integral = self.integral_images[scale]
        
        # Calculate sum of salient pixels in window
        salient_sum = integral.rectangle_sum(x1, y1, x2, y2)
        
        # Calculate area
        area = (x2 - x1) * (y2 - y1)
        
        if area == 0:
            return 0.0
        
        # Density = (sum of salient pixels) / area
        return salient_sum / area
    
    def get_score(self, window):
        """
        Compute final MS score for a window (weighted sum across scales)
        """
        # If saliency maps haven't been computed, compute them
        if not self.integral_images:
            self.compute_saliency_maps()
        
        # Calculate density for each scale
        densities = {}
        for scale in self.scales:
            density = self.get_window_density(window, scale)
            densities[scale] = density
        
        # Weighted sum as per Eq 1 in the paper
        total_score = 0.0
        total_weight = 0.0
        
        for scale, weight in self.weights.items():
            total_score += weight * densities[scale]
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            total_score /= total_weight
        
        return total_score

# For testing directly
if __name__ == "__main__":
    print("Testing MS Cue directly...")
    # Create a test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    ms = MultiScaleSaliencyCue(test_img)
    ms.compute_saliency_maps()
    print("Test passed!")