# test_filtering.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.objectness_detector import ObjectnessDetector
import os

def generate_synthetic_image(seed=42):
    """Generate a synthetic image with multiple distinct objects"""
    np.random.seed(seed)
    h, w = 400, 600
    image = np.ones((h, w, 3), dtype=np.uint8) * 240  # Light background
    
    # Add random background noise/lines
    for _ in range(50):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        color = np.random.randint(200, 255, 3).tolist()
        cv2.line(image, (x1, y1), (x2, y2), color, 1)
        
    # Create distinct objects
    objects = []
    
    # 1. Red Rectangle
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 255), -1)
    objects.append("Red Box")
    
    # 2. Green Circle
    cv2.circle(image, (300, 100), 40, (0, 255, 0), -1)
    objects.append("Green Circle")
    
    # 3. Blue Complex Shape (Triangle)
    pts = np.array([[450, 150], [500, 50], [550, 150]], np.int32)
    cv2.fillPoly(image, [pts], (255, 0, 0))
    objects.append("Blue Triangle")
    
    # 4. Yellow Textured Box (Bottom Left)
    cv2.rectangle(image, (100, 250), (200, 350), (0, 255, 255), -1)
    # Add texture to grid
    for i in range(100, 200, 10):
        cv2.line(image, (i, 250), (i, 350), (0, 0, 0), 1)
    objects.append("Textured Box")

    # 5. Small Purple Square (Bottom Right)
    cv2.rectangle(image, (400, 300), (440, 340), (255, 0, 255), -1)
    objects.append("Purple Square")
    
    return image

def calculate_iou(boxA, boxB):
    """Calculate IoU between two windows"""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes
    
    Args:
        boxes: List of (x1, y1, x2, y2)
        scores: List of scores corresponding to boxes
        iou_threshold: Overlap threshold for suppression
    
    Returns:
        List of selected indices
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays for vectorization if needed, 
    # but using simple list comprehension here for clarity
    
    # Sort indices by score (descending)
    idxs = np.argsort(scores)[::-1]
    
    pick = []
    
    while len(idxs) > 0:
        # Pick the current highest score box
        last = idxs.shape[0]
        i = idxs[0]
        pick.append(i)
        
        # Find IoU of this box with all others
        ious = []
        for j in range(1, last):
            iou = calculate_iou(boxes[i], boxes[idxs[j]])
            ious.append(iou)
        
        # Keep only boxes with IoU less than threshold
        # (remove boxes that overlap significantly with prediction)
        idxs = np.delete(idxs, np.concatenate(([0], np.where(np.array(ious) > iou_threshold)[0] + 1)))
        
    return pick

def filter_windows(windows, score_threshold=0.0, nms_threshold=0.3):
    """
    Filter windows based on score and NMS
    
    Args:
        windows: List of (window, total, ms, ss, ed) tuples
    """
    # 1. Score Thresholding
    kept_windows = [w for w in windows if w[1] > score_threshold]
    
    if not kept_windows:
        return []
        
    # Extract boxes and scores for NMS
    boxes = [w[0] for w in kept_windows]
    scores = [w[1] for w in kept_windows]
    
    # 2. Non-Maximum Suppression
    pick_indices = non_max_suppression(boxes, scores, nms_threshold)
    
    final_windows = [kept_windows[i] for i in pick_indices]
    
    return final_windows

def visualize_filtering_steps(image, raw_windows, filtered_windows, title, output_path):
    """Visualize raw vs filtered results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw Results (Top 50)
    vis_raw = image.copy()
    raw_sorted = sorted(raw_windows, key=lambda x: x[1], reverse=True)[:50]
    
    for i, (window, score, _, _, _) in enumerate(raw_sorted):
        # Color goes from Red (high) to Blue (low)
        color = (0, 0, 255) if i > 10 else (0, 255, 0)
        alpha = 0.5
        x1, y1, x2, y2 = window
        cv2.rectangle(vis_raw, (x1, y1), (x2, y2), color, 2)
    
    axes[0].imshow(cv2.cvtColor(vis_raw, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Raw Top-50 Windows\n(Many redundant boxes)")
    axes[0].axis('off')
    
    # Filtered Results
    vis_filtered = image.copy()
    for i, (window, score, _, _, _) in enumerate(filtered_windows):
        x1, y1, x2, y2 = window
        # Draw nice bounding box
        cv2.rectangle(vis_filtered, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Label
        label = f"Obj: {score:.2f}"
        cv2.putText(vis_filtered, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    axes[1].imshow(cv2.cvtColor(vis_filtered, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Filtered Results (NMS + Threshold)\n({len(filtered_windows)} distinct objects)")
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")

def run_experiment():
    print("="*60)
    print("BOUNDING BOX FILTERING EXPERIMENT")
    print("============================================================")
    
    # Create output directory
    os.makedirs("filtering_results", exist_ok=True)
    
    # Initialize detector with default config
    # We increase top_k to get MANY candidates for NMS to filter
    config_params = {
        'top_k_windows': 200,  # Get many windows to demonstrate filtering
        'ms_weight': 0.4,
        'ss_weight': 0.3,
        'ed_weight': 0.3
    }
    
    detector = ObjectnessDetector()
    detector.params.update(config_params)
    
    # EXPERIMENT 1: Synthetic Image
    print("\nExperiment 1: Synthetic Shapes")
    img1 = generate_synthetic_image(seed=42)
    
    # Get raw windows
    windows1, _ = detector.process_image(img1)
    
    # Apply filtering
    print(f"  Raw windows: {len(windows1)}")
    
    # Strict NMS
    filtered1 = filter_windows(windows1, score_threshold=0.4, nms_threshold=0.1)
    print(f"  Filtered windows: {len(filtered1)}")
    
    visualize_filtering_steps(img1, windows1, filtered1, 
                            "Filtering on Synthetic Shapes", 
                            "filtering_results/exp1_synthetic.jpg")
    
    # EXPERIMENT 2: Another Variation
    print("\nExperiment 2: Noisy Background")
    img2 = generate_synthetic_image(seed=100)
    # Add noise
    noise = np.random.normal(0, 25, img2.shape).astype(np.uint8)
    img2 = cv2.add(img2, noise)
    
    windows2, _ = detector.process_image(img2)
    filtered2 = filter_windows(windows2, score_threshold=0.35, nms_threshold=0.15)
    
    visualize_filtering_steps(img2, windows2, filtered2, 
                            "Filtering on Noisy Image", 
                            "filtering_results/exp2_noisy.jpg")
                            
    print("\n" + "="*60)
    print("Check 'filtering_results/' for output images")

if __name__ == "__main__":
    run_experiment()
