# Generic Objectness Estimation (Group A)
**Student:** Abdul Moiz (BSCS22031)  
**Group:** A (Odd Roll Number)

## 📌 Project Overview
This project implements a **Generic Objectness** measure to quantify how likely an image window is to contain an object of any class. Unlike specific object detectors (like YOLO for "cars"), this system scores the "object-like" quality of windows based on visual cues.

The system combines three complementary cues:
1.  **Multi-scale Saliency (MS)**: Uses spectral residual analysis in the frequency domain to find unique/salient regions.
2.  **Superpixels Straddling (SS)**: Penalizes windows that "straddle" or cut through superpixels (uniform regions).
3.  **Edge Density (ED)**: (Group A Specific) Measures the concentration of edges near window borders.

## 🚀 Key Features
*   **Real-time O(1) Scoring**: All cues utilize **Integral Images** (Summed Area Tables) to calculate scores for any window size in constant time.
*   **Bayesian Parameter Learning**: Optimal parameters (e.g., edge border ratio, superpixel count) are learned by maximizing the **KL Divergence** between Object and Background score distributions on the PASCAL VOC dataset.
*   **Smart Filtering**: Implements **Non-Maximum Suppression (NMS)** and Score Thresholding to remove redundant and background bounding boxes.

## 📂 Project Structure
```text
BSCS22031_AbdulMoiz_06/
├── modules/               # Core algorithms
│   ├── objectness_detector.py # Main orchestrator (incl. NMS)
│   ├── ms_cue.py         # Multi-scale Saliency logic
│   ├── ss_cue.py         # Superpixel Straddling logic
│   ├── ed_cue.py         # Edge Density logic
│   ├── integral_image.py # O(1) sum optimization
│   └── bayesian_learning.py # Parameter optimization
├── main_group_a.py        # Main execution script
├── learn_bayesian.py      # Training script
├── generate_report.py     # PDF report generator
├── data/                  # Dataset folder
└── assignment_report/     # Final generated PDF report
```

## 🛠️ How to Run

### 1. Setup Environment
Enter the project directory and install dependencies:
```bash
cd BSCS22031_AbdulMoiz_06
pip install -r requirements.txt
```

### 2. Parameter Learning (Optional)
To relearn parameters from the PASCAL VOC dataset:
```bash
python learn_bayesian.py --dataset data/VOCdevkit/VOC2007 --output learned_params
```
*(Note: Pre-learned parameters are already included).*

### 3. Run Object Detection
To detect objects in your own images:
```bash
python main_group_a.py --input sample_images --output results
```
*   **Input**: Directory of images.
*   **Output**: Visualizations and text scores will be saved to `results/`.
*   **NMS**: Automatic filtering is enabled by default.

### 4. Generate Report
To generate the comprehensive PDF report with diagrams and results:
```bash
python generate_report.py
```
This will create `assignment_report/assignment_report.pdf`.

## 📊 Methodology Highlights
*   **MS Cue**: Computes FFT -> Log Spectrum -> Spectral Residual -> IFFT.
*   **SS Cue**: Uses SLIC segmentation -> Integral Images of masks -> Straddling Penalty.
*   **ED Cue**: Uses Canny Edges -> Integral Image -> Border Density vs Inner Density.
*   **Integration**: Final Score = $w_{MS} \cdot S_{MS} + w_{SS} \cdot S_{SS} + w_{ED} \cdot S_{ED}$
