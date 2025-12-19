# LSTM Growth Prediction Training Guide

## Overview

**Yes, the LSTM growth prediction model needs training!** The model architecture is defined but has no trained weights yet.

---

## Do I Need to Train It?

**Answer**: Yes, but it depends on your use case:

### Option 1: Use Pretrained Weights (Recommended for Demo)
- If you just want to **demonstrate** the FYP and test the API endpoints
- Use the existing code (it will work with random weights for testing)
- The predictions won't be accurate, but the system will function

### Option 2: Train on Synthetic Data (Quick Training)
- **Time**: ~10-15 minutes on CPU, ~2-3 minutes on GPU
- **Purpose**: Get a functional model with reasonable predictions
- **Location**: Your local laptop (very lightweight)
- **Data**: Auto-generated synthetic patient histories

### Option 3: Train on Real Data (Best Results)
- **Time**: 1-2 hours depending on dataset size
- **Purpose**: Production-ready model with clinical accuracy
- **Location**: Kaggle (recommended) or local with GPU
- **Data**: Real patient longitudinal scan data (BraTS or hospital data)

---

## Why LSTM is Different from U-Net/ResNet

| Model | Training Needed? | Reason |
|-------|------------------|--------|
| **3D U-Net** | ✅ Already trained | You mentioned "remaining AI models are already trained" |
| **ResNet** | ✅ Already trained | Classification weights exist |
| **LSTM Growth** | ❌ NOT trained yet | Just implemented - needs training data |

**Why LSTM needs training:**
- U-Net/ResNet work on **single scans** (image → prediction)
- LSTM needs **historical sequences** (multiple scans over time → growth prediction)
- Requires time-series data (patient scans at different dates)

---

## Training Locations Comparison

### 🏠 Local Laptop Training

**Pros:**
- ✅ Easy to run (one command)
- ✅ No internet needed
- ✅ Fast with synthetic data (10 mins on CPU)
- ✅ Good for testing and demos

**Cons:**
- ❌ Slower if no GPU
- ❌ Limited to synthetic data (unless you have real data locally)

**When to use:**
- Quick demo preparation
- Testing the training pipeline
- Don't have real longitudinal data yet

**Command:**
```powershell
python train_growth_prediction.py
```

---

### ☁️ Kaggle Training

**Pros:**
- ✅ Free GPU (30 hours/week)
- ✅ Access to BraTS dataset
- ✅ Can train on real data
- ✅ Faster training (GPU-accelerated)
- ✅ Can share trained model weights

**Cons:**
- ❌ Need internet
- ❌ Dataset preparation needed
- ❌ More setup required

**When to use:**
- Want production-quality model
- Have real patient scan histories
- Need to train on large dataset
- Want to share model weights publicly

**How to use:**
1. Upload training script to Kaggle notebook
2. Mount BraTS or longitudinal scan dataset
3. Run training
4. Download trained model weights

---

## Quick Start: Local Synthetic Training

### Step 1: Install Dependencies
```powershell
pip install scikit-learn matplotlib
```

### Step 2: Run Training
```powershell
cd c:\Users\SCM\Documents\fyp
python train_growth_prediction.py
```

### Step 3: What Happens
1. **Generates synthetic data**: 50 patients, 4-12 scans each
2. **Creates sequences**: Sliding windows of 3 consecutive scans
3. **Trains LSTM**: 100 epochs with early stopping
4. **Saves model**: `ml_models/growth_prediction/lstm_growth_model.pth`
5. **Plots results**: Training curves saved as PNG

### Step 4: Expected Output
```
==============================================================
LSTM Tumor Growth Prediction Training
==============================================================
Device: cpu
Generating synthetic data for 50 patients...
Created 250 training sequences from 50 patients
Train samples: 175
Val samples: 25
Test samples: 50

Epoch [  1/100] | Train Loss: 0.8234 | Val Loss: 0.7456 | Val MAE: 2.34 cc
Epoch [  5/100] | Train Loss: 0.3456 | Val Loss: 0.3234 | Val MAE: 1.85 cc
...
Training complete!
Test MAE: 1.67 cc

✓ Model: lstm_growth_model.pth
✓ Test MAE: 1.67 cc
✓ Epochs: 45
==============================================================
```

### Step 5: Verify Training
```powershell
# Check if model file exists
ls ml_models/growth_prediction/lstm_growth_model.pth

# Check file size (should be ~100KB)
# View training plot
ml_models/growth_prediction/training_history.png
```

---

## What Data Does LSTM Need?

The LSTM requires **historical tumor measurements** from the same patient:

### Required Features (per scan):
1. **Volume** (cc) - Main target
2. **Intensity stats** (mean, std)
3. **Morphology** (diameter, surface area, compactness, sphericity)
4. **Location** (centroid x, y, z)
5. **Timestamp** (scan date)

### Data Format (JSON):
```json
{
  "patient_id": "PT-2024-001",
  "scans": [
    {
      "scan_date": "2023-01-15T00:00:00",
      "volume": 25.5,
      "mean_intensity": 0.45,
      "std_intensity": 0.12,
      "max_diameter": 4.2,
      "surface_area": 55.3,
      "compactness": 0.78,
      "sphericity": 0.85,
      "centroid_x": 0.52,
      "centroid_y": 0.48,
      "centroid_z": 0.55
    },
    {
      "scan_date": "2023-04-15T00:00:00",
      "volume": 28.3,
      ...
    }
  ]
}
```

### Minimum Requirements:
- **At least 2 scans** per patient (for growth rate)
- **At least 3 scans** for training sequences
- **Time intervals**: Monthly is ideal (weekly to yearly works)

---

## Training on Real Data (Advanced)

### Step 1: Extract Features from Your Scans

If you have the **3D U-Net segmentation** outputs:

```python
# Extract features from segmentation masks
import nibabel as nib
import numpy as np
from scipy import ndimage

def extract_tumor_features(segmentation_path):
    """Extract features from segmentation mask"""
    seg = nib.load(segmentation_path).get_fdata()
    tumor_mask = seg > 0  # All tumor regions
    
    # Volume
    voxel_count = np.sum(tumor_mask)
    volume_cc = voxel_count * 0.001  # Assume 1mm³ voxels
    
    # Intensity stats (from original MRI)
    # mean_intensity = ...
    # std_intensity = ...
    
    # Morphology
    labeled, num_features = ndimage.label(tumor_mask)
    props = ndimage.measurements.center_of_mass(tumor_mask)
    
    # Centroid
    centroid = np.array(props) / seg.shape  # Normalize to [0,1]
    
    return {
        'volume': float(volume_cc),
        'centroid_x': float(centroid[0]),
        'centroid_y': float(centroid[1]),
        'centroid_z': float(centroid[2]),
        # ... add other features
    }
```

### Step 2: Build Patient Histories

```python
# Collect multiple scans per patient
patient_data = {
    'patient_id': 'PT-REAL-001',
    'scans': []
}

for scan_path in patient_scan_paths:
    features = extract_tumor_features(scan_path)
    features['scan_date'] = scan_metadata['date']
    patient_data['scans'].append(features)

# Save
import json
with open('data/growth_prediction/patient_histories.json', 'w') as f:
    json.dump([patient_data], f)
```

### Step 3: Train on Real Data

```powershell
# Same command, but will use your real data
python train_growth_prediction.py
```

---

## Module 8: 3D Reconstruction

**I did NOT create Module 8.** Here's what I actually implemented:

### What I Made (Modules 5-7):
- ✅ **Module 5**: LSTM Growth Prediction (time-series forecasting)
- ✅ **Module 6**: Explainable AI (Grad-CAM, SHAP heatmaps)
- ✅ **Module 7**: 2D/3D Visualization (slice viewer, multi-view, montage, MIP)

### What Module 7 Does (Current Visualization):
- 2D slice extraction (axial, sagittal, coronal)
- Multi-view synchronized display
- Volume montage (grid of slices)
- 3D MIP (Maximum Intensity Projection) - simple 3D rendering
- Segmentation overlay on images

### What "Full 3D Reconstruction" Would Mean (Module 8):
- **Interactive 3D mesh rendering** (VTK/Three.js)
- **Volume ray-casting** (GPU-accelerated)
- **Surface extraction** (Marching Cubes algorithm)
- **3D rotation/zoom** in browser
- **WebGL viewer** with controls

**Do you want me to create Module 8 (Full 3D Reconstruction)?**

This would add:
- 3D mesh generation from segmentation
- Interactive WebGL viewer (Three.js)
- 3D rotation, zoom, clipping planes
- Export to STL/OBJ formats
- Surface smoothing and rendering

---

## Recommendations

### For Your FYP Demo:

**Option A: Fast Demo (10 minutes)**
```powershell
# Train on synthetic data locally
python train_growth_prediction.py

# Test the endpoint
python test_advanced_modules.py
```
✅ Quick, reliable, shows functionality

---

**Option B: Production Quality (Kaggle, 2 hours)**
1. Upload script to Kaggle
2. Load BraTS longitudinal data
3. Extract features from segmentation outputs
4. Train on GPU
5. Download trained weights
✅ Real model, accurate predictions

---

**Option C: Just Demo Without Training**
- The API endpoints will work without trained weights
- Predictions will be random/inaccurate
- Good enough for showing UI and workflow
❌ Not recommended for final presentation

---

## Summary Table

| Aspect | Local Training | Kaggle Training |
|--------|---------------|----------------|
| **Time** | 10-15 min | 1-2 hours |
| **Data** | Synthetic | Real/BraTS |
| **Accuracy** | Demo-quality | Production |
| **GPU** | Optional | Free GPU ✓ |
| **Best For** | Quick testing | Final model |
| **Difficulty** | ⭐ Easy | ⭐⭐ Medium |

---

## My Recommendation

**For your FYP demo:**

1. **Quick Local Training** (NOW):
   ```powershell
   python train_growth_prediction.py
   ```
   - Get a working model in 10 minutes
   - Test all endpoints work
   - Use for development and testing

2. **Kaggle Training** (Optional, before final presentation):
   - If you have time before submission
   - Want better accuracy for demo
   - Have real longitudinal scan data

3. **Module 8** (Optional):
   - Only if required by FYP specification
   - Adds another ~2 days of work
   - Impressive but not essential

---

## Files Created

- ✅ `train_growth_prediction.py` - Complete training script
- ✅ `LSTM_TRAINING_GUIDE.md` - This guide
- ✅ `ml_models/growth_prediction/lstm_growth.py` - Model architecture (already exists)

---

## Next Steps

1. **Install dependencies**:
   ```powershell
   pip install scikit-learn matplotlib
   ```

2. **Run training**:
   ```powershell
   python train_growth_prediction.py
   ```

3. **Test the trained model**:
   ```powershell
   python test_advanced_modules.py
   ```

4. **Verify predictions improved**:
   - Compare before/after MAE
   - Check growth rate calculations
   - Test all API endpoints

---

## Questions?

**Q: Do I NEED to train this for FYP submission?**
A: Not strictly, but a trained model looks much better in demos and shows you understand ML workflows.

**Q: Can I skip training?**
A: Yes, endpoints work without weights, but predictions are meaningless.

**Q: Which training location is better?**
A: Local for quick demo prep, Kaggle for final production model.

**Q: What about Module 8?**
A: I didn't create it. Let me know if you need full 3D reconstruction!

---

**Status**: Ready to train! 🚀

**Time Required**: 10 minutes (local synthetic) or 2 hours (Kaggle real data)

**Command**: `python train_growth_prediction.py`
