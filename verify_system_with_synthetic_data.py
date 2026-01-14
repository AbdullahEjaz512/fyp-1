
import os
import sys
import torch
import numpy as np
import nibabel as nib
import json
from pathlib import Path
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.segmentation.unet3d import UNet3D
from ml_models.classification.resnet_classifier import ResNetClassifier
# Try importing Growth Service, handle dependencies
try:
    from ml_models.growth_prediction.lstm_growth import GrowthPredictionService
    HAS_GROWTH = True
except ImportError:
    HAS_GROWTH = False

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def generate_synthetic_mri(path="synthetic_mri.nii.gz", shape=(128, 128, 128)):
    print_header("1. Data Generation (Simulating 4-Channel MRI)")
    print("Generating synthetic brain volume with tumor...")
    
    # 4 channels: T1, T1ce, T2, FLAIR
    channels = 4
    data = np.zeros((channels,) + shape, dtype=np.float32)
    
    # Create a "brain" (large sphere)
    c = shape[0] // 2
    r_brain = 50
    y, x, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask_brain = ((x - c)**2 + (y - c)**2 + (z - c)**2) <= r_brain**2
    
    # Fill brain tissue (Channel 0: T1 - uniform grey)
    data[0][mask_brain] = 0.6
    # T2 (Watery/bright)
    data[2][mask_brain] = 0.5
    
    # Create a "tumor" (small sphere offset from center)
    t_c = c + 15
    r_tumor = 10
    mask_tumor = ((x - t_c)**2 + (y - t_c)**2 + (z - c)**2) <= r_tumor**2
    
    # Tumor Characteristics
    # T1: Hypointense (Dark)
    data[0][mask_tumor] = 0.2
    # T1ce: Ring enhancement (Bright rim, dark core)
    mask_rim = mask_tumor & ~(((x - t_c)**2 + (y - t_c)**2 + (z - c)**2) <= (r_tumor-2)**2)
    data[1][mask_rim] = 0.9
    # T2: Hyperintense (Bright)
    data[2][mask_tumor] = 0.9
    # FLAIR: Hyperintense
    data[3][mask_tumor] = 0.95
    
    # Save as NIfTI
    # Standard NIfTI expects (H, W, D, C) or (C, H, W, D) depending on convention.
    # Our model expects (C, H, W, D). NifTI usually stores spatial then time/channel.
    # We will save as (H, W, D, C) for standard viewing, but transpose for model.
    save_data = data.transpose(1, 2, 3, 0) # -> H, W, D, C
    affine = np.eye(4)
    img = nib.Nifti1Image(save_data, affine)
    nib.save(img, path)
    
    print(f"✓ Synthetic scan saved to: {path}")
    print(f"✓ Dimensions: {save_data.shape} (Channels last)")
    return data, path

def test_segmentation(data_tensor, model_path):
    print_header("2. Testing Segmentation (U-Net)")
    
    if not os.path.exists(model_path):
        print(f"❌ Model missing: {model_path}")
        return None
        
    try:
        model = UNet3D()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Prepare input: (1, 4, 128, 128, 128)
        input_tensor = torch.from_numpy(data_tensor).unsqueeze(0)
        
        print("Running inference...")
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            
        tumor_voxels = (pred > 0).sum().item()
        total_voxels = pred.numel()
        tumor_vol_cc = tumor_voxels * 0.001 # Assuming 1mmiso res
        
        print(f"✓ Inference successful")
        print(f"✓ Output shape: {pred.shape}")
        print(f"✓ Detected Tumor Volume: {tumor_vol_cc:.2f} cubic units")
        
        if tumor_voxels > 0:
            print(f"✅ Tumor detected! ({tumor_voxels} voxels)")
        else:
            print(f"⚠️ No tumor detected (Check noise/intensity thresholds)")
            
        return tumor_vol_cc
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return None

def test_classification(data_tensor, model_path):
    print_header("3. Testing Classification (ResNet50)")
    
    if not os.path.exists(model_path):
        print(f"❌ Model missing: {model_path}")
        return None
        
    try:
        model = ResNetClassifier(num_classes=4, in_channels=4, pretrained=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle state dict wrapping
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        
        # Extract middle slice for 2D classification: (128, 128) -> Resize to (224, 224)
        # Input data is (4, 128, 128, 128)
        mid_slice_idx = data_tensor.shape[3] // 2
        # Slice: (4, 128, 128)
        slice_2d = torch.from_numpy(data_tensor[:, :, :, mid_slice_idx])
        
        # Resize to (224, 224)
        slice_resized = F.interpolate(
            slice_2d.unsqueeze(0), 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        print(f"Input slice shape: {slice_resized.shape}")
        
        with torch.no_grad():
            output = model(slice_resized)
            # Handle model return type (tuple vs tensor)
            if isinstance(output, tuple):
                logits = output[0]
                conf = output[1] if len(output) > 1 else torch.softmax(logits, dim=1)
            else:
                logits = output
                conf = torch.softmax(logits, dim=1)
                
            pred_idx = torch.argmax(logits, dim=1).item()
            confidence = conf[0].tolist()
            
        classes = ["GBM", "LGG", "Meningioma", "Other"]
        pred_label = classes[pred_idx] if pred_idx < len(classes) else "Unknown"
        
        print(f"✓ Inference successful")
        print(f"✓ Predicted Class: {pred_label} (Index {pred_idx})")
        print(f"✓ Confidence: {max(confidence):.4f}")
        print(f"✅ Classification pipeline functional")
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")

def test_growth(current_volume):
    print_header("4. Testing Growth Prediction")
    
    if not HAS_GROWTH:
        print("⚠️ Growth service dependencies missing or not imported.")
        return

    if current_volume is None:
        print("⚠️ Skipping growth test (No current volume)")
        return
        
    try:
        service = GrowthPredictionService()
        
        # Create dummy history leading up to current volume
        history = [
            {'volume': current_volume * 0.8, 'timestamp': '2024-01-01', 'mean_intensity': 0.5},
            {'volume': current_volume * 0.9, 'timestamp': '2024-03-01', 'mean_intensity': 0.52},
            # Current scan would be the 3rd point normally
        ]
        
        print(f"Simulating history: {history[0]['volume']:.2f} -> {history[1]['volume']:.2f}")
        print(f"Current volume: {current_volume:.2f}")
        
        # We append current hypothetical measurement
        current = {'volume': current_volume, 'timestamp': '2024-05-01', 'mean_intensity': 0.55}
        history.append(current)
        
        result = service.predict_growth(history)
        
        print(f"✓ Prediction Result: {result['predictions']}")
        print(f"✓ Growth Rate: {result['growth_rate']:.2f}%")
        print(f"✅ Growth prediction functional")
        
    except Exception as e:
        print(f"❌ Growth prediction failed: {e}")

def main():
    # Paths
    seg_model = 'ml_models/segmentation/unet_model.pth'
    cls_model = 'ml_models/classification/resnet_model.pth'
    
    # 1. Generate
    data, nii_path = generate_synthetic_mri()
    
    # 2. Segment
    vol = test_segmentation(data, seg_model)
    
    # 3. Classify
    test_classification(data, cls_model)
    
    # 4. Predict Growth
    test_growth(vol)
    
    # Clean up
    if os.path.exists(nii_path):
        os.remove(nii_path)
        print("\n(Cleaned up synthetic file)")

if __name__ == "__main__":
    main()
