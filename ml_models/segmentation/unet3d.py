"""
Module 3: Tumor Segmentation using 3D U-Net
Implements FR9.1 to FR9.8 - Automatic tumor segmentation
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
from datetime import datetime

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import monai
    from monai.networks.nets import UNet
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        Spacingd, Orientationd, ScaleIntensityRanged,
        CropForegroundd, RandCropByPosNegLabeld,
        RandFlipd, RandRotate90d, EnsureTyped
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/MONAI not yet installed")

try:
    from config import (
        SEGMENTATION_MODEL,
        SEGMENTATION_TRAINING,
        DEVICE,
        TUMOR_REGIONS,
        REGION_LABELS
    )
except ImportError:
    # Fallback configuration
    DEVICE = "cpu"
    SEGMENTATION_MODEL = {
        "in_channels": 4,
        "out_channels": 4,
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
    }
    TUMOR_REGIONS = {
        0: "Background",
        1: "NCR/NET",
        2: "ED",
        3: "ET"
    }


class UNet3D(nn.Module):
    """
    3D U-Net Model for Brain Tumor Segmentation - FR9.1
    Based on MONAI's UNet architecture
    
    Architecture:
    - Encoder: Downsampling path with residual connections
    - Bottleneck: Deepest layer with highest feature dimension
    - Decoder: Upsampling path with skip connections
    - Output: Multi-class segmentation (4 channels)
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # 4 MRI modalities
        out_channels: int = 4,  # Background + 3 tumor regions
        channels: Tuple = (32, 64, 128, 256, 320),  # BraTS standard config
        strides: Tuple = (2, 2, 2, 2),
        dropout: float = 0.0
    ):
        super(UNet3D, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and MONAI required for 3D U-Net")
        
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=2,
            dropout=dropout,
            norm='instance',  # Instance norm - matches Kaggle training
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BraTSDataset(Dataset):
    """
    BraTS Dataset Loader - FR9.1
    Loads multi-modal MRI scans and segmentation masks
    """
    
    def __init__(
        self,
        data_dir: Path,
        transform=None,
        modalities: List[str] = None,
        cache_data: bool = False
    ):
        """
        Args:
            data_dir: Path to BraTS dataset directory
            transform: MONAI transforms to apply
            modalities: List of modalities to load (default: all 4)
            cache_data: Whether to cache data in memory
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.modalities = modalities or ["t1n", "t1c", "t2w", "t2f"]
        self.cache_data = cache_data
        
        # Find all patient cases
        self.cases = self._find_cases()
        print(f"Found {len(self.cases)} cases in {data_dir}")
        
        # Cache for preprocessed data
        self.cache = {} if cache_data else None
    
    def _find_cases(self) -> List[Path]:
        """Find all patient case directories"""
        cases = []
        
        if self.data_dir.exists():
            # BraTS format: each case is a subdirectory
            for case_dir in self.data_dir.iterdir():
                if case_dir.is_dir() and case_dir.name.startswith("BraTS"):
                    cases.append(case_dir)
        
        return sorted(cases)
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        """Load a single case with all modalities"""
        case_dir = self.cases[idx]
        
        # Check cache
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        # Build file paths for each modality
        data_dict = {"image": []}
        
        for modality in self.modalities:
            # BraTS naming convention: BraTS-XXX-XXXXX-XXX-{modality}.nii.gz
            modality_files = list(case_dir.glob(f"*{modality}*.nii.gz"))
            
            if modality_files:
                data_dict["image"].append(str(modality_files[0]))
        
        # Load segmentation mask if it exists
        seg_files = list(case_dir.glob("*seg.nii.gz"))
        if seg_files:
            data_dict["label"] = str(seg_files[0])
        
        # Apply transforms
        if self.transform:
            data_dict = self.transform(data_dict)
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = data_dict
        
        return data_dict


class SegmentationTrainer:
    """
    Training Pipeline for 3D U-Net - FR9.1
    Handles training, validation, and model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        config: Dict = None
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config or SEGMENTATION_TRAINING
        
        # Loss function - Dice + Cross Entropy as per SRS
        self.loss_function = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Metrics
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "learning_rates": []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.loss_function(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        return epoch_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model - FR9.3, FR9.6"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.loss_function(outputs, labels)
                val_loss += loss.item()
                
                # Calculate Dice score
                self.dice_metric(y_pred=outputs, y=labels)
        
        val_loss /= len(val_loader)
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        
        return val_loss, dice_score
    
    def save_checkpoint(self, filepath: Path, epoch: int, best_metric: float):
        """Save model checkpoint - FR9.4"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": best_metric,
            "history": self.history,
            "config": self.config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint


class TumorSegmentationInference:
    """
    Inference Engine for Tumor Segmentation - FR9.2, FR9.3
    Performs prediction on new MRI scans with optional Test-Time Augmentation (TTA)
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda", use_tta: bool = True):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.use_tta = use_tta  # Enable TTA by default for better accuracy
    
    def predict(
        self, 
        inputs: torch.Tensor,
        return_probabilities: bool = False,
        use_tta: bool = None  # Override instance setting
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict tumor segmentation - FR9.2
        
        Args:
            inputs: Input tensor [B, C, D, H, W]
            return_probabilities: Whether to return class probabilities
            use_tta: Whether to use Test-Time Augmentation (improves accuracy by ~3-4%)
            
        Returns:
            Segmentation mask and optionally probabilities
        """
        # Determine whether to use TTA
        apply_tta = use_tta if use_tta is not None else self.use_tta
        
        with torch.no_grad():
            if apply_tta:
                # Test-Time Augmentation: average predictions from flipped inputs
                outputs = self._predict_with_tta(inputs)
            else:
                outputs = self.model(inputs)
            
            if return_probabilities:
                # Softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                return preds, probs
            else:
                preds = torch.argmax(outputs, dim=1)
                return preds, None
    
    def _predict_with_tta(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Test-Time Augmentation: Average predictions from multiple augmented versions.
        This typically improves Dice score by 2-4% without any retraining.
        
        Augmentations applied:
        - Original
        - Flip along depth (axis 2)
        - Flip along height (axis 3)  
        - Flip along width (axis 4)
        - Flip along depth + height
        - Flip along depth + width
        - Flip along height + width
        - Flip along all three axes
        """
        predictions = []
        
        # 1. Original prediction
        pred_orig = self.model(inputs)
        predictions.append(pred_orig)
        
        # 2. Flip along depth (axis 2 in BCDHW format)
        flipped_d = torch.flip(inputs, [2])
        pred_d = torch.flip(self.model(flipped_d), [2])
        predictions.append(pred_d)
        
        # 3. Flip along height (axis 3)
        flipped_h = torch.flip(inputs, [3])
        pred_h = torch.flip(self.model(flipped_h), [3])
        predictions.append(pred_h)
        
        # 4. Flip along width (axis 4)
        flipped_w = torch.flip(inputs, [4])
        pred_w = torch.flip(self.model(flipped_w), [4])
        predictions.append(pred_w)
        
        # 5. Flip along depth + height
        flipped_dh = torch.flip(inputs, [2, 3])
        pred_dh = torch.flip(self.model(flipped_dh), [2, 3])
        predictions.append(pred_dh)
        
        # 6. Flip along depth + width
        flipped_dw = torch.flip(inputs, [2, 4])
        pred_dw = torch.flip(self.model(flipped_dw), [2, 4])
        predictions.append(pred_dw)
        
        # 7. Flip along height + width
        flipped_hw = torch.flip(inputs, [3, 4])
        pred_hw = torch.flip(self.model(flipped_hw), [3, 4])
        predictions.append(pred_hw)
        
        # 8. Flip along all three axes
        flipped_all = torch.flip(inputs, [2, 3, 4])
        pred_all = torch.flip(self.model(flipped_all), [2, 3, 4])
        predictions.append(pred_all)
        
        # Average all predictions (logits)
        avg_prediction = torch.stack(predictions, dim=0).mean(dim=0)
        
        return avg_prediction
    
    def predict_with_postprocessing(
        self,
        inputs: torch.Tensor,
        min_component_size: int = 100,
        return_probabilities: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Predict with TTA and post-processing for best accuracy.
        
        Post-processing steps:
        1. Test-Time Augmentation (averaging flipped predictions)
        2. Remove small connected components (noise removal)
        
        Args:
            inputs: Input tensor [B, C, D, H, W]
            min_component_size: Minimum voxel count to keep a component
            return_probabilities: Whether to return probabilities
            
        Returns:
            Tuple of (segmentation, probabilities, postprocessing_info)
        """
        from scipy import ndimage
        
        # Get TTA prediction
        preds, probs = self.predict(inputs, return_probabilities=True, use_tta=True)
        
        # Post-processing: remove small components for each class
        preds_np = preds.cpu().numpy()
        removed_components = {}
        
        for batch_idx in range(preds_np.shape[0]):
            batch_mask = preds_np[batch_idx]
            
            for class_idx in range(1, 4):  # Skip background (0)
                class_mask = (batch_mask == class_idx).astype(np.int32)
                
                if class_mask.sum() > 0:
                    # Label connected components
                    labeled, num_features = ndimage.label(class_mask)
                    
                    components_removed = 0
                    for i in range(1, num_features + 1):
                        component_size = np.sum(labeled == i)
                        if component_size < min_component_size:
                            # Remove small component - set to background
                            batch_mask[labeled == i] = 0
                            components_removed += 1
                    
                    removed_components[f"class_{class_idx}"] = components_removed
            
            preds_np[batch_idx] = batch_mask
        
        # Convert back to tensor
        preds_cleaned = torch.from_numpy(preds_np).to(self.device)
        
        postprocessing_info = {
            "tta_enabled": True,
            "num_augmentations": 8,
            "min_component_size": min_component_size,
            "removed_components": removed_components
        }
        
        if return_probabilities:
            return preds_cleaned, probs, postprocessing_info
        else:
            return preds_cleaned, None, postprocessing_info
    
    def calculate_confidence(self, probabilities: torch.Tensor) -> Dict[str, float]:
        """
        Calculate confidence scores for each tumor region - FR9.3
        
        Returns:
            Dictionary with confidence scores for each region
        """
        confidence_scores = {}
        
        for region_idx, region_name in TUMOR_REGIONS.items():
            if region_idx == 0:  # Skip background
                continue
            
            # Get max probability for this class
            region_probs = probabilities[:, region_idx, ...].flatten()
            max_conf = torch.max(region_probs).item()
            mean_conf = torch.mean(region_probs).item()
            
            confidence_scores[region_name] = {
                "max_confidence": max_conf * 100,  # Convert to percentage
                "mean_confidence": mean_conf * 100
            }
        
        return confidence_scores
    
    def calculate_tumor_volumes(
        self, 
        segmentation: torch.Tensor,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        """
        Calculate tumor volumes for each region - FR9.6
        
        Args:
            segmentation: Segmentation mask
            voxel_spacing: Voxel spacing in mm (x, y, z)
            
        Returns:
            Dictionary with volume measurements in mm³
        """
        voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
        
        volumes = {}
        seg_np = segmentation.cpu().numpy()
        
        for region_idx, region_name in TUMOR_REGIONS.items():
            if region_idx == 0:  # Skip background
                continue
            
            # Count voxels for this region
            voxel_count = np.sum(seg_np == region_idx)
            volume_mm3 = voxel_count * voxel_volume
            volume_cm3 = volume_mm3 / 1000  # Convert to cm³
            
            volumes[region_name] = {
                "voxel_count": int(voxel_count),
                "volume_mm3": float(volume_mm3),
                "volume_cm3": float(volume_cm3)
            }
        
        # Calculate combined regions as per BraTS
        # Whole Tumor (WT) = ED + NCR/NET + ET
        # Tumor Core (TC) = NCR/NET + ET
        # Enhancing Tumor (ET) = already calculated
        
        return volumes


if __name__ == "__main__":
    print("=" * 60)
    print("Module 3: Tumor Segmentation - 3D U-Net")
    print("=" * 60)
    
    if TORCH_AVAILABLE:
        print(f"\n✓ PyTorch/MONAI available")
        print(f"  Device: {DEVICE}")
        
        # Initialize model
        print("\n[1] Initializing 3D U-Net model...")
        model = UNet3D(**SEGMENTATION_MODEL)
        num_params = model.get_num_parameters()
        print(f"  ✓ Model created with {num_params:,} parameters")
        
        # Test forward pass
        print("\n[2] Testing forward pass...")
        batch_size = 1
        dummy_input = torch.randn(batch_size, 4, 128, 128, 128)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  ✓ Input shape: {dummy_input.shape}")
        print(f"  ✓ Output shape: {output.shape}")
        
        print("\n✅ Module 3 initialization successful!")
    else:
        print("\n⚠ PyTorch/MONAI not installed yet")
        print("  Install with: pip install torch monai")
