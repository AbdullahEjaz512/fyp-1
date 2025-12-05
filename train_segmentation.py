"""
Training Script for 3D U-Net Segmentation
Full training from scratch on BraTS dataset
"""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import numpy as np

from ml_models.segmentation.unet3d import (
    UNet3D,
    BraTSDataset,
    SegmentationTrainer
)
from config import (
    SEGMENTATION_MODEL,
    SEGMENTATION_TRAINING,
    DEVICE,
    BRATS_DATASET_PATH,
    PREPROCESSING
)

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, Orientationd, ScaleIntensityRanged,
    CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped,
    RandShiftIntensityd, RandAdjustContrastd
)


def get_train_transforms():
    """Get training transforms with data augmentation"""
    return Compose([
        # Load images
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Spatial transforms
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=PREPROCESSING["target_spacing"],
            mode=("bilinear", "nearest")
        ),
        
        # Intensity normalization
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        
        # Crop foreground
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # Random crop patches (128x128x128)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PREPROCESSING["target_size"],
            pos=1,  # Positive samples ratio
            neg=1,  # Negative samples ratio
            num_samples=4  # Number of patches per image
        ),
        
        # Data augmentation (random)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
        
        # Intensity augmentation
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.5),
        
        # Convert to tensors
        EnsureTyped(keys=["image", "label"])
    ])


def get_val_transforms():
    """Get validation transforms (no augmentation)"""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=PREPROCESSING["target_spacing"],
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"])
    ])


def train_unet_segmentation():
    """
    Train 3D U-Net from scratch
    This requires:
    - BraTS dataset at the path specified in config.py
    - ~12-16GB GPU memory (or use smaller batch size)
    - ~2-3 days of training time (300 epochs)
    """
    
    print("=" * 70)
    print("3D U-Net Brain Tumor Segmentation Training")
    print("=" * 70)
    
    # ============================================================
    # STAGE 0: Setup
    # ============================================================
    print("\n[Stage 0] Setup...")
    
    # Check if BraTS dataset exists
    if not BRATS_DATASET_PATH.exists():
        print(f"\n⚠ BraTS dataset not found at {BRATS_DATASET_PATH}")
        print("\nTo download BraTS dataset:")
        print("1. Register at: https://www.synapse.org/#!Synapse:syn51514132")
        print("2. Download BraTS 2023 Training Data")
        print("3. Extract to: D:/BraTs 2023/")
        print("4. Update BRATS_DATASET_PATH in config.py if needed")
        return
    
    # Create model
    model = UNet3D(**SEGMENTATION_MODEL)
    print(f"✓ Model initialized with {model.get_num_parameters():,} parameters")
    print(f"✓ Using device: {DEVICE}")
    
    if DEVICE.type == "cpu":
        print("\n⚠ WARNING: Training on CPU will be EXTREMELY slow!")
        print("  Recommended: Use GPU with at least 12GB VRAM")
        print("  Or use Google Colab/Kaggle for free GPU access")
    
    # ============================================================
    # STAGE 1: Load Dataset
    # ============================================================
    print("\n[Stage 1] Loading BraTS dataset...")
    
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    
    # Load full dataset
    print("Loading training data (this may take a few minutes)...")
    full_dataset = BraTSDataset(
        data_dir=BRATS_DATASET_PATH,
        transform=train_transforms,
        cache_data=False  # Set True if you have enough RAM (>32GB)
    )
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=SEGMENTATION_TRAINING["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Validate with full volumes
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # ============================================================
    # STAGE 2: Training
    # ============================================================
    print("\n[Stage 2] Training...")
    print("=" * 70)
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        device=DEVICE,
        config=SEGMENTATION_TRAINING
    )
    
    # Training loop
    best_dice = 0
    patience_counter = 0
    num_epochs = SEGMENTATION_TRAINING["num_epochs"]
    early_stopping_patience = SEGMENTATION_TRAINING["early_stopping_patience"]
    
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Batch size: {SEGMENTATION_TRAINING['batch_size']}")
    print(f"Learning rate: {SEGMENTATION_TRAINING['learning_rate']}")
    print("\nThis will take approximately 2-3 days on a modern GPU.\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train one epoch
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate every 5 epochs (saves time)
        if (epoch + 1) % 5 == 0:
            val_loss, val_dice = trainer.validate(val_loader)
            
            # Update learning rate scheduler
            trainer.scheduler.step(val_loss)
            
            # Save history
            trainer.history["train_loss"].append(train_loss)
            trainer.history["val_loss"].append(val_loss)
            trainer.history["val_dice"].append(val_dice)
            trainer.history["learning_rates"].append(
                trainer.optimizer.param_groups[0]['lr']
            )
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                
                checkpoint_path = Path("ml_models/segmentation/checkpoints/best_model.pth")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(checkpoint_path, epoch, best_dice)
                print(f"  ✓ Best model saved (Dice: {val_dice:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                break
        
        else:
            # Just train, no validation
            print(f"  Train Loss: {train_loss:.4f}")
            trainer.history["train_loss"].append(train_loss)
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = Path(f"ml_models/segmentation/checkpoints/epoch_{epoch+1}.pth")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(checkpoint_path, epoch, best_dice)
            print(f"  ✓ Checkpoint saved: epoch_{epoch+1}.pth")
    
    # ============================================================
    # FINAL: Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"✓ Best Dice score: {best_dice:.4f}")
    print(f"✓ Total epochs: {epoch+1}")
    print(f"✓ Model saved to: ml_models/segmentation/checkpoints/best_model.pth")
    print("\nTo use in production:")
    print("  1. Copy best_model.pth to ml_models/segmentation/unet_model.pth")
    print("  2. Update backend to load this checkpoint")
    print("  3. Test with: python test_segmentation.py")
    
    # Save training history
    history_path = Path("ml_models/segmentation/checkpoints/training_history.npy")
    np.save(history_path, trainer.history)
    print(f"✓ Training history saved to: {history_path}")


if __name__ == "__main__":
    train_unet_segmentation()
