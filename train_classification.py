
"""
Training Script for ResNet50 Tumor Classifier
Two-stage fine-tuning approach for optimal results
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm

from ml_models.classification.resnet_classifier import (
    ResNetClassifier,
    TumorClassificationDataset,
    ClassificationTrainer
)
from config import (
    CLASSIFICATION_MODEL,
    CLASSIFICATION_TRAINING,
    DEVICE,
    BRATS_DATASET_PATH
)


def train_resnet_classifier():
    """
    Train ResNet50 with two-stage fine-tuning:
    Stage 1: Freeze backbone, train only classification head (5-10 epochs)
    Stage 2: Unfreeze all layers, train end-to-end (20-30 epochs)
    """
    
    print("=" * 70)
    print("ResNet50 Tumor Classification Training")
    print("=" * 70)
    
    # ============================================================
    # STAGE 0: Setup
    # ============================================================
    print("\n[Stage 0] Setup...")
    
    # Create model - extract only the parameters ResNetClassifier expects
    model_config = {
        "num_classes": CLASSIFICATION_MODEL.get("num_classes", 4),
        "pretrained": CLASSIFICATION_MODEL.get("pretrained", True),
        "in_channels": CLASSIFICATION_MODEL.get("in_channels", 4),
        "dropout": CLASSIFICATION_MODEL.get("dropout", 0.5)
    }
    model = ResNetClassifier(**model_config)
    print(f"✓ Model initialized with {model.get_num_parameters():,} parameters")
    print(f"✓ Using device: {DEVICE}")
    
    # Load dataset
    # TODO: Update this path to your preprocessed classification data
    data_dir = Path("data/classification_dataset")
    
    if not data_dir.exists():
        print(f"\n⚠ Dataset not found at {data_dir}")
        print("\nTo prepare classification dataset:")
        print("1. Extract 2D slices from BraTS 3D volumes")
        print("2. Organize by tumor type: data/classification_dataset/GBM/")
        print("3. Save as .npy files (shape: [4, 224, 224])")
        print("\nRun: python prepare_classification_data.py")
        return
    
    # Data transforms for training (augmentation)
    train_transform = torch.nn.Sequential(
        # Random horizontal flip
        # Random rotation
        # Normalization will be added here
    )
    
    # Load full dataset
    full_dataset = TumorClassificationDataset(
        data_dir=data_dir,
        transform=train_transform
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
        batch_size=CLASSIFICATION_TRAINING["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CLASSIFICATION_TRAINING["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # ============================================================
    # STAGE 1: Train Classification Head Only (Frozen Backbone)
    # ============================================================
    print("\n" + "=" * 70)
    print("[Stage 1] Fine-tuning: Classification head only")
    print("Backbone: FROZEN ❄️ | Head: TRAINABLE 🔥")
    print("=" * 70)
    
    # Freeze backbone
    model.freeze_backbone()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {trainable_params:,} (head only)")
    
    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        device=DEVICE,
        config={
            **CLASSIFICATION_TRAINING,
            "learning_rate": 1e-3,  # Higher LR for head training
            "num_epochs": 10
        }
    )
    
    # Train for 10 epochs
    best_val_acc = 0
    print("\nTraining classification head...")
    
    for epoch in range(10):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)
        
        # Update scheduler
        trainer.scheduler.step(val_acc)
        
        # Save history
        trainer.history["train_loss"].append(train_loss)
        trainer.history["train_acc"].append(train_acc)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_acc"].append(val_acc)
        trainer.history["learning_rates"].append(
            trainer.optimizer.param_groups[0]['lr']
        )
        
        print(f"\nEpoch {epoch+1}/10")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path("ml_models/classification/checkpoints/stage1_best.pth")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(checkpoint_path, epoch, best_val_acc)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n✓ Stage 1 complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    # ============================================================
    # STAGE 2: Full Fine-tuning (All Layers)
    # ============================================================
    print("\n" + "=" * 70)
    print("[Stage 2] Full fine-tuning: All layers")
    print("Backbone: TRAINABLE 🔥 | Head: TRAINABLE 🔥")
    print("=" * 70)
    
    # Unfreeze backbone
    model.unfreeze_backbone()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {trainable_params:,} (all layers)")
    
    # Recreate trainer with lower learning rate
    trainer = ClassificationTrainer(
        model=model,
        device=DEVICE,
        config={
            **CLASSIFICATION_TRAINING,
            "learning_rate": 1e-5,  # Much lower LR for fine-tuning
            "num_epochs": 30
        }
    )
    
    # Continue training for 30 epochs
    best_val_acc = 0
    print("\nFine-tuning all layers...")
    
    for epoch in range(30):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)
        
        # Update scheduler
        trainer.scheduler.step(val_acc)
        
        # Save history
        trainer.history["train_loss"].append(train_loss)
        trainer.history["train_acc"].append(train_acc)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_acc"].append(val_acc)
        trainer.history["learning_rates"].append(
            trainer.optimizer.param_groups[0]['lr']
        )
        
        print(f"\nEpoch {epoch+1}/30")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = Path("ml_models/classification/checkpoints/final_best.pth")
            trainer.save_checkpoint(checkpoint_path, epoch, best_val_acc)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping
        if epoch > 15 and val_acc < best_val_acc - 5:
            print("\n⚠ Early stopping triggered (no improvement)")
            break
    
    print(f"\n✓ Stage 2 complete! Final validation accuracy: {best_val_acc:.2f}%")
    
    # ============================================================
    # FINAL: Save Final Model
    # ============================================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Model saved to: ml_models/classification/checkpoints/final_best.pth")
    print("\nTo use in production:")
    print("  1. Copy final_best.pth to ml_models/classification/resnet_model.pth")
    print("  2. Update backend to load this checkpoint")
    print("  3. Test with: python test_classification.py")


if __name__ == "__main__":
    train_resnet_classifier()
