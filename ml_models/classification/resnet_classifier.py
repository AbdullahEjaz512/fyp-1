"""
Module 4: Tumor Classification using ResNet
Implements FR13.1 to FR13.8 - Tumor type and malignancy classification
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
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models
    from torchvision.models import ResNet50_Weights
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    ResNet50_Weights = None
    print("Warning: PyTorch not yet installed")

try:
    from config import (
        CLASSIFICATION_MODEL,
        CLASSIFICATION_TRAINING,
        DEVICE,
        TUMOR_TYPES,
        WHO_GRADES
    )
except ImportError:
    # Fallback configuration
    DEVICE = "cpu"
    CLASSIFICATION_MODEL = {
        "num_classes": 4,
        "pretrained": True
    }
    TUMOR_TYPES = {
        0: "Glioblastoma (GBM)",
        1: "Low-Grade Glioma (LGG)",
        2: "Meningioma",
        3: "Other"
    }
    WHO_GRADES = {1: "Grade I", 2: "Grade II", 3: "Grade III", 4: "Grade IV"}


class ResNetClassifier(nn.Module):
    """
    ResNet-based Tumor Classifier - FR13.1
    Classifies brain tumors by type and malignancy level
    
    Architecture:
    - Backbone: ResNet50 (pretrained on ImageNet)
    - Modified input: Accepts multi-channel MRI input
    - Classification head: Fully connected layers
    - Output: Tumor type probabilities
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 4,  # 4 MRI modalities
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super(ResNetClassifier, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ResNet classifier")
        
        # Load ResNet50 with modern weights API
        if pretrained and ResNet50_Weights is not None:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Modify first conv layer to accept 4-channel input
        # Original: conv1 expects 3 channels (RGB)
        # Modified: conv1 expects 4 channels (MRI modalities)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # Initialize new conv1 weights
        if pretrained:
            # Average pretrained weights across input channels
            with torch.no_grad():
                pretrained_weight = original_conv1.weight
                # Repeat weights for additional channels
                self.backbone.conv1.weight[:, :3, :, :] = pretrained_weight
                self.backbone.conv1.weight[:, 3:, :, :] = pretrained_weight.mean(dim=1, keepdim=True)
        
        # Get number of features from ResNet backbone
        num_features = self.backbone.fc.in_features
        
        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        self.in_channels = in_channels
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.backbone(x)
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Freeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze classification head
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class TumorClassificationDataset(Dataset):
    """
    Dataset for Tumor Classification - FR13.1
    Loads MRI slices and tumor type labels
    """
    
    def __init__(
        self,
        data_dir: Path,
        transform=None,
        label_map: Dict = None
    ):
        """
        Args:
            data_dir: Path to classification dataset
            transform: Transformations to apply
            label_map: Mapping of tumor types to labels
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.label_map = label_map or TUMOR_TYPES
        
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load dataset samples with labels"""
        samples = []
        
        # Assume directory structure: data_dir/tumor_type/sample.npy
        for tumor_type_dir in self.data_dir.iterdir():
            if tumor_type_dir.is_dir():
                tumor_type = tumor_type_dir.name
                
                # Find label for this tumor type
                label = None
                for idx, name in self.label_map.items():
                    if tumor_type.lower() in name.lower():
                        label = idx
                        break
                
                if label is not None:
                    # Load all samples for this type
                    for sample_file in tumor_type_dir.glob("*.npy"):
                        samples.append((sample_file, label))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load a single sample"""
        filepath, label = self.samples[idx]
        
        # Load preprocessed MRI data
        data = np.load(filepath)
        
        # Convert to tensor
        data = torch.from_numpy(data).float()
        
        # Apply transforms
        if self.transform:
            data = self.transform(data)
        
        return data, label


class ClassificationTrainer:
    """
    Training Pipeline for Tumor Classification - FR13.1
    Handles training, validation, and model evaluation
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        config: Dict = None
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config or CLASSIFICATION_TRAINING
        
        # Loss function - Cross Entropy for multi-class classification
        self.loss_function = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize accuracy
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.loss_function(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss /= len(train_loader)
        accuracy = 100 * correct / total
        
        return epoch_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model - FR13.5"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.loss_function(outputs, labels)
                val_loss += loss.item()
                
                # Track accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        return val_loss, accuracy
    
    def save_checkpoint(self, filepath: Path, epoch: int, best_metric: float):
        """Save model checkpoint"""
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
        print(f"  ✓ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        print(f"  ✓ Checkpoint loaded from {filepath}")
        return checkpoint


class TumorClassificationInference:
    """
    Inference Engine for Tumor Classification - FR13.2 to FR13.7
    Performs tumor type and malignancy classification
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def predict(
        self, 
        inputs: torch.Tensor,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict tumor type and characteristics - FR13.2, FR13.5
        
        Returns:
            Dictionary containing:
            - predicted_class: Tumor type
            - confidence: Classification confidence (%)
            - probabilities: Class probabilities
            - who_grade: WHO grade estimation
            - malignancy_level: Low/Medium/High
        """
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence, predicted_class = torch.max(probabilities, dim=1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item() * 100  # Convert to percentage
            
            # Get tumor type name - FR13.2
            tumor_type = TUMOR_TYPES.get(predicted_class, "Unknown")
            
            # Estimate WHO grade and malignancy - FR13.3, FR13.4
            who_grade, malignancy = self._estimate_malignancy(
                tumor_type, 
                confidence
            )
            
            result = {
                "predicted_class": predicted_class,
                "tumor_type": tumor_type,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist(),
                "who_grade": who_grade,
                "malignancy_level": malignancy,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Extract characteristic features - FR13.6
            features = self._extract_imaging_features(tumor_type)
            result["imaging_characteristics"] = features
            
            return result
    
    def _estimate_malignancy(
        self, 
        tumor_type: str, 
        confidence: float
    ) -> Tuple[str, str]:
        """
        Estimate WHO grade and malignancy level - FR13.3, FR13.4
        
        Based on tumor type and imaging characteristics
        """
        # Glioblastoma: Grade IV, High malignancy
        if "GBM" in tumor_type or "Glioblastoma" in tumor_type:
            return "Grade IV", "High"
        
        # Low-Grade Glioma: Grade I-II, Low-Medium malignancy
        elif "LGG" in tumor_type or "Low-Grade" in tumor_type:
            return "Grade II", "Low"
        
        # Meningioma: Usually Grade I, Low malignancy
        elif "Meningioma" in tumor_type:
            return "Grade I", "Low"
        
        # Unknown/Other
        else:
            return "Unknown", "Medium"
    
    def _extract_imaging_features(self, tumor_type: str) -> List[str]:
        """
        Extract characteristic imaging features - FR13.6
        Based on tumor type
        """
        features_map = {
            "Glioblastoma (GBM)": [
                "Ring-enhancing lesion",
                "Central necrosis",
                "Irregular margins",
                "Extensive perilesional edema",
                "Mass effect with midline shift"
            ],
            "Low-Grade Glioma (LGG)": [
                "Minimal or no enhancement",
                "Well-defined borders",
                "No or minimal edema",
                "Homogeneous signal intensity"
            ],
            "Meningioma": [
                "Extra-axial location",
                "Homogeneous enhancement",
                "Dural tail sign",
                "Well-circumscribed mass"
            ],
            "Other": [
                "Variable enhancement pattern",
                "Mixed signal characteristics"
            ]
        }
        
        return features_map.get(tumor_type, ["Features not characterized"])
    
    def batch_predict(self, data_loader: DataLoader) -> List[Dict]:
        """
        Batch prediction for multiple samples - FR13.8
        Useful for maintaining classification history
        """
        results = []
        
        for inputs, _ in data_loader:
            inputs = inputs.to(self.device)
            batch_results = self.predict(inputs)
            results.append(batch_results)
        
        return results


if __name__ == "__main__":
    print("=" * 60)
    print("Module 4: Tumor Classification - ResNet")
    print("=" * 60)
    
    if TORCH_AVAILABLE:
        print(f"\n✓ PyTorch available")
        print(f"  Device: {DEVICE}")
        
        # Initialize model
        print("\n[1] Initializing ResNet classifier...")
        model = ResNetClassifier(**CLASSIFICATION_MODEL)
        num_params = model.get_num_parameters()
        print(f"  ✓ Model created with {num_params:,} parameters")
        
        # Test forward pass
        print("\n[2] Testing forward pass...")
        batch_size = 2
        dummy_input = torch.randn(batch_size, 4, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  ✓ Input shape: {dummy_input.shape}")
        print(f"  ✓ Output shape: {output.shape}")
        
        # Test inference
        print("\n[3] Testing inference...")
        inference_engine = TumorClassificationInference(model, device="cpu")
        result = inference_engine.predict(dummy_input[0:1])
        
        print(f"  ✓ Predicted tumor type: {result['tumor_type']}")
        print(f"  ✓ Confidence: {result['confidence']:.2f}%")
        print(f"  ✓ WHO Grade: {result['who_grade']}")
        print(f"  ✓ Malignancy: {result['malignancy_level']}")
        
        print("\n✅ Module 4 initialization successful!")
    else:
        print("\n⚠ PyTorch not installed yet")
        print("  Install with: pip install torch torchvision")
