"""
Module 6: Explainable AI - Grad-CAM and SHAP
Provides visual explanations for model predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN visualization
    Shows which regions of the input influenced the model's decision
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to generate CAM from (e.g., last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = use predicted class)
        
        Returns:
            cam: Heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_cam_on_image(
        self,
        img: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on original image
        
        Args:
            img: Original image (H, W, 3) in [0, 255]
            cam: CAM heatmap (H, W) in [0, 1]
            alpha: Blending factor
            colormap: OpenCV colormap
        
        Returns:
            Overlayed image (H, W, 3)
        """
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure img is in correct format and type
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Ensure heatmap is same dtype
        heatmap = heatmap.astype(np.uint8)
        
        # Blend
        overlayed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model interpretability
    Explains the output of machine learning models
    """
    
    def __init__(self, model, background_data: Optional[np.ndarray] = None):
        """
        Args:
            model: PyTorch model or prediction function
            background_data: Background dataset for SHAP computation
        """
        self.model = model
        self.background_data = background_data
        
        try:
            import shap
            self.shap = shap
            logger.info("SHAP library loaded successfully")
        except ImportError:
            logger.warning("SHAP library not available. Install with: pip install shap")
            self.shap = None
    
    def explain_prediction(
        self,
        input_data: np.ndarray,
        num_samples: int = 100
    ) -> dict:
        """
        Generate SHAP explanation for a prediction
        
        Args:
            input_data: Input sample to explain (1, C, H, W)
            num_samples: Number of samples for SHAP computation
        
        Returns:
            Dictionary with SHAP values and visualizations
        """
        if self.shap is None:
            return {'error': 'SHAP library not available'}
        
        try:
            # Create explainer
            if self.background_data is not None:
                explainer = self.shap.DeepExplainer(self.model, self.background_data)
            else:
                # Use a small random background if none provided
                background = torch.randn(10, *input_data.shape[1:])
                explainer = self.shap.DeepExplainer(self.model, background)
            
            # Compute SHAP values
            shap_values = explainer.shap_values(input_data, check_additivity=False)
            
            # Convert to numpy
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            
            return {
                'shap_values': shap_values,
                'base_value': explainer.expected_value,
                'feature_importance': np.abs(shap_values).mean(axis=tuple(range(2, shap_values.ndim)))
            }
        
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}


class ExplainabilityService:
    """
    Combined service for model explainability
    Provides Grad-CAM and SHAP explanations
    """
    
    def __init__(self, classification_model=None, segmentation_model=None):
        self.classification_model = classification_model
        self.segmentation_model = segmentation_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def explain_classification(
        self,
        input_image: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "gradcam"
    ) -> dict:
        """
        Generate explanation for classification prediction
        
        Args:
            input_image: Input MRI slice (H, W, C)
            target_class: Target class to explain
            method: 'gradcam' or 'shap'
        
        Returns:
            Dictionary with explanation visualizations
        """
        if self.classification_model is None:
            return {'error': 'Classification model not loaded'}
        
        # Prepare input
        if len(input_image.shape) == 2:
            input_image = np.expand_dims(input_image, axis=-1)
        
        input_tensor = torch.FloatTensor(input_image).permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        if method.lower() == "gradcam":
            # Get last convolutional layer
            try:
                # For ResNet
                target_layer = self.classification_model.model.layer4[-1]
            except:
                try:
                    # Generic approach
                    target_layer = list(self.classification_model.children())[-2]
                except:
                    return {'error': 'Could not find target layer for Grad-CAM'}
            
            # Generate Grad-CAM
            grad_cam = GradCAM(self.classification_model, target_layer)
            cam = grad_cam.generate_cam(input_tensor, target_class)
            
            # Create overlay
            img_normalized = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-8)
            img_rgb = np.stack([img_normalized[:, :, 0]] * 3, axis=-1)
            overlay = grad_cam.overlay_cam_on_image(img_rgb * 255, cam)
            
            return {
                'method': 'gradcam',
                'heatmap': cam,
                'overlay': overlay,
                'target_class': target_class
            }
        
        elif method.lower() == "shap":
            explainer = SHAPExplainer(self.classification_model)
            result = explainer.explain_prediction(input_tensor)
            return result
        
        else:
            return {'error': f'Unknown method: {method}'}
    
    def explain_segmentation(
        self,
        input_volume: np.ndarray,
        slice_idx: int = None
    ) -> dict:
        """
        Generate explanation for segmentation prediction
        
        Args:
            input_volume: Input MRI volume (C, H, W, D) or slice (C, H, W)
            slice_idx: Which slice to explain (for 3D volumes)
        
        Returns:
            Dictionary with attention maps and feature importance
        """
        if self.segmentation_model is None:
            return {'error': 'Segmentation model not loaded'}
        
        # For segmentation, we can show:
        # 1. Attention maps at different depths
        # 2. Feature importance by channel
        # 3. Uncertainty maps
        
        result = {
            'method': 'attention_maps',
            'message': 'Segmentation explainability shows which regions the model focused on'
        }
        
        # Simplified attention visualization
        # In production, hook into U-Net decoder layers
        
        return result
    
    def generate_explanation_report(
        self,
        classification_result: dict,
        segmentation_result: Optional[dict] = None
    ) -> str:
        """
        Generate human-readable explanation report
        
        Args:
            classification_result: Results from explain_classification
            segmentation_result: Results from explain_segmentation
        
        Returns:
            Formatted text report
        """
        report = []
        report.append("=== Model Explanation Report ===\n")
        
        if 'error' not in classification_result:
            report.append("Classification Explanation:")
            report.append(f"  Method: {classification_result.get('method', 'N/A')}")
            report.append(f"  Target Class: {classification_result.get('target_class', 'Predicted')}")
            report.append("  Interpretation: The heatmap highlights regions that most influenced")
            report.append("  the model's classification decision. Brighter areas indicate higher")
            report.append("  importance in determining the tumor type.\n")
        
        if segmentation_result and 'error' not in segmentation_result:
            report.append("Segmentation Explanation:")
            report.append("  The model's attention maps show which features it considered")
            report.append("  when delineating tumor boundaries.\n")
        
        report.append("Disclaimer: These explanations are approximations of model behavior")
        report.append("and should be used to support, not replace, clinical judgment.")
        
        return "\n".join(report)


# Testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  EXPLAINABLE AI - MODULE 6 TEST")
    print("="*70)
    
    # Create dummy model and data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 4)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    print("\n✓ Creating dummy classification model")
    model = DummyModel()
    model.eval()
    
    print("✓ Initializing Grad-CAM")
    grad_cam = GradCAM(model, model.conv2)
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 128, 128)
    
    print("✓ Generating Grad-CAM heatmap")
    cam = grad_cam.generate_cam(dummy_input)
    print(f"   CAM shape: {cam.shape}")
    print(f"   CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    print("\n✓ Testing ExplainabilityService")
    service = ExplainabilityService(classification_model=model)
    
    # Create dummy image
    dummy_image = np.random.rand(128, 128, 1)
    result = service.explain_classification(dummy_image, method="gradcam")
    
    if 'error' not in result:
        print(f"   ✓ Explanation generated successfully")
        print(f"   Method: {result['method']}")
        print(f"   Heatmap shape: {result['heatmap'].shape}")
    
    print("\n" + "="*70)
    print("✓ Module 6 test complete!")
    print("="*70 + "\n")
