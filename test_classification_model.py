
"""
Test script for the ResNet50 tumor classification model
"""
import torch
import sys
import os
import io
import contextlib
sys.path.insert(0, '.')
from ml_models.classification.resnet_classifier import ResNetClassifier

print('='*60)
print('Testing Trained ResNet Classification Model')
print('='*60)

# 1. Initialize Model
model = ResNetClassifier(num_classes=4, in_channels=4, pretrained=False)
print(f'Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}')

# 2. Check for Weights
model_path = 'ml_models/classification/resnet_model.pth'
if not os.path.exists(model_path):
    print(f"\n❌ Model weights not found at: {model_path}")
    print("   Cannot test trained model.")
else:
    print(f'\nLoading checkpoint: {model_path}')
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            if 'best_acc' in checkpoint:
                print(f"Best Accuracy: {checkpoint['best_acc']:.2f}%")
        else:
            state_dict = checkpoint
            print("loaded state dict directly")
            
        # 3. Load Weights
        model.load_state_dict(state_dict)
        print('\n✅ Model weights loaded successfully!')
        
        # 4. Test Inference
        model.eval()
        dummy_input = torch.randn(1, 4, 224, 224) # ResNet Standard Input Size
        print(f'\nTest input shape: {dummy_input.shape}')
        
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f'Output shape: {output.shape}')
        print(f'Raw Logits: {output[0].tolist()}')
        
        # Calculate probabilities manually since model returns raw logits
        probabilities = torch.nn.functional.softmax(output, dim=1)
        print(f'Probabilities (Confidence): {probabilities[0].tolist()}')
        
        pred_idx = torch.argmax(output, dim=1).item()
        print(f'Predicted Class Index: {pred_idx}')
        
        print('\n' + '='*60)
        print('✅ CLASSIFICATION MODEL TEST PASSED')
        print('='*60)
            
    except Exception as e:
        print(f"\n❌ Error validating classification model: {e}")
