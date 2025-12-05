"""
Test script for the trained U-Net segmentation model
"""
import torch
import sys
sys.path.insert(0, '.')
from ml_models.segmentation.unet3d import UNet3D

print('='*60)
print('Testing Trained U-Net Model')
print('='*60)

# Create model with same architecture as Kaggle
model = UNet3D()
print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

# Load the trained weights
checkpoint_path = 'ml_models/segmentation/unet_model.pth'
print(f'\nLoading checkpoint: {checkpoint_path}')

checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f'Checkpoint keys: {list(checkpoint.keys())}')
print(f'Best Dice: {checkpoint.get("best_dice", "N/A")}')
epoch = checkpoint.get("epoch", -1)
print(f'Trained epochs: {epoch + 1 if epoch >= 0 else "N/A"}')

# Load state dict
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
print('\n✅ Model weights loaded successfully!')

# Test inference
model.eval()
dummy_input = torch.randn(1, 4, 128, 128, 128)
print(f'\nTest input shape: {dummy_input.shape}')

with torch.no_grad():
    output = model(dummy_input)
    
print(f'Output shape: {output.shape}')
print(f'Output range: [{output.min():.3f}, {output.max():.3f}]')

# Get prediction
pred = torch.argmax(output, dim=1)
print(f'Prediction shape: {pred.shape}')
print(f'Unique classes predicted: {torch.unique(pred).tolist()}')

print('\n' + '='*60)
print('✅ MODEL TEST PASSED - Ready for inference!')
print('='*60)
