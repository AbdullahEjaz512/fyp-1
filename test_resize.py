
import torch
from monai.transforms import Resize
import numpy as np

def test_resize():
    print("Testing resize logic...")
    
    # Case 1: 3D volume (512, 512, 1)
    data_3d = np.random.rand(512, 512, 1).astype(np.float32)
    tensor_3d = torch.from_numpy(data_3d).float()
    print(f"Case 1 Input shape: {tensor_3d.shape}")
    
    try:
        resizer = Resize(spatial_size=(128, 128, 128))
        # Add channel dim: (1, 512, 512, 1)
        input_3d = tensor_3d.unsqueeze(0)
        print(f"Case 1 Input to Resize: {input_3d.shape}")
        output_3d = resizer(input_3d)
        print(f"Case 1 Output shape: {output_3d.shape}")
        print("Case 1: Success")
    except Exception as e:
        print(f"Case 1: Failed - {e}")

    # Case 2: 2D slice (512, 512)
    data_2d = np.random.rand(512, 512).astype(np.float32)
    tensor_2d = torch.from_numpy(data_2d).float()
    print(f"\nCase 2 Input shape: {tensor_2d.shape}")
    
    try:
        resizer = Resize(spatial_size=(128, 128, 128))
        # Add channel dim: (1, 512, 512)
        input_2d = tensor_2d.unsqueeze(0)
        print(f"Case 2 Input to Resize: {input_2d.shape}")
        output_2d = resizer(input_2d)
        print(f"Case 2 Output shape: {output_2d.shape}")
        print("Case 2: Success")
    except Exception as e:
        print(f"Case 2: Failed - {e}")

if __name__ == "__main__":
    test_resize()
