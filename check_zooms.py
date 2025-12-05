
import nibabel as nib
import numpy as np

def check_zooms():
    filename = "debug_scan_2d.nii.gz"
    
    # Create the file again just in case
    data = np.random.rand(512, 512, 1).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)
    
    # Load and check zooms
    img_loaded = nib.load(filename)
    zooms = img_loaded.header.get_zooms()
    print(f"Shape: {img_loaded.shape}")
    print(f"Zooms: {zooms}")
    print(f"Zooms[:3]: {zooms[:3]}")
    
    try:
        s0 = zooms[0]
        s1 = zooms[1]
        s2 = zooms[2]
        print(f"Accessing index 2: {s2}")
    except IndexError:
        print("Accessing index 2 failed!")

    # Test with 2D shape (no 3rd dim)
    filename_2d = "debug_scan_2d_flat.nii.gz"
    data_2d = np.random.rand(512, 512).astype(np.float32)
    img_2d = nib.Nifti1Image(data_2d, affine)
    nib.save(img_2d, filename_2d)
    
    img_loaded_2d = nib.load(filename_2d)
    zooms_2d = img_loaded_2d.header.get_zooms()
    print(f"\nFlat 2D Shape: {img_loaded_2d.shape}")
    print(f"Flat 2D Zooms: {zooms_2d}")
    
    try:
        s2 = zooms_2d[2]
        print(f"Accessing index 2: {s2}")
    except IndexError:
        print("Accessing index 2 failed!")

if __name__ == "__main__":
    check_zooms()
