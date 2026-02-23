"""
Module 7: 2D/3D MRI Visualization Service
Provides slice viewing and 3D volume rendering capabilities
"""

import numpy as np
import nibabel as nib
from typing import Optional, Tuple, Union, List
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available")


class MRIVisualizationService:
    """
    Service for 2D slice and 3D volume visualization of MRI scans
    """
    
    def __init__(self):
        self.tumor_colormap = self._create_tumor_colormap()
    
    def _create_tumor_colormap(self):
        """Create custom colormap for tumor regions"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Define colors for different tumor regions
        colors = [
            (0, 0, 0, 0),        # Background - transparent
            (1, 0, 0, 0.6),      # NCR/NET - red
            (0, 1, 0, 0.6),      # ED - green  
            (0, 0, 1, 0.6),      # ET - blue
        ]
        
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('tumor', colors, N=n_bins)
        return cmap
    
    def load_nifti(self, filepath: str) -> Tuple[np.ndarray, dict]:
        """
        Load NIfTI file and extract metadata
        
        Args:
            filepath: Path to .nii or .nii.gz file
        
        Returns:
            (volume_data, metadata)
        """
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            
            metadata = {
                'shape': data.shape,
                'affine': img.affine.tolist(),
                'header': dict(img.header),
                'voxel_size': img.header.get_zooms()
            }
            
            return data, metadata
        
        except Exception as e:
            logger.error(f"Failed to load NIfTI file: {e}")
            raise
    
    def normalize_intensity(
        self,
        volume: np.ndarray,
        percentile_range: Tuple[float, float] = (1, 99)
    ) -> np.ndarray:
        """
        Normalize volume intensity using percentile clipping
        
        Args:
            volume: Input volume
            percentile_range: (min_percentile, max_percentile)
        
        Returns:
            Normalized volume in [0, 1]
        """
        lower = np.percentile(volume, percentile_range[0])
        upper = np.percentile(volume, percentile_range[1])
        
        volume = np.clip(volume, lower, upper)
        volume = (volume - lower) / (upper - lower + 1e-8)
        
        return volume
    
    def extract_slice(
        self,
        volume: np.ndarray,
        slice_idx: int,
        axis: int = 2,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract 2D slice from 3D volume
        
        Args:
            volume: 3D volume (H, W, D) or 4D multi-channel (C, H, W, D)
            slice_idx: Index of slice to extract
            axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
            normalize: Whether to normalize intensity
        
        Returns:
            2D slice
        """
        if len(volume.shape) == 4:
            # Multi-channel - take first channel
            volume = volume[0]
        
        if axis == 0:
            slice_2d = volume[slice_idx, :, :]
        elif axis == 1:
            slice_2d = volume[:, slice_idx, :]
        else:  # axis == 2
            slice_2d = volume[:, :, slice_idx]
        
        if normalize:
            slice_2d = self.normalize_intensity(slice_2d)
        
        return slice_2d
    
    def create_slice_visualization(
        self,
        mri_slice: np.ndarray,
        segmentation_slice: Optional[np.ndarray] = None,
        title: str = "MRI Slice",
        colorbar: bool = True
    ) -> str:
        """
        Create visualization of MRI slice with optional segmentation overlay
        
        Args:
            mri_slice: 2D MRI slice
            segmentation_slice: Optional segmentation mask to overlay
            title: Plot title
            colorbar: Whether to show colorbar
        
        Returns:
            Base64-encoded PNG image
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available for visualization")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display MRI slice
        im = ax.imshow(mri_slice, cmap='gray', aspect='auto')
        
        # Overlay segmentation if provided
        if segmentation_slice is not None:
            extent = [0, mri_slice.shape[1], mri_slice.shape[0], 0]
            ax.imshow(segmentation_slice, cmap=self.tumor_colormap, alpha=0.5, aspect='auto', extent=extent)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if colorbar and segmentation_slice is None:
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return img_base64
    
    def create_multi_view(
        self,
        volume: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        center_coords: Optional[Tuple[int, int, int]] = None
    ) -> str:
        """
        Create multi-view visualization (axial, coronal, sagittal)
        For 2D images (depth=1), only show axial view
        
        Args:
            volume: 3D MRI volume (or 2D with depth=1)
            segmentation: Optional segmentation mask
            center_coords: (x, y, z) coordinates for slice positions
        
        Returns:
            Base64-encoded PNG image
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available for visualization")
        
        # Determine slice positions
        if center_coords is None:
            center_coords = tuple(s // 2 for s in volume.shape)
        
        x, y, z = center_coords
        
        # Always show all three views for consistency
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        views = [
            ('Sagittal', self.extract_slice(volume, x, axis=0), 0),
            ('Coronal', self.extract_slice(volume, y, axis=1), 1),
            ('Axial', self.extract_slice(volume, z, axis=2), 2)
        ]
        
        for (title, slice_2d, axis_idx), ax in zip(views, axes):
            ax.imshow(slice_2d, cmap='gray', aspect='auto')
            
            if segmentation is not None:
                # If volume is 4D (C, H, W, D), we need the spatial dimensions which are the last 3
                vol_spatial_shape = volume.shape[-3:]
                seg_idx = int([x, y, z][axis_idx] * (segmentation.shape[axis_idx] / vol_spatial_shape[axis_idx]))
                seg_idx = max(0, min(seg_idx, segmentation.shape[axis_idx] - 1))
                seg_slice = self.extract_slice(segmentation, 
                                              seg_idx, 
                                              axis=axis_idx, 
                                              normalize=False)
                extent = [0, slice_2d.shape[1], slice_2d.shape[0], 0]
                ax.imshow(seg_slice, cmap=self.tumor_colormap, alpha=0.5, aspect='auto', extent=extent)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return img_base64
    
    def generate_volume_montage(
        self,
        volume: np.ndarray,
        num_slices: int = 12,
        axis: int = 2
    ) -> str:
        """
        Create montage of multiple slices
        
        Args:
            volume: 3D volume
            num_slices: Number of slices to show
            axis: Axis to slice along
        
        Returns:
            Base64-encoded PNG image
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available for visualization")
        
        # Calculate slice indices
        depth = volume.shape[axis]
        slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        
        # Create grid
        rows = int(np.ceil(np.sqrt(num_slices)))
        cols = int(np.ceil(num_slices / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if num_slices > 1 else [axes]
        
        for idx, slice_idx in enumerate(slice_indices):
            slice_2d = self.extract_slice(volume, slice_idx, axis=axis)
            axes[idx].imshow(slice_2d, cmap='gray', aspect='auto')
            axes[idx].set_title(f'Slice {slice_idx}', fontsize=10)
            axes[idx].axis('off')
        
        # Hide extra subplots
        for idx in range(num_slices, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return img_base64
    
    def calculate_volume_metrics(
        self,
        segmentation: np.ndarray,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> dict:
        """
        Calculate volume metrics from segmentation mask
        
        Args:
            segmentation: 3D segmentation mask
            voxel_size: Physical size of voxels (mm)
        
        Returns:
            Dictionary with volume metrics
        """
        voxel_volume = np.prod(voxel_size)  # mm³
        
        metrics = {}
        
        # Calculate volume for each region
        for region_id in range(1, 4):  # Assuming 3 tumor regions
            mask = segmentation == region_id
            num_voxels = np.sum(mask)
            volume_mm3 = num_voxels * voxel_volume
            volume_cc = volume_mm3 / 1000  # Convert to cc
            
            region_names = {1: 'NCR/NET', 2: 'ED', 3: 'ET'}
            metrics[region_names.get(region_id, f'Region_{region_id}')] = {
                'voxels': int(num_voxels),
                'volume_mm3': float(volume_mm3),
                'volume_cc': float(volume_cc)
            }
        
        # Total tumor volume
        total_mask = segmentation > 0
        total_voxels = np.sum(total_mask)
        total_volume_cc = (total_voxels * voxel_volume) / 1000
        
        metrics['Total'] = {
            'voxels': int(total_voxels),
            'volume_mm3': float(total_voxels * voxel_volume),
            'volume_cc': float(total_volume_cc)
        }
        
        return metrics
    
    def generate_3d_projection(
        self,
        volume: np.ndarray,
        method: str = 'mip'
    ) -> str:
        """
        Generate 3D projection visualization
        
        Args:
            volume: 3D volume
            method: 'mip' (maximum intensity projection) or 'average'
        
        Returns:
            Base64-encoded PNG image
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib not available for visualization")
        
        # Always generate all three projections for consistency
        if method == 'mip':
            # Maximum intensity projection from three angles
            proj_axial = np.max(volume, axis=2)
            proj_coronal = np.max(volume, axis=1)
            proj_sagittal = np.max(volume, axis=0)
        else:  # average
            proj_axial = np.mean(volume, axis=2)
            proj_coronal = np.mean(volume, axis=1)
            proj_sagittal = np.mean(volume, axis=0)
        
        # Always show all three projections
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        method_label = 'MIP' if method == 'mip' else 'Average'
        projections = [
            (f'Sagittal {method_label}', proj_sagittal),
            (f'Coronal {method_label}', proj_coronal),
            (f'Axial {method_label}', proj_axial)
        ]
        
        for (title, proj), ax in zip(projections, axes):
                proj_norm = self.normalize_intensity(proj)
                ax.imshow(proj_norm, cmap='hot', aspect='auto')
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return img_base64


# Testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  2D/3D VISUALIZATION - MODULE 7 TEST")
    print("="*70)
    
    print("\n✓ Initializing visualization service")
    service = MRIVisualizationService()
    
    # Create dummy volume
    print("✓ Creating dummy 3D volume")
    dummy_volume = np.random.rand(128, 128, 64)
    dummy_segmentation = np.random.randint(0, 4, (128, 128, 64))
    
    # Test slice extraction
    print("✓ Testing slice extraction")
    slice_2d = service.extract_slice(dummy_volume, 32, axis=2)
    print(f"   Extracted slice shape: {slice_2d.shape}")
    
    # Test volume metrics
    print("✓ Testing volume metrics calculation")
    metrics = service.calculate_volume_metrics(dummy_segmentation, voxel_size=(1.0, 1.0, 1.0))
    print(f"   Total tumor volume: {metrics['Total']['volume_cc']:.2f} cc")
    
    if MATPLOTLIB_AVAILABLE:
        print("✓ Testing visualization generation")
        try:
            img_b64 = service.create_slice_visualization(slice_2d, title="Test Slice")
            print(f"   Generated visualization: {len(img_b64)} bytes (base64)")
            
            montage_b64 = service.generate_volume_montage(dummy_volume, num_slices=9)
            print(f"   Generated montage: {len(montage_b64)} bytes (base64)")
        except Exception as e:
            print(f"   ⚠ Visualization test failed: {e}")
    else:
        print("⚠ Matplotlib not available, skipping visualization tests")
    
    print("\n" + "="*70)
    print("✓ Module 7 test complete!")
    print("="*70 + "\n")
