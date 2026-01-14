"""
Module 8: 3D Tumor Reconstruction Service
Generates interactive 3D meshes from segmentation masks using Marching Cubes
Exports to STL/OBJ formats and provides data for VTK.js/Three.js visualization
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from skimage import measure
from stl import mesh as stl_mesh
import json
import base64
import io

logger = logging.getLogger(__name__)


class TumorReconstruction3D:
    """
    3D Tumor Reconstruction Service
    Converts segmentation masks to 3D meshes for visualization
    """
    
    def __init__(self):
        self.tumor_regions = {
            1: {"name": "NCR", "color": [255, 165, 0], "opacity": 0.8},  # Orange
            2: {"name": "ED", "color": [0, 255, 0], "opacity": 0.5},     # Green
            3: {"name": "ET", "color": [255, 0, 0], "opacity": 0.9}      # Red
        }
    
    def load_segmentation(self, segmentation_path: str) -> Tuple[np.ndarray, Tuple]:
        """
        Load segmentation NIfTI file
        
        Args:
            segmentation_path: Path to segmentation .nii.gz file
        
        Returns:
            Tuple of (segmentation_data, voxel_spacing)
        """
        try:
            img = nib.load(segmentation_path)
            data = img.get_fdata()
            
            # Get voxel spacing
            if hasattr(img.header, 'get_zooms'):
                spacing = img.header.get_zooms()[:3]
            else:
                spacing = (1.0, 1.0, 1.0)
            
            logger.info(f"Loaded segmentation: shape={data.shape}, spacing={spacing}")
            return data, spacing
            
        except Exception as e:
            logger.error(f"Failed to load segmentation: {e}")
            raise
    
    def extract_surface_mesh(
        self,
        volume: np.ndarray,
        region_label: int,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        smoothing: bool = True,
        step_size: int = 1
    ) -> Dict:
        """
        Extract 3D surface mesh using Marching Cubes algorithm
        
        Args:
            volume: 3D segmentation volume
            region_label: Label value for this region (1=NCR, 2=ED, 3=ET)
            spacing: Voxel spacing in mm
            smoothing: Apply mesh smoothing
            step_size: Step size for marching cubes (higher = faster, lower quality)
        
        Returns:
            Dictionary with vertices, faces, normals
        """
        try:
            # Create binary mask for this region
            mask = (volume == region_label).astype(np.uint8)
            
            # Check if region exists
            if np.sum(mask) == 0:
                logger.warning(f"No voxels found for region {region_label}")
                return None
            
            # Apply marching cubes
            verts, faces, normals, values = measure.marching_cubes(
                mask,
                level=0.5,
                spacing=spacing,
                step_size=step_size,
                allow_degenerate=False
            )
            
            # Apply smoothing if requested
            if smoothing:
                verts = self._smooth_mesh(verts, faces)
            
            # Calculate mesh statistics
            num_vertices = len(verts)
            num_faces = len(faces)
            
            # Calculate bounding box
            bbox_min = np.min(verts, axis=0)
            bbox_max = np.max(verts, axis=0)
            
            # Calculate surface area (approximate)
            surface_area = self._calculate_surface_area(verts, faces)
            
            mesh_data = {
                'vertices': verts.tolist(),
                'faces': faces.tolist(),
                'normals': normals.tolist(),
                'region_label': region_label,
                'region_name': self.tumor_regions[region_label]['name'],
                'color': self.tumor_regions[region_label]['color'],
                'opacity': self.tumor_regions[region_label]['opacity'],
                'num_vertices': num_vertices,
                'num_faces': num_faces,
                'bounding_box': {
                    'min': bbox_min.tolist(),
                    'max': bbox_max.tolist()
                },
                'surface_area_mm2': float(surface_area)
            }
            
            logger.info(f"Extracted mesh for region {region_label}: "
                       f"{num_vertices} vertices, {num_faces} faces")
            
            return mesh_data
            
        except Exception as e:
            logger.error(f"Mesh extraction failed for region {region_label}: {e}")
            return None
    
    def _smooth_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        iterations: int = 8
    ) -> np.ndarray:
        """
        Apply Laplacian smoothing to mesh vertices
        
        Args:
            vertices: Vertex coordinates
            faces: Face indices
            iterations: Number of smoothing iterations
        
        Returns:
            Smoothed vertices
        """
        smoothed_verts = vertices.copy()
        
        # Build vertex-vertex adjacency
        n_verts = len(vertices)
        adjacency = [set() for _ in range(n_verts)]
        
        for face in faces:
            for i in range(3):
                v1 = face[i]
                v2 = face[(i + 1) % 3]
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
        
        # Iterative smoothing
        for _ in range(iterations):
            new_verts = smoothed_verts.copy()
            
            for i in range(n_verts):
                if len(adjacency[i]) > 0:
                    neighbors = list(adjacency[i])
                    neighbor_coords = smoothed_verts[neighbors]
                    new_verts[i] = 0.5 * smoothed_verts[i] + 0.5 * np.mean(neighbor_coords, axis=0)
            
            smoothed_verts = new_verts
        
        return smoothed_verts
    
    def _calculate_surface_area(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> float:
        """
        Calculate total surface area of mesh
        
        Args:
            vertices: Vertex coordinates
            faces: Face indices
        
        Returns:
            Total surface area in mm²
        """
        total_area = 0.0
        
        for face in faces:
            # Get triangle vertices
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Calculate triangle area using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross)
            
            total_area += area
        
        return total_area
    
    def reconstruct_all_regions(
        self,
        segmentation_path: str,
        smoothing: bool = True,
        step_size: int = 1
    ) -> Dict:
        """
        Reconstruct 3D meshes for all tumor regions
        
        Args:
            segmentation_path: Path to segmentation file
            smoothing: Apply mesh smoothing
            step_size: Marching cubes step size
        
        Returns:
            Dictionary with meshes for all regions
        """
        # Load segmentation
        volume, spacing = self.load_segmentation(segmentation_path)
        
        # Extract meshes for each region
        meshes = {}
        total_surface_area = 0.0
        
        for region_label in [1, 2, 3]:  # NCR, ED, ET
            mesh_data = self.extract_surface_mesh(
                volume,
                region_label,
                spacing,
                smoothing,
                step_size
            )
            
            if mesh_data is not None:
                meshes[mesh_data['region_name']] = mesh_data
                total_surface_area += mesh_data['surface_area_mm2']
        
        # Calculate overall bounding box
        if meshes:
            all_mins = [m['bounding_box']['min'] for m in meshes.values()]
            all_maxs = [m['bounding_box']['max'] for m in meshes.values()]
            
            global_bbox = {
                'min': np.min(all_mins, axis=0).tolist(),
                'max': np.max(all_maxs, axis=0).tolist()
            }
        else:
            global_bbox = {'min': [0, 0, 0], 'max': [0, 0, 0]}
        
        result = {
            'meshes': meshes,
            'num_regions': len(meshes),
            'total_surface_area_mm2': float(total_surface_area),
            'global_bounding_box': global_bbox,
            'spacing': spacing
        }
        
        logger.info(f"Reconstructed {len(meshes)} regions, "
                   f"total surface area: {total_surface_area:.2f} mm²")
        
        return result
    
    def export_to_stl(
        self,
        mesh_data: Dict,
        output_path: str
    ) -> str:
        """
        Export mesh to STL format for 3D printing or external visualization
        
        Args:
            mesh_data: Mesh dictionary with vertices and faces
            output_path: Output STL file path
        
        Returns:
            Path to saved STL file
        """
        try:
            vertices = np.array(mesh_data['vertices'])
            faces = np.array(mesh_data['faces'])
            
            # Create STL mesh
            tumor_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
            
            for i, face in enumerate(faces):
                for j in range(3):
                    tumor_mesh.vectors[i][j] = vertices[face[j]]
            
            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tumor_mesh.save(str(output_path))
            
            logger.info(f"STL saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"STL export failed: {e}")
            raise
    
    def export_to_obj(
        self,
        mesh_data: Dict,
        output_path: str
    ) -> str:
        """
        Export mesh to OBJ format (widely supported)
        
        Args:
            mesh_data: Mesh dictionary with vertices and faces
            output_path: Output OBJ file path
        
        Returns:
            Path to saved OBJ file
        """
        try:
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                # Write header
                f.write(f"# OBJ file - {mesh_data['region_name']}\n")
                f.write(f"# Vertices: {len(vertices)}\n")
                f.write(f"# Faces: {len(faces)}\n\n")
                
                # Write vertices
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Write faces (OBJ uses 1-indexed)
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            logger.info(f"OBJ saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"OBJ export failed: {e}")
            raise
    
    def prepare_for_threejs(
        self,
        meshes: Dict
    ) -> Dict:
        """
        Prepare mesh data in format optimized for Three.js
        
        Args:
            meshes: Dictionary of mesh data
        
        Returns:
            Three.js compatible geometry data
        """
        threejs_data = {
            'geometries': []
        }
        
        for region_name, mesh in meshes.items():
            vertices = np.array(mesh['vertices']).flatten().tolist()
            faces = np.array(mesh['faces']).flatten().tolist()
            
            geometry = {
                'type': 'BufferGeometry',
                'data': {
                    'attributes': {
                        'position': {
                            'itemSize': 3,
                            'type': 'Float32Array',
                            'array': vertices
                        }
                    },
                    'index': {
                        'type': 'Uint32Array',
                        'array': faces
                    }
                },
                'metadata': {
                    'region': region_name,
                    'color': mesh['color'],
                    'opacity': mesh['opacity']
                }
            }
            
            threejs_data['geometries'].append(geometry)
        
        return threejs_data
    
    def prepare_for_vtkjs(
        self,
        meshes: Dict
    ) -> Dict:
        """
        Prepare mesh data in format optimized for VTK.js
        
        Args:
            meshes: Dictionary of mesh data
        
        Returns:
            VTK.js compatible data structure
        """
        vtkjs_data = {
            'vtkClass': 'vtkPolyData',
            'regions': []
        }
        
        for region_name, mesh in meshes.items():
            region_data = {
                'name': region_name,
                'points': {
                    'vtkClass': 'vtkDataArray',
                    'name': 'points',
                    'dataType': 'Float32Array',
                    'values': np.array(mesh['vertices']).flatten().tolist()
                },
                'polys': {
                    'vtkClass': 'vtkCellArray',
                    'name': 'polys',
                    'dataType': 'Uint32Array',
                    'values': self._convert_to_vtk_polys(mesh['faces'])
                },
                'metadata': {
                    'color': mesh['color'],
                    'opacity': mesh['opacity'],
                    'numVertices': mesh['num_vertices'],
                    'numFaces': mesh['num_faces']
                }
            }
            
            vtkjs_data['regions'].append(region_data)
        
        return vtkjs_data
    
    def _convert_to_vtk_polys(self, faces: List) -> List:
        """
        Convert faces to VTK polygon format
        VTK format: [n, v1, v2, ..., vn] for each polygon
        """
        vtk_polys = []
        for face in faces:
            vtk_polys.append(3)  # Triangle has 3 vertices
            vtk_polys.extend(face)
        return vtk_polys
    
    def generate_cross_sections(
        self,
        meshes: Dict,
        num_slices: int = 10
    ) -> Dict:
        """
        Generate cross-sectional views at different heights
        Useful for slice-based exploration
        
        Args:
            meshes: Dictionary of mesh data
            num_slices: Number of cross-sections to generate
        
        Returns:
            Dictionary with cross-section data
        """
        # Get global bounding box
        all_mins = [m['bounding_box']['min'] for m in meshes.values()]
        all_maxs = [m['bounding_box']['max'] for m in meshes.values()]
        
        z_min = min(m[2] for m in all_mins)
        z_max = max(m[2] for m in all_maxs)
        
        # Generate evenly spaced slice positions
        slice_positions = np.linspace(z_min, z_max, num_slices)
        
        cross_sections = {
            'num_slices': num_slices,
            'slice_positions': slice_positions.tolist(),
            'z_range': [float(z_min), float(z_max)]
        }
        
        return cross_sections


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test reconstruction
    reconstructor = TumorReconstruction3D()
    
    # Example segmentation path
    seg_path = "data/processed/sample_segmentation.nii.gz"
    
    if Path(seg_path).exists():
        print("Reconstructing 3D meshes...")
        
        result = reconstructor.reconstruct_all_regions(
            seg_path,
            smoothing=True,
            step_size=2
        )
        
        print(f"\nReconstruction complete:")
        print(f"  Regions: {result['num_regions']}")
        print(f"  Total surface area: {result['total_surface_area_mm2']:.2f} mm²")
        
        # Export meshes
        for region_name, mesh in result['meshes'].items():
            stl_path = f"data/3d_models/{region_name}.stl"
            obj_path = f"data/3d_models/{region_name}.obj"
            
            reconstructor.export_to_stl(mesh, stl_path)
            reconstructor.export_to_obj(mesh, obj_path)
            
            print(f"\n{region_name}:")
            print(f"  Vertices: {mesh['num_vertices']}")
            print(f"  Faces: {mesh['num_faces']}")
            print(f"  Surface area: {mesh['surface_area_mm2']:.2f} mm²")
            print(f"  STL: {stl_path}")
            print(f"  OBJ: {obj_path}")
        
        # Prepare for web visualization
        threejs_data = reconstructor.prepare_for_threejs(result['meshes'])
        vtkjs_data = reconstructor.prepare_for_vtkjs(result['meshes'])
        
        print(f"\nVisualization data prepared:")
        print(f"  Three.js geometries: {len(threejs_data['geometries'])}")
        print(f"  VTK.js regions: {len(vtkjs_data['regions'])}")
        
    else:
        print(f"Segmentation file not found: {seg_path}")
        print("Please provide a valid segmentation path.")
