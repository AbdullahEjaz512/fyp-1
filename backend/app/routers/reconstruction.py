"""
Module 8: 3D Tumor Reconstruction API Routes
Provides endpoints for 3D mesh generation, STL/OBJ export, and web visualization data
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from typing import Optional
import logging
from pathlib import Path
import json

from app.database import get_db, File as DBFile, AnalysisResult, FileAccessPermission, CaseCollaboration
from app.dependencies.auth import get_current_user
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reconstruction", tags=["3D Reconstruction"])

# Lazy load reconstruction service
_reconstruction_service = None

def get_reconstruction_service():
    """Lazy load reconstruction service"""
    global _reconstruction_service
    if _reconstruction_service is None:
        from ml_models.reconstruction.tumor_reconstruction_3d import TumorReconstruction3D
        _reconstruction_service = TumorReconstruction3D()
    return _reconstruction_service


@router.post("/generate/{file_id}")
async def generate_3d_mesh(
    file_id: int,
    smoothing: bool = True,
    step_size: int = 2,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate 3D meshes from segmentation mask
    
    Args:
        file_id: File ID with segmentation
        smoothing: Apply Laplacian smoothing
        step_size: Marching cubes step size (higher = faster, lower quality)
    
    Returns:
        Dictionary with mesh data for all tumor regions
    """
    try:
        # Get file
        file = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check access: owner, doctor with permission, or collaborating doctor
        requester_id = user.get('user_id') if isinstance(user, dict) else user.user_id
        requester_role = user.get('role') if isinstance(user, dict) else user.role
        
        has_access = False
        if file.user_id == requester_id:
            has_access = True
        elif requester_role in ['doctor', 'radiologist', 'oncologist']:
            permission = db.query(FileAccessPermission).filter(
                FileAccessPermission.file_id == file_id,
                FileAccessPermission.doctor_id == requester_id,
                FileAccessPermission.status == 'active'
            ).first()
            
            if permission:
                has_access = True
            else:
                collab = db.query(CaseCollaboration).filter(
                    CaseCollaboration.file_id == file_id,
                    CaseCollaboration.collaborating_doctor_id == requester_id,
                    CaseCollaboration.status == 'active'
                ).first()
                if collab:
                    has_access = True
        
        if not has_access:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get segmentation result
        analysis = db.query(AnalysisResult).filter(
            AnalysisResult.file_id == file_id
        ).order_by(AnalysisResult.analysis_date.desc()).first()
        
        if not analysis or not analysis.segmentation_data:
            raise HTTPException(
                status_code=404,
                detail="No segmentation found for this file"
            )
        
        # Find segmentation file
        file_path = Path(file.file_path)
        seg_candidates = [
            file_path.parent / f"{file_path.stem}_segmentation.nii.gz",
            file_path.parent / f"{file_path.stem}_segmentation.nii",
            Path("data/segmentation") / f"{file_id}_segmentation.nii.gz",
            Path("data/segmentation") / f"{file_id}_segmentation.nii",
        ]
        seg_path = next((p for p in seg_candidates if p.exists()), None)
        
        if not seg_path:
            raise HTTPException(
                status_code=404,
                detail="Segmentation file not found on disk"
            )
        
        # Generate 3D meshes
        reconstructor = get_reconstruction_service()
        result = reconstructor.reconstruct_all_regions(
            str(seg_path),
            smoothing=smoothing,
            step_size=step_size
        )
        
        logger.info(f"Generated 3D mesh for file {file_id}: "
                   f"{result['num_regions']} regions")
        
        return {
            "file_id": file_id,
            "patient_id": file.patient_id,
            "meshes": result['meshes'],
            "num_regions": result['num_regions'],
            "total_surface_area_mm2": result['total_surface_area_mm2'],
            "bounding_box": result['global_bounding_box'],
            "parameters": {
                "smoothing": smoothing,
                "step_size": step_size
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"3D reconstruction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/viewer-data/{file_id}")
async def get_viewer_data(
    file_id: int,
    format: str = "threejs",  # threejs or vtkjs
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get 3D mesh data formatted for web viewer (Three.js or VTK.js)
    
    Args:
        file_id: File ID with segmentation
        format: Output format (threejs or vtkjs)
    
    Returns:
        Formatted mesh data for web visualization
    """
    try:
        logger.info(f"Getting viewer data for file {file_id}, format={format}")
        
        # Get file
        file = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check access: owner, doctor with permission, or collaborating doctor
        user_id = user.get('user_id') if isinstance(user, dict) else user.user_id
        user_role = user.get('role') if isinstance(user, dict) else user.role
        
        has_access = False
        if file.user_id == user_id:
            has_access = True
        elif user_role in ['doctor', 'radiologist', 'oncologist']:
            # Check if doctor has granted access
            permission = db.query(FileAccessPermission).filter(
                FileAccessPermission.file_id == file_id,
                FileAccessPermission.doctor_id == user_id,
                FileAccessPermission.status == 'active'
            ).first()
            
            if permission:
                has_access = True
            else:
                # Check collaboration
                collab = db.query(CaseCollaboration).filter(
                    CaseCollaboration.file_id == file_id,
                    CaseCollaboration.collaborating_doctor_id == user_id,
                    CaseCollaboration.status == 'active'
                ).first()
                
                if collab:
                    has_access = True
        
        if not has_access:
            logger.warning(f"Access denied for user {user_id} to file {file_id}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Try to generate or load segmentation
        file_path = Path(file.file_path)
        seg_candidates = [
            file_path.parent / f"{file_path.stem}_segmentation.nii.gz",
            file_path.parent / f"{file_path.stem}_segmentation.nii",
            Path("data/segmentation") / f"{file_id}_segmentation.nii.gz",
            Path("data/segmentation") / f"{file_id}_segmentation.nii",
        ]

        seg_path = next((candidate for candidate in seg_candidates if candidate.exists()), None)
        
        if not seg_path:
            error_msg = f"Segmentation file not found for file {file_id}. Tried: {', '.join(str(c) for c in seg_candidates)}"
            logger.warning(error_msg)
            # Return error in response so frontend can display it
            if format == "threejs":
                return {
                    "file_id": file_id,
                    "geometries": [],
                    "error": "Segmentation file not found. Please run analysis first."
                }
            else:  # vtkjs
                return {
                    "file_id": file_id,
                    "regions": [],
                    "error": "Segmentation file not found. Please run analysis first."
                }
        
        # Generate 3D meshes
        logger.info(f"Generating 3D mesh from {seg_path}")
        reconstructor = get_reconstruction_service()
        result = reconstructor.reconstruct_all_regions(
            str(seg_path),
            smoothing=True,
            step_size=2
        )
        
        logger.info(f"Generated {result['num_regions']} regions")
        
        # Format for viewer
        if format == "threejs":
            formatted = reconstructor.prepare_for_threejs(result['meshes'])
            return {
                "file_id": file_id,
                "geometries": formatted.get('geometries', [])
            }
        else:  # vtkjs
            formatted = reconstructor.prepare_for_vtkjs(result['meshes'])
            return {
                "file_id": file_id,
                "regions": formatted.get('regions', [])
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get viewer data for file {file_id}: {e}", exc_info=True)
        # Return minimal fallback mesh instead of error
        if format == "threejs":
            return {
                "file_id": file_id,
                "geometries": [],
                "error": str(e)
            }
        else:
            return {
                "file_id": file_id,
                "regions": [],
                "error": str(e)
            }


@router.get("/mesh/{file_id}/{region}")
async def get_mesh_data(
    file_id: int,
    region: str,  # NCR, ED, or ET
    format: str = "json",  # json, threejs, vtkjs
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get mesh data for specific region in different formats
    
    Args:
        file_id: File ID
        region: Tumor region (NCR, ED, ET)
        format: Output format (json, threejs, vtkjs)
    
    Returns:
        Mesh data in requested format
    """
    try:
        # Generate mesh first
        mesh_result = await generate_3d_mesh(
            file_id=file_id,
            smoothing=True,
            step_size=2,
            user=user,
            db=db
        )
        
        if region not in mesh_result['meshes']:
            raise HTTPException(
                status_code=404,
                detail=f"Region {region} not found"
            )
        
        mesh_data = mesh_result['meshes'][region]
        
        # Format conversion
        reconstructor = get_reconstruction_service()
        
        if format == "threejs":
            formatted_data = reconstructor.prepare_for_threejs({region: mesh_data})
            return formatted_data['geometries'][0]
        
        elif format == "vtkjs":
            formatted_data = reconstructor.prepare_for_vtkjs({region: mesh_data})
            return formatted_data['regions'][0]
        
        else:  # json
            return mesh_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mesh data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/stl/{file_id}/{region}")
async def export_stl(
    file_id: int,
    region: str,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export mesh to STL format for 3D printing or external software
    
    Args:
        file_id: File ID
        region: Tumor region (NCR, ED, ET)
    
    Returns:
        STL file as binary download
    """
    try:
        # Generate mesh
        mesh_result = await generate_3d_mesh(
            file_id=file_id,
            smoothing=True,
            step_size=1,  # High quality for export
            user=user,
            db=db
        )
        
        if region not in mesh_result['meshes']:
            raise HTTPException(
                status_code=404,
                detail=f"Region {region} not found"
            )
        
        mesh_data = mesh_result['meshes'][region]
        
        # Export to temporary file
        output_dir = Path("data/3d_exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stl_path = output_dir / f"tumor_{file_id}_{region}.stl"
        
        reconstructor = get_reconstruction_service()
        reconstructor.export_to_stl(mesh_data, str(stl_path))
        
        # Read and return file
        with open(stl_path, "rb") as f:
            stl_content = f.read()
        
        return Response(
            content=stl_content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=tumor_{region}.stl"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STL export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/obj/{file_id}/{region}")
async def export_obj(
    file_id: int,
    region: str,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export mesh to OBJ format (widely supported 3D format)
    
    Args:
        file_id: File ID
        region: Tumor region (NCR, ED, ET)
    
    Returns:
        OBJ file as text download
    """
    try:
        # Generate mesh
        mesh_result = await generate_3d_mesh(
            file_id=file_id,
            smoothing=True,
            step_size=1,
            user=user,
            db=db
        )
        
        if region not in mesh_result['meshes']:
            raise HTTPException(
                status_code=404,
                detail=f"Region {region} not found"
            )
        
        mesh_data = mesh_result['meshes'][region]
        
        # Export to temporary file
        output_dir = Path("data/3d_exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        obj_path = output_dir / f"tumor_{file_id}_{region}.obj"
        
        reconstructor = get_reconstruction_service()
        reconstructor.export_to_obj(mesh_data, str(obj_path))
        
        # Read and return file
        with open(obj_path, "r") as f:
            obj_content = f.read()
        
        return Response(
            content=obj_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=tumor_{region}.obj"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OBJ export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cross-sections/{file_id}")
async def get_cross_sections(
    file_id: int,
    num_slices: int = 10,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get cross-sectional data for slice-based exploration
    
    Args:
        file_id: File ID
        num_slices: Number of cross-sections to generate
    
    Returns:
        Cross-section positions and metadata
    """
    try:
        # Generate meshes
        mesh_result = await generate_3d_mesh(
            file_id=file_id,
            smoothing=True,
            step_size=2,
            user=user,
            db=db
        )
        
        reconstructor = get_reconstruction_service()
        cross_sections = reconstructor.generate_cross_sections(
            mesh_result['meshes'],
            num_slices=num_slices
        )
        
        return {
            'file_id': file_id,
            'cross_sections': cross_sections
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cross-section generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{file_id}")
async def get_3d_statistics(
    file_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get 3D mesh statistics and quality metrics
    
    Args:
        file_id: File ID
    
    Returns:
        Statistics about 3D reconstruction
    """
    try:
        # Generate meshes
        mesh_result = await generate_3d_mesh(
            file_id=file_id,
            smoothing=True,
            step_size=2,
            user=user,
            db=db
        )
        
        # Calculate statistics
        total_vertices = sum(m['num_vertices'] for m in mesh_result['meshes'].values())
        total_faces = sum(m['num_faces'] for m in mesh_result['meshes'].values())
        
        region_stats = []
        for region_name, mesh in mesh_result['meshes'].items():
            bbox = mesh['bounding_box']
            dimensions = [
                bbox['max'][i] - bbox['min'][i]
                for i in range(3)
            ]
            
            region_stats.append({
                'region': region_name,
                'vertices': mesh['num_vertices'],
                'faces': mesh['num_faces'],
                'surface_area_mm2': mesh['surface_area_mm2'],
                'dimensions_mm': dimensions,
                'color': mesh['color']
            })
        
        return {
            'file_id': file_id,
            'total_vertices': total_vertices,
            'total_faces': total_faces,
            'total_surface_area_mm2': mesh_result['total_surface_area_mm2'],
            'num_regions': mesh_result['num_regions'],
            'regions': region_stats,
            'bounding_box': mesh_result['bounding_box']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistics calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
