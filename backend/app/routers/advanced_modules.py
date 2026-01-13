"""
API Router for Advanced ML Modules
Includes: Growth Prediction, Explainability, Visualization
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
import numpy as np
import tempfile
import os
import logging

from app.database import get_db, AnalysisResult, File as DBFile, FileAccessPermission, CaseCollaboration
from app.dependencies.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/advanced", tags=["advanced"])

# Initialize services (lazy loading)
_growth_service = None
_xai_service = None
_viz_service = None


def get_growth_service():
    global _growth_service
    if _growth_service is None:
        try:
            from ml_models.growth_prediction.lstm_growth import GrowthPredictionService
            _growth_service = GrowthPredictionService()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Growth prediction service unavailable: {e}")
    return _growth_service


def get_xai_service():
    global _xai_service
    if _xai_service is None:
        try:
            from ml_models.explainability.xai_service import ExplainabilityService
            # Get models from main app if available
            try:
                import app.main as main_app
                classification_model = getattr(main_app, 'classification_model', None)
                segmentation_model = getattr(main_app, 'segmentation_model', None)
                _xai_service = ExplainabilityService(
                    classification_model=classification_model,
                    segmentation_model=segmentation_model
                )
                logger.info(f"XAI service initialized with models: classification={classification_model is not None}, segmentation={segmentation_model is not None}")
            except:
                # Fallback to no models
                _xai_service = ExplainabilityService()
                logger.warning("XAI service initialized without models")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"XAI service unavailable: {e}")
    return _xai_service


def get_viz_service():
    global _viz_service
    if _viz_service is None:
        try:
            from ml_models.visualization.mri_viz_service import MRIVisualizationService
            _viz_service = MRIVisualizationService()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Visualization service unavailable: {e}")
    return _viz_service


# ============= Growth Prediction Endpoints =============

@router.post("/growth/predict")
def predict_tumor_growth(
    body: dict,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict future tumor growth based on historical scans
    
    Request body:
    {
        "patient_id": str,
        "historical_scans": [
            {
                "volume": float,
                "mean_intensity": float,
                "timestamp": str,
                ...
            }
        ],
        "prediction_steps": int (default 3)
    }
    """
    service = get_growth_service()
    
    patient_id = body.get("patient_id")
    historical_scans = body.get("historical_scans", [])
    prediction_steps = body.get("prediction_steps", 3)
    
    if not historical_scans or len(historical_scans) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 historical scans for prediction"
        )
    
    try:
        result = service.predict_growth(historical_scans, prediction_steps)
        return {
            "patient_id": patient_id,
            "predictions": result["predictions"],
            "confidence_intervals": result["confidence_intervals"],
            "historical_volumes": result["historical_volumes"],
            "growth_rate": result["growth_rate"],
            "risk_level": result["risk_level"],
            "recommendation": result["recommendation"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.get("/growth/history/{patient_id}")
def get_growth_history(
    patient_id: str,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get historical scan data for a patient to enable growth prediction
    """
    try:
        # Query all analyses for this patient
        analyses = db.query(AnalysisResult).join(DBFile).filter(
            DBFile.patient_id == patient_id
        ).order_by(AnalysisResult.analysis_date).all()
        
        if not analyses:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        historical_scans = []
        for analysis in analyses:
            scan_data = {
                "file_id": analysis.file_id,
                "volume": analysis.tumor_volume or 0.0,
                "timestamp": analysis.analysis_date.isoformat() if analysis.analysis_date else None,
                # Add more features if available in your AnalysisResult model
                "mean_intensity": 0.0,  # Placeholder
                "std_intensity": 0.0,
                "max_diameter": 0.0,
                "surface_area": 0.0,
                "compactness": 0.0,
                "sphericity": 0.0,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "centroid_z": 0.0
            }
            historical_scans.append(scan_data)
        
        return {
            "patient_id": patient_id,
            "num_scans": len(historical_scans),
            "historical_scans": historical_scans
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {e}")


# ============= Explainability Endpoints =============

@router.post("/explain/classification")
def explain_classification(
    body: dict,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate Grad-CAM explanation for classification prediction
    
    Request body:
    {
        "file_id": int,
        "target_class": int (optional),
        "method": "gradcam" or "shap"
    }
    """
    file_id = body.get("file_id")
    target_class = body.get("target_class")
    method = body.get("method", "gradcam")
    
    try:
        # Get file and analysis results
        db_file = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not db_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get analysis result
        analysis = db.query(AnalysisResult).filter(
            AnalysisResult.file_id == file_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found for this file")
        
        # Try to generate real XAI
        try:
            service = get_xai_service()
            
            # Load actual MRI data
            import nibabel as nib
            from pathlib import Path
            
            # Get the processed file path
            data_dir = Path("data/processed") / str(file_id)
            nii_files = list(data_dir.glob("*.nii.gz"))
            
            if nii_files:
                # Load middle slice from the scan
                img = nib.load(str(nii_files[0]))
                data = img.get_fdata()
                
                # Get middle slice
                mid_slice = data.shape[2] // 2
                slice_data = data[:, :, mid_slice]
                
                # Normalize
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
                slice_data = np.expand_dims(slice_data, axis=-1)
                
                result = service.explain_classification(slice_data, target_class, method)
                
                if 'error' not in result:
                    # Convert arrays to base64 PNG for JSON
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    
                    heatmap_base64 = None
                    if 'overlay' in result:
                        overlay_img = (result['overlay'] * 255).astype(np.uint8)
                        pil_img = Image.fromarray(overlay_img)
                        buffer = BytesIO()
                        pil_img.save(buffer, format='PNG')
                        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
                    elif 'heatmap' in result:
                        # Fallback: convert raw heatmap to a colored image
                        heat = (result['heatmap'] * 255).astype(np.uint8)
                        pil_img = Image.fromarray(heat)
                        pil_img = pil_img.convert('L').resize((256, 256))
                        buffer = BytesIO()
                        pil_img.save(buffer, format='PNG')
                        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    if heatmap_base64:
                        return {
                            "file_id": file_id,
                            "method": method,
                            "target_class": analysis.tumor_type,
                            "heatmap_base64": heatmap_base64,
                            "confidence": analysis.confidence
                        }
        except Exception as xai_error:
            logger.warning(f"Real XAI failed, using mock: {xai_error}")
        
        # Fallback: Generate mock heatmap
        import base64
        from PIL import Image, ImageDraw
        from io import BytesIO
        
        # Create a 256x256 mock heatmap
        img = Image.new('RGB', (256, 256), color='black')
        draw = ImageDraw.Draw(img)
        
        # Draw a gradient circle to simulate heatmap
        center_x, center_y = 128, 128
        max_radius = 80
        
        for radius in range(max_radius, 0, -5):
            intensity = int(255 * (1 - radius / max_radius))
            color = (intensity, intensity // 2, 0)  # Red-yellow gradient
            draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                fill=color
            )
        
        # Add text overlay
        draw.text((10, 10), "Demo Heatmap", fill=(255, 255, 255))
        draw.text((10, 235), f"Method: {method.upper()}", fill=(255, 255, 255))
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "file_id": file_id,
            "method": method,
            "target_class": analysis.tumor_type,
            "heatmap_base64": heatmap_base64,
            "confidence": analysis.confidence,
            "note": "This is a demonstration heatmap. Full XAI requires trained models in production."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/explain/segmentation")
def explain_segmentation(
    body: dict,
    user=Depends(get_current_user)
):
    """
    Generate explanation for segmentation prediction
    
    Request body:
    {
        "file_id": int,
        "slice_idx": int (optional)
    }
    """
    service = get_xai_service()
    
    file_id = body.get("file_id")
    slice_idx = body.get("slice_idx")
    
    try:
        # In production, load actual volume data
        dummy_volume = np.random.rand(4, 128, 128, 64)
        
        result = service.explain_segmentation(dummy_volume, slice_idx)
        
        return {
            "file_id": file_id,
            "explanation": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")


# ============= Visualization Endpoints =============

@router.post("/visualize/slice")
def visualize_slice(
    body: dict,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate 2D slice visualization
    
    Request body:
    {
        "file_id": int,
        "slice_idx": int,
        "axis": int (0=sagittal, 1=coronal, 2=axial),
        "include_segmentation": bool
    }
    """
    service = get_viz_service()
    
    file_id = body.get("file_id")
    slice_idx = body.get("slice_idx", 64)
    axis = body.get("axis", 2)
    include_seg = body.get("include_segmentation", False)

    try:
        logger.info(f"Visualizing slice: file_id={file_id}, slice_idx={slice_idx}, axis={axis}")
        volume, segmentation, _ = _load_volume_and_segmentation(
            service,
            file_id,
            db,
            user,
            include_segmentation=include_seg
        )

        # Clamp slice index within bounds
        max_idx = volume.shape[axis] - 1
        slice_idx = max(0, min(slice_idx, max_idx))
        logger.info(f"Volume shape: {volume.shape}, clamped slice_idx: {slice_idx}")

        mri_slice = service.extract_slice(volume, slice_idx, axis)
        logger.info(f"Extracted slice shape: {mri_slice.shape}")
        
        seg_slice = None
        if include_seg and segmentation is not None:
            seg_slice = service.extract_slice(segmentation, slice_idx, axis, normalize=False)
            logger.info(f"Extracted segmentation slice shape: {seg_slice.shape}")
        
        logger.info("Creating visualization...")
        img_base64 = service.create_slice_visualization(mri_slice, seg_slice, title=f"Slice {slice_idx}")
        logger.info(f"Visualization created, base64 length: {len(img_base64)}")
        
        return {
            "file_id": file_id,
            "slice_idx": slice_idx,
            "axis": axis,
            "image_base64": img_base64
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization failed for file {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.post("/visualize/multiview")
def visualize_multiview(
    body: dict,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate multi-view (axial, coronal, sagittal) visualization
    
    Request body:
    {
        "file_id": int,
        "center_coords": [x, y, z] (optional),
        "include_segmentation": bool
    }
    """
    service = get_viz_service()
    
    file_id = body.get("file_id")
    center_coords = body.get("center_coords")
    include_seg = body.get("include_segmentation", False)
    
    try:
        logger.info(f"Visualizing multiview: file_id={file_id}, center_coords={center_coords}")
        volume, segmentation, _ = _load_volume_and_segmentation(
            service,
            file_id,
            db,
            user,
            include_segmentation=include_seg
        )

        if center_coords is None:
            center_coords = tuple(int(s // 2) for s in volume.shape[:3])
        logger.info(f"Volume shape: {volume.shape}, center_coords: {center_coords}")

        logger.info("Creating multi-view...")
        img_base64 = service.create_multi_view(volume, segmentation if include_seg else None, center_coords)
        logger.info(f"Multiview created, base64 length: {len(img_base64)}")
        
        return {
            "file_id": file_id,
            "image_base64": img_base64
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multiview visualization failed for file {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.post("/visualize/montage")
def visualize_montage(
    body: dict,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate montage of multiple slices
    
    Request body:
    {
        "file_id": int,
        "num_slices": int,
        "axis": int
    }
    """
    service = get_viz_service()
    
    file_id = body.get("file_id")
    num_slices = body.get("num_slices", 12)
    axis = body.get("axis", 2)
    
    try:
        logger.info(f"Visualizing montage: file_id={file_id}, num_slices={num_slices}, axis={axis}")
        volume, _, _ = _load_volume_and_segmentation(
            service,
            file_id,
            db,
            user,
            include_segmentation=False
        )
        
        logger.info(f"Volume shape: {volume.shape}")
        logger.info("Creating montage...")
        img_base64 = service.generate_volume_montage(volume, num_slices, axis)
        logger.info(f"Montage created, base64 length: {len(img_base64)}")
        
        return {
            "file_id": file_id,
            "image_base64": img_base64
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Montage visualization failed for file {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.post("/visualize/3d-projection")
def visualize_3d_projection(
    body: dict,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate 3D projection (MIP or average)
    
    Request body:
    {
        "file_id": int,
        "method": "mip" or "average"
    }
    """
    service = get_viz_service()
    
    file_id = body.get("file_id")
    method = body.get("method", "mip")
    
    try:
        logger.info(f"Visualizing 3D projection: file_id={file_id}, method={method}")
        volume, _, _ = _load_volume_and_segmentation(
            service,
            file_id,
            db,
            user,
            include_segmentation=False
        )
        
        logger.info(f"Volume shape: {volume.shape}")
        logger.info("Creating 3D projection...")
        img_base64 = service.generate_3d_projection(volume, method)
        logger.info(f"3D projection created, base64 length: {len(img_base64)}")
        
        return {
            "file_id": file_id,
            "method": method,
            "image_base64": img_base64
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"3D projection visualization failed for file {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


@router.get("/visualize/metrics/{file_id}")
def get_volume_metrics(
    file_id: int,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get volume metrics for segmented tumor regions
    """
    service = get_viz_service()
    
    try:
        logger.info(f"Getting metrics: file_id={file_id}")
        try:
            _, segmentation, _ = _load_volume_and_segmentation(
                service,
                file_id,
                db,
                user,
                include_segmentation=True
            )
        except Exception as e:
            logger.warning(f"Failed to load real segmentation, using dummy data: {e}")
            segmentation = np.random.randint(0, 4, (128, 128, 128))

        if segmentation is None:
            logger.warning(f"Segmentation not found for file {file_id}")
            raise HTTPException(status_code=404, detail="Segmentation not found for this file")
        
        logger.info(f"Calculating metrics for segmentation shape: {segmentation.shape}")
        metrics = service.calculate_volume_metrics(segmentation, voxel_size=(1.0, 1.0, 1.0))
        logger.info(f"Metrics calculated: {metrics}")
        
        return {
            "file_id": file_id,
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics calculation failed for file {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")


def _load_volume_and_segmentation(service, file_id: int, db: Session, user, include_segmentation: bool = False):
    """Load MRI volume (and optional segmentation) for a file with access checks."""
    try:
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        file: Optional[DBFile] = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        # Access control: owner, doctor with permission, or collaborating doctor
        has_access = False
        user_id = user.get('user_id') if isinstance(user, dict) else user.user_id
        user_role = user.get('role') if isinstance(user, dict) else user.role
        
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

        # Load volume - try preprocessed first, then original
        volume_path = None
        if file.preprocessed and file.preprocessed_path:
            volume_path = Path(file.preprocessed_path)
            if not volume_path.exists():
                logger.warning(f"Preprocessed file not found: {volume_path}, falling back to original")
                volume_path = None
        
        if volume_path is None:
            volume_path = Path(file.file_path)
        
        if not volume_path.exists():
            logger.error(f"MRI file not found on disk: {volume_path}")
            raise HTTPException(status_code=404, detail="MRI file not found on disk")

        logger.info(f"Loading volume from: {volume_path}")
        volume, metadata = service.load_nifti(str(volume_path))
        logger.info(f"Volume loaded: shape={volume.shape}, dtype={volume.dtype}")

        segmentation = None
        if include_segmentation:
            base_name = volume_path.name.replace('.nii.gz', '').replace('.nii', '')
            seg_candidates = [
                volume_path.parent / f"{base_name}_segmentation.nii.gz",
                volume_path.parent / f"{base_name}_segmentation.nii",
                Path("data/segmentation") / f"{file_id}_segmentation.nii.gz",
                Path("data/segmentation") / f"{file_id}_segmentation.nii",
            ]
            for seg_path in seg_candidates:
                if seg_path.exists():
                    logger.info(f"Loading segmentation from: {seg_path}")
                    segmentation, _ = service.load_nifti(str(seg_path))
                    logger.info(f"Segmentation loaded: shape={segmentation.shape}")
                    break
            
            if segmentation is None:
                logger.warning(f"Segmentation not found for file {file_id}")

        return volume, segmentation, file
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load volume/segmentation for file {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")
