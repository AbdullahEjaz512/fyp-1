"""
Module 2: MRI Upload & Preprocessing
Handles MRI file uploads, DICOM validation, preprocessing pipeline
Implements FR8.1 to FR8.8
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import nibabel as nib
    import SimpleITK as sitk
    import pydicom
    from skimage import filters
    IMAGING_LIBS_AVAILABLE = True
except ImportError:
    IMAGING_LIBS_AVAILABLE = False
    print("Warning: Medical imaging libraries not yet installed")

try:
    from config import (
        UPLOADS_DIR,
        MAX_UPLOAD_SIZE,
        ALLOWED_EXTENSIONS,
        MRI_MODALITIES,
        PREPROCESSING
    )
except ImportError:
    # Fallback configuration
    UPLOADS_DIR = Path("data/uploads")
    MAX_UPLOAD_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
    ALLOWED_EXTENSIONS = [".nii", ".nii.gz", ".dcm", ".dicom"]
    MRI_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
    PREPROCESSING = {
        "normalize_method": "z_score",
        "clip_percentiles": (1, 99),
        "target_spacing": (1.0, 1.0, 1.0),
        "target_size": (128, 128, 128),
        "noise_reduction": True,
        "bias_correction": True,
    }


class MRIFileValidator:
    """
    MRI File Validation Service - FR8.1, FR8.5
    Validates uploaded MRI files for format compliance and brain MRI characteristics
    """
    
    # Expected brain MRI characteristics
    BRAIN_MRI_SPECS = {
        "min_dimensions": 3,  # Must be 3D
        "typical_shape_ranges": {
            "min": (100, 100, 50),    # Minimum expected dimensions
            "max": (512, 512, 512),   # Maximum expected dimensions
            "brats_standard": (240, 240, 155)  # BraTS standard dimensions
        },
        "intensity_ranges": {
            "min_unique_values": 100,  # Brain MRI should have many unique intensity values
            "min_nonzero_ratio": 0.05,  # At least 5% of voxels should be non-zero (brain tissue)
            "max_nonzero_ratio": 0.70,  # No more than 70% non-zero (rest is background/air)
        }
    }
    
    @staticmethod
    def validate_file_extension(filename: str) -> Tuple[bool, str]:
        """
        Validate file extension - FR8.1
        Only DICOM and NIfTI files accepted
        """
        file_ext = Path(filename).suffix.lower()
        
        # Check for double extension (.nii.gz)
        if filename.lower().endswith('.nii.gz'):
            file_ext = '.nii.gz'
        
        if file_ext in ALLOWED_EXTENSIONS:
            return True, f"Valid file format: {file_ext}"
        else:
            return False, f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    @staticmethod
    def validate_file_size(file_size: int) -> Tuple[bool, str]:
        """
        Validate file size - FR8.1, FR8.4
        Maximum 5GB as per SRS
        """
        if file_size <= MAX_UPLOAD_SIZE:
            size_mb = file_size / (1024 * 1024)
            return True, f"File size: {size_mb:.2f} MB"
        else:
            max_gb = MAX_UPLOAD_SIZE / (1024 * 1024 * 1024)
            return False, f"File too large. Maximum size: {max_gb} GB"
    
    @staticmethod
    def validate_file_integrity(filepath: Path) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate file can be opened and read - checks for corruption
        Returns: (is_valid, message, metadata_dict or None)
        """
        if not IMAGING_LIBS_AVAILABLE:
            return True, "Imaging libraries not available for deep validation", None
        
        filepath = Path(filepath)
        
        try:
            if filepath.suffix.lower() in ['.dcm', '.dicom']:
                # Validate DICOM
                dcm = pydicom.dcmread(str(filepath))
                if not hasattr(dcm, 'pixel_array'):
                    return False, "DICOM file has no image data (pixel_array missing)", None
                data = dcm.pixel_array
                metadata = {
                    "modality": str(getattr(dcm, 'Modality', 'Unknown')),
                    "shape": data.shape,
                    "dtype": str(data.dtype)
                }
            else:
                # Validate NIfTI
                img = nib.load(str(filepath))
                data = img.get_fdata()
                metadata = {
                    "shape": data.shape,
                    "dtype": str(data.dtype),
                    "voxel_spacing": img.header.get_zooms() if hasattr(img.header, 'get_zooms') else None
                }
            
            # Check data is not empty
            if data.size == 0:
                return False, "File contains no image data (empty array)", None
            
            # Check for NaN or Inf values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                return False, "File contains invalid values (NaN or Inf)", metadata
            
            return True, "File integrity validated successfully", metadata
            
        except Exception as e:
            return False, f"File is corrupted or unreadable: {str(e)}", None
    
    @staticmethod
    def validate_brain_mri_characteristics(filepath: Path) -> Tuple[bool, str, Dict]:
        """
        Validate that the file appears to be a brain MRI scan.
        Checks dimensions, intensity patterns, and structural characteristics.
        
        Returns: (is_valid, message, validation_details)
        """
        validation_details = {
            "checks_performed": [],
            "warnings": [],
            "is_brain_mri_likely": False,
            "confidence": 0.0
        }
        
        if not IMAGING_LIBS_AVAILABLE:
            validation_details["warnings"].append("Deep validation skipped - imaging libraries unavailable")
            return True, "Basic validation passed (detailed check skipped)", validation_details
        
        filepath = Path(filepath)
        specs = MRIFileValidator.BRAIN_MRI_SPECS
        confidence_score = 0
        total_checks = 5
        
        try:
            # Load the image data
            if filepath.suffix.lower() in ['.dcm', '.dicom']:
                dcm = pydicom.dcmread(str(filepath))
                data = dcm.pixel_array.astype(np.float32)
                # DICOM might be 2D slice - this is acceptable
                is_single_slice = data.ndim == 2
                if is_single_slice:
                    validation_details["checks_performed"].append("DICOM single slice detected")
                    confidence_score += 1  # Single DICOM slice is valid
            else:
                img = nib.load(str(filepath))
                data = img.get_fdata()
                is_single_slice = False
            
            # Check 1: Dimensionality (must be 2D or 3D)
            validation_details["checks_performed"].append(f"Dimensions: {data.ndim}D, Shape: {data.shape}")
            
            if data.ndim < 2 or data.ndim > 4:
                return False, f"Invalid dimensions: Expected 2D-4D, got {data.ndim}D", validation_details
            
            # For 4D data (multiple modalities), use first volume
            if data.ndim == 4:
                data = data[..., 0]
                validation_details["warnings"].append("4D data detected - validating first volume only")
            
            confidence_score += 1
            
            # Check 2: Shape within expected range (for 3D data)
            if not is_single_slice and data.ndim == 3:
                shape = data.shape
                min_shape = specs["typical_shape_ranges"]["min"]
                max_shape = specs["typical_shape_ranges"]["max"]
                
                shape_valid = all(
                    min_shape[i] <= shape[i] <= max_shape[i] 
                    for i in range(min(3, len(shape)))
                )
                
                if shape_valid:
                    confidence_score += 1
                    validation_details["checks_performed"].append(f"Shape {shape} within expected range")
                else:
                    validation_details["warnings"].append(
                        f"Shape {shape} outside typical brain MRI range "
                        f"(expected {min_shape} to {max_shape})"
                    )
            else:
                confidence_score += 0.5  # Partial credit for 2D DICOM
            
            # Check 3: Intensity diversity (not a blank/uniform image)
            unique_values = len(np.unique(data.flatten()[:10000]))  # Sample for speed
            min_unique = specs["intensity_ranges"]["min_unique_values"]
            
            if unique_values >= min_unique:
                confidence_score += 1
                validation_details["checks_performed"].append(
                    f"Intensity diversity OK ({unique_values} unique values)"
                )
            else:
                validation_details["warnings"].append(
                    f"Low intensity diversity ({unique_values} unique values) - "
                    "may be blank or corrupted image"
                )
            
            # Check 4: Non-zero ratio (brain tissue vs background)
            total_voxels = data.size
            nonzero_voxels = np.count_nonzero(data)
            nonzero_ratio = nonzero_voxels / total_voxels
            
            min_ratio = specs["intensity_ranges"]["min_nonzero_ratio"]
            max_ratio = specs["intensity_ranges"]["max_nonzero_ratio"]
            
            if min_ratio <= nonzero_ratio <= max_ratio:
                confidence_score += 1
                validation_details["checks_performed"].append(
                    f"Tissue ratio OK ({nonzero_ratio:.1%} non-zero voxels)"
                )
            else:
                validation_details["warnings"].append(
                    f"Unusual tissue ratio ({nonzero_ratio:.1%}) - "
                    f"expected {min_ratio:.0%}-{max_ratio:.0%}"
                )
            
            # Check 5: Basic intensity statistics (should look like MRI)
            if nonzero_voxels > 0:
                nonzero_data = data[data > 0]
                mean_intensity = np.mean(nonzero_data)
                std_intensity = np.std(nonzero_data)
                cv = std_intensity / mean_intensity if mean_intensity > 0 else 0  # Coefficient of variation
                
                # Brain MRI typically has moderate intensity variation
                if 0.2 < cv < 2.0:
                    confidence_score += 1
                    validation_details["checks_performed"].append(
                        f"Intensity statistics OK (CV={cv:.2f})"
                    )
                else:
                    validation_details["warnings"].append(
                        f"Unusual intensity distribution (CV={cv:.2f})"
                    )
            
            # Calculate final confidence
            confidence = confidence_score / total_checks
            validation_details["confidence"] = round(confidence * 100, 1)
            validation_details["is_brain_mri_likely"] = confidence >= 0.6
            
            if confidence >= 0.8:
                message = f"High confidence brain MRI ({validation_details['confidence']}%)"
                is_valid = True
            elif confidence >= 0.6:
                message = f"Likely brain MRI ({validation_details['confidence']}%) - some characteristics differ from typical"
                is_valid = True
            elif confidence >= 0.4:
                message = f"Uncertain if brain MRI ({validation_details['confidence']}%) - proceed with caution"
                is_valid = True  # Allow but warn
            else:
                message = f"Unlikely to be valid brain MRI ({validation_details['confidence']}%)"
                is_valid = False
            
            return is_valid, message, validation_details
            
        except Exception as e:
            validation_details["warnings"].append(f"Validation error: {str(e)}")
            return False, f"Failed to validate brain MRI characteristics: {str(e)}", validation_details
    
    @staticmethod
    def comprehensive_validation(filepath: Path, filename: str = None) -> Dict:
        """
        Perform all validation checks and return comprehensive results.
        
        Returns dictionary with:
        - is_valid: Overall validity
        - can_process: Whether file can be processed (may be valid but with warnings)
        - checks: Individual check results
        - recommendations: Suggestions for the user
        """
        filepath = Path(filepath)
        filename = filename or filepath.name
        
        result = {
            "is_valid": True,
            "can_process": True,
            "filename": filename,
            "checks": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check 1: File extension
        ext_valid, ext_msg = MRIFileValidator.validate_file_extension(filename)
        result["checks"]["extension"] = {"valid": ext_valid, "message": ext_msg}
        if not ext_valid:
            result["is_valid"] = False
            result["can_process"] = False
            result["errors"].append(ext_msg)
            return result  # No point continuing if wrong file type
        
        # Check 2: File size
        if filepath.exists():
            size_valid, size_msg = MRIFileValidator.validate_file_size(filepath.stat().st_size)
            result["checks"]["size"] = {"valid": size_valid, "message": size_msg}
            if not size_valid:
                result["is_valid"] = False
                result["can_process"] = False
                result["errors"].append(size_msg)
                return result
        
        # Check 3: File integrity
        integrity_valid, integrity_msg, metadata = MRIFileValidator.validate_file_integrity(filepath)
        result["checks"]["integrity"] = {
            "valid": integrity_valid, 
            "message": integrity_msg,
            "metadata": metadata
        }
        if not integrity_valid:
            result["is_valid"] = False
            result["can_process"] = False
            result["errors"].append(integrity_msg)
            result["recommendations"].append("Please ensure the file is not corrupted and try uploading again")
            return result
        
        # Check 4: Brain MRI characteristics
        brain_valid, brain_msg, brain_details = MRIFileValidator.validate_brain_mri_characteristics(filepath)
        result["checks"]["brain_mri"] = {
            "valid": brain_valid,
            "message": brain_msg,
            "details": brain_details
        }
        
        if not brain_valid:
            result["is_valid"] = False
            result["can_process"] = False
            result["errors"].append(brain_msg)
            result["recommendations"].append(
                "Please upload a valid brain MRI scan (NIfTI or DICOM format). "
                "The file should be a 3D volumetric brain scan or 2D DICOM slice."
            )
        elif brain_details.get("warnings"):
            result["warnings"].extend(brain_details["warnings"])
            result["recommendations"].append(
                "The file passed basic validation but has some unusual characteristics. "
                "Analysis results should be reviewed carefully."
            )
        
        # Add confidence info
        if brain_details.get("confidence"):
            result["brain_mri_confidence"] = brain_details["confidence"]
        
        return result
    
    @staticmethod
    def is_dicom_file(filepath: Path) -> bool:
        """Check if file is valid DICOM"""
        if not IMAGING_LIBS_AVAILABLE:
            return filepath.suffix.lower() in ['.dcm', '.dicom']
        
        try:
            pydicom.dcmread(str(filepath))
            return True
        except:
            return False
    
    @staticmethod
    def is_nifti_file(filepath: Path) -> bool:
        """Check if file is valid NIfTI"""
        if not IMAGING_LIBS_AVAILABLE:
            return filepath.suffix.lower() in ['.nii', '.nii.gz']
        
        try:
            nib.load(str(filepath))
            return True
        except:
            return False
    
    @staticmethod
    def validate_dicom_metadata(filepath: Path) -> Dict:
        """
        Extract and validate DICOM metadata - FR8.5
        Returns metadata or raises exception
        """
        if not IMAGING_LIBS_AVAILABLE:
            return {"error": "Imaging libraries not installed"}
        
        try:
            dcm = pydicom.dcmread(str(filepath))
            
            metadata = {
                "patient_id": str(getattr(dcm, 'PatientID', 'Unknown')),
                "patient_name": str(getattr(dcm, 'PatientName', 'Unknown')),
                "study_date": str(getattr(dcm, 'StudyDate', 'Unknown')),
                "modality": str(getattr(dcm, 'Modality', 'Unknown')),
                "series_description": str(getattr(dcm, 'SeriesDescription', 'Unknown')),
                "institution": str(getattr(dcm, 'InstitutionName', 'Unknown')),
                "manufacturer": str(getattr(dcm, 'Manufacturer', 'Unknown')),
                "image_shape": (
                    int(getattr(dcm, 'Rows', 0)),
                    int(getattr(dcm, 'Columns', 0))
                ),
                "validation_status": "valid",
                "extracted_at": datetime.utcnow().isoformat()
            }
            
            return metadata
            
        except Exception as e:
            return {
                "validation_status": "invalid",
                "error": str(e)
            }


class MRIPreprocessor:
    """
    MRI Preprocessing Pipeline - FR8.6, FR8.7, FR8.8
    Handles intensity normalization and noise reduction
    """
    
    def __init__(self, config: Dict = None):
        """Initialize preprocessor with configuration"""
        self.config = config or PREPROCESSING
        self.processing_steps = []
    
    def extract_metadata(self, filepath: str) -> Dict:
        """
        Extract metadata from NIfTI or DICOM file
        Returns comprehensive metadata about the MRI scan
        """
        filepath = Path(filepath)
        
        # Check if DICOM
        if filepath.suffix.lower() in ['.dcm', '.dicom']:
            validator = MRIFileValidator()
            return validator.validate_dicom_metadata(filepath)
        
        # Handle NIfTI files
        if not IMAGING_LIBS_AVAILABLE:
            return {
                "error": "Imaging libraries not installed",
                "filename": filepath.name,
                "file_size": filepath.stat().st_size if filepath.exists() else 0
            }
        
        try:
            img = nib.load(str(filepath))
            data = img.get_fdata()
            
            metadata = {
                "filename": filepath.name,
                "file_size": filepath.stat().st_size,
                "dimensions": data.shape,
                "voxel_spacing": img.header.get_zooms() if hasattr(img.header, 'get_zooms') else None,
                "data_type": str(data.dtype),
                "min_intensity": float(np.min(data)),
                "max_intensity": float(np.max(data)),
                "mean_intensity": float(np.mean(data)),
                "std_intensity": float(np.std(data)),
                "non_zero_voxels": int(np.count_nonzero(data)),
                "total_voxels": int(data.size),
                "extracted_at": datetime.utcnow().isoformat()
            }
            
            return metadata
            
        except Exception as e:
            return {
                "error": str(e),
                "filename": filepath.name,
                "extracted_at": datetime.utcnow().isoformat()
            }
    
    def preprocess_pipeline(
        self,
        input_path: str,
        output_path: str,
        normalize: str = "zscore",
        denoise: bool = True,
        bias_correction: bool = True
    ) -> Dict:
        """
        Complete preprocessing pipeline with configurable options
        
        Args:
            input_path: Path to input NIfTI file
            output_path: Path to save preprocessed file
            normalize: Normalization method ('zscore' or 'minmax')
            denoise: Whether to apply noise reduction
            bias_correction: Whether to apply bias field correction
        
        Returns:
            Dictionary with preprocessing results and statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Update config based on parameters
        temp_config = self.config.copy()
        temp_config["normalize_method"] = "z_score" if normalize == "zscore" else "min_max"
        temp_config["noise_reduction"] = denoise
        temp_config["bias_correction"] = bias_correction
        
        # Save current config and use temp config
        original_config = self.config
        self.config = temp_config
        
        try:
            # Run preprocessing
            preprocessed_data, info = self.preprocess_mri(input_path, output_path)
            
            # Restore original config
            self.config = original_config
            
            return {
                "success": True,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "preprocessing_info": info,
                "parameters": {
                    "normalize": normalize,
                    "denoise": denoise,
                    "bias_correction": bias_correction
                }
            }
            
        except Exception as e:
            # Restore original config
            self.config = original_config
            raise Exception(f"Preprocessing pipeline failed: {str(e)}")
    
    def load_nifti(self, filepath: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """Load NIfTI file and return data array and image object"""
        if not IMAGING_LIBS_AVAILABLE:
            raise ImportError("nibabel not installed")
        
        img = nib.load(str(filepath))
        data = img.get_fdata()
        return data, img
    
    def load_dicom(self, filepath: Path) -> np.ndarray:
        """Load DICOM file and return data array"""
        if not IMAGING_LIBS_AVAILABLE:
            raise ImportError("pydicom not installed")
        
        try:
            dcm = pydicom.dcmread(str(filepath))
            data = dcm.pixel_array.astype(np.float32)
            
            # Add a channel dimension if 2D
            if data.ndim == 2:
                data = data[:, :, np.newaxis]
            
            return data
        except Exception as e:
            raise Exception(f"Failed to load DICOM file: {str(e)}")
    
    def load_image_data(self, filepath: Path) -> Tuple[np.ndarray, Optional[nib.Nifti1Image]]:
        """
        Load image data from NIfTI or DICOM file
        Returns (data_array, nifti_image_object or None)
        """
        filepath = Path(filepath)
        
        # Check file type
        if filepath.suffix.lower() in ['.dcm', '.dicom']:
            # DICOM file
            data = self.load_dicom(filepath)
            return data, None
        elif filepath.suffix.lower() in ['.nii'] or str(filepath).lower().endswith('.nii.gz'):
            # NIfTI file
            data, img = self.load_nifti(filepath)
            return data, img
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def normalize_intensity(self, data: np.ndarray, method: str = "z_score") -> np.ndarray:
        """
        Normalize MRI intensity values - FR8.6
        
        Methods:
        - z_score: Standardize to zero mean, unit variance
        - min_max: Scale to [0, 1] range
        """
        self.processing_steps.append(f"normalize_{method}")
        
        if method == "z_score":
            # Z-score normalization
            mean = np.mean(data[data > 0])  # Exclude background
            std = np.std(data[data > 0])
            
            if std > 0:
                data = (data - mean) / std
            
        elif method == "min_max":
            # Min-max normalization to [0, 1]
            min_val = np.min(data)
            max_val = np.max(data)
            
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val)
        
        return data
    
    def clip_intensity_outliers(
        self, 
        data: np.ndarray, 
        lower_percentile: float = 1, 
        upper_percentile: float = 99
    ) -> np.ndarray:
        """
        Clip intensity outliers - FR8.6
        Removes extreme values that could affect normalization
        """
        self.processing_steps.append("clip_outliers")
        
        lower_bound = np.percentile(data[data > 0], lower_percentile)
        upper_bound = np.percentile(data[data > 0], upper_percentile)
        
        data = np.clip(data, lower_bound, upper_bound)
        return data
    
    def reduce_noise(self, data: np.ndarray, method: str = "gaussian") -> np.ndarray:
        """
        Apply noise reduction - FR8.7
        
        Methods:
        - gaussian: Gaussian smoothing
        - median: Median filtering
        - bilateral: Edge-preserving bilateral filter
        """
        if not IMAGING_LIBS_AVAILABLE:
            return data
        
        self.processing_steps.append(f"denoise_{method}")
        
        if method == "gaussian":
            # Gaussian smoothing
            data = filters.gaussian(data, sigma=0.5, preserve_range=True)
        
        elif method == "median":
            # Median filtering (more computational)
            from scipy.ndimage import median_filter
            data = median_filter(data, size=3)
        
        return data
    
    def bias_field_correction(self, data: np.ndarray) -> np.ndarray:
        """
        N4 Bias Field Correction using SimpleITK
        Corrects intensity inhomogeneity in MRI
        """
        if not IMAGING_LIBS_AVAILABLE:
            return data
        
        self.processing_steps.append("bias_correction")
        
        try:
            # Convert to SimpleITK image
            image = sitk.GetImageFromArray(data.astype(np.float32))
            
            # Apply N4 bias correction
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrected_image = corrector.Execute(image)
            
            # Convert back to numpy array
            corrected_data = sitk.GetArrayFromImage(corrected_image)
            return corrected_data
            
        except Exception as e:
            print(f"Bias correction failed: {e}. Returning original data.")
            return data
    
    def preprocess_mri(
        self, 
        filepath: Path, 
        output_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete preprocessing pipeline - FR8.6, FR8.7, FR8.8
        
        Steps:
        1. Load MRI data
        2. Clip intensity outliers
        3. Apply noise reduction (if enabled)
        4. Apply bias correction (if enabled)
        5. Normalize intensity values
        6. Save preprocessed data (optional)
        
        Returns:
            Tuple of (preprocessed_data, processing_info)
        """
        print(f"Starting preprocessing for: {filepath.name}")
        self.processing_steps = []
        start_time = datetime.utcnow()
        
        # Load data (supports both NIfTI and DICOM)
        data, img = self.load_image_data(filepath)
        original_shape = data.shape
        print(f"  Loaded data shape: {original_shape}")
        
        # Clip outliers
        lower, upper = self.config.get("clip_percentiles", (1, 99))
        data = self.clip_intensity_outliers(data, lower, upper)
        
        # Noise reduction
        if self.config.get("noise_reduction", False):
            data = self.reduce_noise(data, method="gaussian")
        
        # Bias correction
        if self.config.get("bias_correction", False):
            data = self.bias_field_correction(data)
        
        # Normalization
        method = self.config.get("normalize_method", "z_score")
        data = self.normalize_intensity(data, method=method)
        
        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as NIfTI
            if img is not None:
                # Original was NIfTI, preserve metadata
                new_img = nib.Nifti1Image(data, img.affine, img.header)
            else:
                # Original was DICOM, create new NIfTI with identity affine
                affine = np.eye(4)
                new_img = nib.Nifti1Image(data, affine)
            
            nib.save(new_img, str(output_path))
            print(f"  Saved preprocessed data to: {output_path}")
        
        # Processing info
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        info = {
            "original_shape": original_shape,
            "preprocessed_shape": data.shape,
            "processing_steps": self.processing_steps,
            "processing_time_seconds": processing_time,
            "normalization_method": method,
            "config_used": self.config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        print(f"  Preprocessing completed in {processing_time:.2f}s")
        return data, info
    
    def preprocess_multimodal(
        self, 
        modality_paths: Dict[str, Path],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Tuple[np.ndarray, Dict]]:
        """
        Preprocess multiple MRI modalities - FR8.2
        
        Args:
            modality_paths: Dict of {modality_name: file_path}
            output_dir: Optional directory to save preprocessed files
            
        Returns:
            Dict of {modality_name: (preprocessed_data, info)}
        """
        results = {}
        
        for modality, filepath in modality_paths.items():
            print(f"\nProcessing modality: {modality}")
            
            output_path = None
            if output_dir:
                output_path = output_dir / f"{modality}_preprocessed.nii.gz"
            
            preprocessed_data, info = self.preprocess_mri(filepath, output_path)
            results[modality] = (preprocessed_data, info)
        
        return results


if __name__ == "__main__":
    print("=" * 60)
    print("MRI Upload & Preprocessing Module - Test Suite")
    print("=" * 60)
    
    # Test file validation
    print("\n[1] Testing File Validation...")
    validator = MRIFileValidator()
    
    test_files = [
        "scan.nii.gz",
        "scan.dcm",
        "scan.txt",
        "image.nii"
    ]
    
    for filename in test_files:
        is_valid, msg = validator.validate_file_extension(filename)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {filename}: {msg}")
    
    # Test file size validation
    print("\n[2] Testing File Size Validation...")
    test_sizes = [
        100 * 1024 * 1024,  # 100 MB
        6 * 1024 * 1024 * 1024,  # 6 GB (too large)
    ]
    
    for size in test_sizes:
        is_valid, msg = validator.validate_file_size(size)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {msg}")
    
    print("\n✅ Module 2 tests completed!")
    print(f"   Imaging libraries available: {IMAGING_LIBS_AVAILABLE}")
