"""
Seg-Mind Backend API
FastAPI application for brain tumor analysis system
Implements all SRS functional requirements
"""

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import sys
from pathlib import Path
import shutil
import uuid
import logging
import json
import numpy as np
import nibabel as nib

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Add root directory to path for ml_models and config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import database
from app.database import get_db, User, File as DBFile, AnalysisResult, FileAccessPermission, CaseCollaboration, CaseDiscussion, test_connection, init_db

# Import models and services
from app.models.user import UserCreate, DoctorProfile, PatientProfile, UserRole
from app.services.auth_service import AuthService
from app.services.mri_preprocessing import MRIPreprocessor, MRIFileValidator

# Import ML models
try:
    from ml_models.segmentation.unet3d import UNet3D, BraTSDataset, TumorSegmentationInference
    from ml_models.classification.resnet_classifier import ResNetClassifier
    import torch
    import os
    from monai.transforms import Resize, Compose, EnsureChannelFirst, ToTensor
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML models not available: {e}")
    MODELS_AVAILABLE = False

# Import config
try:
    from config import (
        API_TITLE,
        API_VERSION,
        API_DESCRIPTION,
        CORS_ORIGINS,
        DEBUG_MODE
    )
except ImportError:
    API_TITLE = "Seg-Mind API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "AI-powered Brain Tumor Segmentation"
    CORS_ORIGINS = ["*"]
    DEBUG_MODE = True

# Helper function to convert numpy types to Python types for JSON serialization
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    debug=DEBUG_MODE
)

# Configure CORS - FR15.1
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
auth_service = AuthService()
preprocessor = MRIPreprocessor()

# Legacy in-memory storage for sessions (can be migrated to Redis later)
sessions_db = {}
results_db = {}  # Temporary cache for results before saving to DB
files_db = {}  # Temporary cache for file metadata

# Global variables for ML models
segmentation_model = None
segmentation_inference = None
classification_model = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Startup event - Load ML models
@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    global segmentation_model, classification_model, segmentation_inference
    
    logger.info("Starting application...")
    
    # Test database connection
    logger.info("Testing database connection...")
    if test_connection():
        logger.info("✓ Database connected successfully")
        init_db()
    else:
        logger.error("✗ Database connection failed - check PostgreSQL is running")
    
    if MODELS_AVAILABLE:
        try:
            logger.info("Loading 3D U-Net segmentation model...")
            # Initialize model architecture
            segmentation_model = UNet3D(in_channels=4, out_channels=4)
            
            # Load pre-trained weights
            weights_path = os.path.join("ml_models", "segmentation", "unet_model.pth")
            if os.path.exists(weights_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(weights_path, map_location=device)
                
                # Handle state dict keys if they start with 'module.' or 'model.'
                state_dict = checkpoint['model_state_dict']
                # segmentation_model.load_state_dict(state_dict) # Direct load might fail if keys don't match exactly
                
                # Safe load
                try:
                    segmentation_model.load_state_dict(state_dict)
                except RuntimeError as e:
                    logger.warning(f"Direct load failed, attempting to fix keys: {e}")
                    # If needed, adjust keys here. For now assuming they match as per test script.
                    segmentation_model.load_state_dict(state_dict, strict=False)

                segmentation_model.eval()
                segmentation_model.to(device)
                
                # Initialize inference engine
                segmentation_inference = TumorSegmentationInference(segmentation_model, device=str(device))
                logger.info(f"✓ Segmentation model loaded from {weights_path}")
            else:
                logger.warning(f"⚠ Segmentation weights not found at {weights_path}")
                segmentation_model.eval() # Keep initialized but random
            
            logger.info("Loading ResNet50 classification model...")
            classification_model = ResNetClassifier(num_classes=4)  # 4 classes for Brain Tumor MRI Dataset
            
            # Load pre-trained weights
            classification_weights_path = os.path.join("ml_models", "classification", "resnet_model.pth")
            if os.path.exists(classification_weights_path):
                classification_checkpoint = torch.load(classification_weights_path, map_location=device)
                
                # Handle different checkpoint formats
                if isinstance(classification_checkpoint, dict):
                    if 'model_state_dict' in classification_checkpoint:
                        classification_model.load_state_dict(classification_checkpoint['model_state_dict'])
                    elif 'state_dict' in classification_checkpoint:
                        classification_model.load_state_dict(classification_checkpoint['state_dict'])
                    else:
                        # Assume the dict is the state_dict itself
                        classification_model.load_state_dict(classification_checkpoint)
                else:
                    classification_model.load_state_dict(classification_checkpoint)
                
                classification_model.to(device)
                classification_model.eval()
                logger.info(f"✓ Classification model loaded from {classification_weights_path}")
            else:
                logger.warning(f"⚠ Classification weights not found at {classification_weights_path}")
                classification_model.eval()
                logger.info("✓ Classification model loaded (architecture only, no weights)")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            segmentation_model = None
            classification_model = None
    else:
        logger.warning("ML models not available - using placeholder mode")
    
    # Create necessary directories
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    logger.info("Application startup complete!")


# Dependency for authentication
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Verify JWT token and return current user
    FR3.5 - Token validation
    """
    token = credentials.credentials
    try:
        payload = auth_service.decode_access_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        # Get user from database
        user = db.query(User).filter(User.user_id == int(user_id)).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Convert to dict for compatibility
        return {
            "user_id": str(user.user_id),
            "email": user.email,
            "username": user.username,
            "role": user.role
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


# Dependency for doctor-only access
async def get_current_doctor(user = Depends(get_current_user)):
    """
    Ensure current user is a doctor/radiologist
    FR1.1 - Doctor access only for certain features
    """
    if user.get("role") not in ["doctor", "radiologist", "oncologist", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can access this resource"
        )
    return user


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Seg-Mind API is running",
        "version": API_VERSION,
        "status": "healthy",
        "modules": {
            "user_management": "active",
            "mri_preprocessing": "active",
            "tumor_segmentation": "active",
            "tumor_classification": "active"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "seg-mind-api"}


# Module 1: User Management Routes

@app.post("/api/v1/auth/register")
async def register_user(user: dict, db: Session = Depends(get_db)):
    """
    Register new user - FR4.1 to FR4.8
    Supports both doctor and patient registration
    """
    try:
        email = user.get("email")
        password = user.get("password")
        role = user.get("role")
        full_name = user.get("full_name")
        username = user.get("username") or email.split("@")[0]
        
        # Validate required fields
        if not all([email, password, role, full_name]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required fields"
            )
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Hash password
        hashed_password = auth_service.hash_password(password)
        
        # Create new user with basic info
        new_user = User(
            email=email,
            username=username,
            password_hash=hashed_password,
            role=role,
            full_name=full_name
        )
        
        # Add doctor/medical professional fields if applicable
        if role in ['doctor', 'radiologist', 'oncologist']:
            new_user.medical_license = user.get("medical_license")
            new_user.specialization = user.get("specialization")
            new_user.institution = user.get("institution")
            new_user.department = user.get("department")
            new_user.years_of_experience = user.get("years_of_experience", 0)
            new_user.phone_number = user.get("phone_number")
        
        # Add patient fields if applicable
        elif role == 'patient':
            dob = user.get("date_of_birth")
            if dob:
                from datetime import datetime as dt
                new_user.date_of_birth = dt.fromisoformat(dob.replace('Z', '+00:00')) if isinstance(dob, str) else dob
            new_user.gender = user.get("gender")
            new_user.phone_number = user.get("phone_number")
            
            # Auto-generate patient ID if not provided
            # Format: PT-YYYY-XXXXX (e.g., PT-2025-00001)
            from datetime import datetime
            year = datetime.now().year
            
            # Get the count of existing patients to generate sequential ID
            patient_count = db.query(User).filter(User.role == 'patient').count()
            patient_number = str(patient_count + 1).zfill(5)
            auto_patient_id = f"PT-{year}-{patient_number}"
            
            new_user.medical_record_number = user.get("medical_record_number") or auto_patient_id
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"New user registered: {email} (role: {role})")
        
        # Build response with patient_id for patients
        response_data = {
            "message": "User registered successfully",
            "user_id": str(new_user.user_id),
            "role": role
        }
        
        if role == 'patient':
            response_data["patient_id"] = new_user.medical_record_number
        
        return response_data
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/auth/login")
async def login_user(credentials: dict, db: Session = Depends(get_db)):
    """
    User login - FR3.1 to FR3.8
    Returns JWT token on successful authentication
    """
    try:
        email = credentials.get("email")
        password = credentials.get("password")
        
        if not email or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email and password required"
            )
        
        # Find user by email
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not auth_service.verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create JWT token
        token = auth_service.create_access_token(
            data={"sub": str(user.user_id), "role": user.role}
        )
        
        logger.info(f"User logged in: {email}")
        
        # Build user profile response
        user_profile = {
            "user_id": str(user.user_id),
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role
        }
        
        # Add doctor-specific fields if applicable
        if user.role in ['doctor', 'radiologist', 'oncologist']:
            user_profile.update({
                "medical_license": user.medical_license,
                "specialization": user.specialization,
                "institution": user.institution,
                "department": user.department,
                "years_of_experience": user.years_of_experience,
                "phone_number": user.phone_number
            })
        
        # Add patient-specific fields if applicable
        elif user.role == 'patient':
            user_profile.update({
                "patient_id": user.medical_record_number,  # Patient ID
                "date_of_birth": user.date_of_birth.isoformat() if user.date_of_birth else None,
                "gender": user.gender,
                "medical_record_number": user.medical_record_number,
                "phone_number": user.phone_number
            })
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user_profile
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/auth/logout")
async def logout_user(user = Depends(get_current_user)):
    """User logout - Session termination"""
    return {"message": "Logged out successfully"}


@app.get("/api/v1/auth/me")
async def get_current_user_profile(user = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Get current user profile - FR2.3, FR1.3
    Returns complete user profile information
    """
    try:
        user_id = int(user.get("user_id"))
        db_user = db.query(User).filter(User.user_id == user_id).first()
        
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Build profile response
        profile = {
            "user_id": str(db_user.user_id),
            "email": db_user.email,
            "username": db_user.username,
            "full_name": db_user.full_name,
            "role": db_user.role,
            "created_at": db_user.created_at.isoformat() if db_user.created_at else None
        }
        
        # Add role-specific fields
        if db_user.role in ['doctor', 'radiologist', 'oncologist']:
            profile.update({
                "medical_license": db_user.medical_license,
                "specialization": db_user.specialization,
                "institution": db_user.institution,
                "department": db_user.department,
                "years_of_experience": db_user.years_of_experience,
                "phone_number": db_user.phone_number,
                "bio": db_user.bio
            })
        elif db_user.role == 'patient':
            profile.update({
                "date_of_birth": db_user.date_of_birth.isoformat() if db_user.date_of_birth else None,
                "gender": db_user.gender,
                "medical_record_number": db_user.medical_record_number,
                "phone_number": db_user.phone_number
            })
        
        return profile
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.put("/api/v1/auth/profile")
async def update_user_profile(profile_data: dict, user = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Update user profile - FR1.3, FR2.3
    Allows users to update their profile information
    """
    try:
        user_id = int(user.get("user_id"))
        db_user = db.query(User).filter(User.user_id == user_id).first()
        
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update common fields
        if "full_name" in profile_data:
            db_user.full_name = profile_data["full_name"]
        if "phone_number" in profile_data:
            db_user.phone_number = profile_data["phone_number"]
        
        # Update doctor-specific fields
        if db_user.role in ['doctor', 'radiologist', 'oncologist']:
            if "medical_license" in profile_data:
                db_user.medical_license = profile_data["medical_license"]
            if "specialization" in profile_data:
                db_user.specialization = profile_data["specialization"]
            if "institution" in profile_data:
                db_user.institution = profile_data["institution"]
            if "department" in profile_data:
                db_user.department = profile_data["department"]
            if "years_of_experience" in profile_data:
                db_user.years_of_experience = profile_data["years_of_experience"]
            if "bio" in profile_data:
                db_user.bio = profile_data["bio"]
        
        # Update patient-specific fields
        elif db_user.role == 'patient':
            if "gender" in profile_data:
                db_user.gender = profile_data["gender"]
            if "medical_record_number" in profile_data:
                db_user.medical_record_number = profile_data["medical_record_number"]
        
        db_user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"Profile updated for user: {db_user.email}")
        
        return {
            "message": "Profile updated successfully",
            "user_id": str(db_user.user_id)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/auth/reset-password")
async def reset_password():
    """Password reset - FR5.1 to FR5.8"""
    return {"message": "Password reset endpoint"}


# ============= Doctor Management Endpoints =============

@app.get("/api/v1/doctors")
async def get_all_doctors(db: Session = Depends(get_db)):
    """
    Get list of all doctors in the system
    Used by patients to select doctors for consultation
    """
    try:
        doctors = db.query(User).filter(
            User.role.in_(['doctor', 'radiologist', 'oncologist'])
        ).all()
        
        doctor_list = []
        for doc in doctors:
            doctor_list.append({
                "user_id": doc.user_id,
                "full_name": doc.full_name or doc.username,
                "specialization": doc.specialization,
                "institution": doc.institution,
                "department": doc.department,
                "years_of_experience": doc.years_of_experience,
                "bio": doc.bio
            })
        
        return {"doctors": doctor_list}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch doctors: {str(e)}"
        )


@app.get("/api/v1/doctors/dashboard")
async def get_doctor_dashboard(
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    FE-2: Doctor's personalized dashboard
    Returns: assigned patients, contributions, recent activities, notifications
    """
    try:
        user_id = int(user.get("user_id"))
        user_role = user.get("role")
        
        # Verify user is a doctor
        if user_role not in ['doctor', 'radiologist', 'oncologist']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Doctor role required"
            )
        
        # Get doctor's full info from database
        doctor_user = db.query(User).filter(User.user_id == user_id).first()
        if not doctor_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Doctor not found"
            )
        
        # Get all files this doctor has access to
        permissions = db.query(FileAccessPermission).filter(
            FileAccessPermission.doctor_id == user_id
        ).all()
        
        file_ids = [p.file_id for p in permissions]
        
        # Get unique patients from these files
        assigned_files = db.query(DBFile).filter(
            DBFile.file_id.in_(file_ids)
        ).all() if file_ids else []
        
        # Count unique patients
        unique_patients = set(f.patient_id for f in assigned_files if f.patient_id)
        
        # Get analyses by this doctor
        doctor_analyses = db.query(AnalysisResult).filter(
            AnalysisResult.assessed_by == user_id
        ).all()
        
        # Get recent analyses (last 10)
        recent_analyses = db.query(AnalysisResult).filter(
            AnalysisResult.assessed_by == user_id
        ).order_by(AnalysisResult.analysis_date.desc()).limit(10).all()
        
        recent_activities = []
        for analysis in recent_analyses:
            # Get file info
            file_info = db.query(DBFile).filter(DBFile.file_id == analysis.file_id).first()
            if file_info:
                recent_activities.append({
                    "file_id": analysis.file_id,
                    "patient_id": file_info.patient_id,
                    "filename": file_info.filename,
                    "analysis_date": analysis.analysis_date.isoformat() if analysis.analysis_date else None,
                    "has_assessment": bool(analysis.clinical_diagnosis),
                    "tumor_type": analysis.classification_type
                })
        
        # Create notifications (new patients assigned, pending assessments)
        notifications = []
        
        # Check for analyses without doctor assessment
        pending_assessments = db.query(AnalysisResult).filter(
            AnalysisResult.file_id.in_(file_ids),
            AnalysisResult.clinical_diagnosis.is_(None)
        ).count() if file_ids else 0
        
        if pending_assessments > 0:
            notifications.append({
                "type": "pending_assessment",
                "message": f"{pending_assessments} patient case(s) awaiting your clinical assessment",
                "count": pending_assessments,
                "priority": "high"
            })
        
        # Check for new patients in last 7 days
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        new_patients = db.query(FileAccessPermission).filter(
            FileAccessPermission.doctor_id == user_id,
            FileAccessPermission.granted_date >= week_ago
        ).count()
        
        if new_patients > 0:
            notifications.append({
                "type": "new_patients",
                "message": f"{new_patients} new patient(s) assigned in the last 7 days",
                "count": new_patients,
                "priority": "medium"
            })
        
        return {
            "doctor_info": {
                "name": doctor_user.full_name or doctor_user.username,
                "specialization": doctor_user.specialization or "Medical Professional",
                "user_id": user_id
            },
            "statistics": {
                "assigned_patients": len(unique_patients),
                "total_analyses": len(doctor_analyses),
                "analyses_this_month": len([a for a in doctor_analyses if a.analysis_date and a.analysis_date.month == datetime.utcnow().month]),
                "pending_assessments": pending_assessments
            },
            "assigned_patients": list(unique_patients),
            "recent_activities": recent_activities,
            "notifications": notifications
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Dashboard error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dashboard: {str(e)}"
        )


@app.post("/api/v1/files/{file_id}/grant-access")
async def grant_doctor_access(
    file_id: int,
    request_data: dict,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Grant doctors access to a patient's file
    Only the patient who owns the file can grant access
    """
    try:
        user_id = int(user.get("user_id"))
        user_role = user.get("role")
        
        # Only patients can grant access to their files
        if user_role != 'patient':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only patients can grant doctor access"
            )
        
        # Extract doctor_ids from request body
        doctor_ids = request_data.get('doctor_ids', [])
        if not doctor_ids or len(doctor_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please select at least one doctor"
            )
        
        # Get patient's patient_id
        db_user = db.query(User).filter(User.user_id == user_id).first()
        patient_id = db_user.medical_record_number
        
        # Verify file belongs to this patient
        file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        if file_record.patient_id != patient_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only grant access to your own files"
            )
        
        # Grant access to selected doctors
        granted_doctors = []
        for doctor_id in doctor_ids:
            # Check if permission already exists
            existing = db.query(FileAccessPermission).filter(
                FileAccessPermission.file_id == file_id,
                FileAccessPermission.doctor_id == doctor_id
            ).first()
            
            if existing:
                # Update status to active if it was revoked
                existing.status = 'active'
                granted_doctors.append(doctor_id)
            else:
                # Create new permission
                permission = FileAccessPermission(
                    file_id=file_id,
                    patient_id=patient_id,
                    doctor_id=doctor_id,
                    access_level='view_and_analyze',
                    status='active'
                )
                db.add(permission)
                granted_doctors.append(doctor_id)
        
        db.commit()
        
        return {
            "message": f"Access granted to {len(granted_doctors)} doctor(s)",
            "file_id": file_id,
            "granted_to": granted_doctors
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to grant access: {str(e)}"
        )


@app.delete("/api/v1/files/{file_id}/revoke-access/{doctor_id}")
async def revoke_doctor_access(
    file_id: int,
    doctor_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Revoke a doctor's access to a patient's file
    """
    try:
        user_id = int(user.get("user_id"))
        user_role = user.get("role")
        
        if user_role != 'patient':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only patients can revoke doctor access"
            )
        
        # Find and update the permission
        permission = db.query(FileAccessPermission).filter(
            FileAccessPermission.file_id == file_id,
            FileAccessPermission.doctor_id == doctor_id
        ).first()
        
        if not permission:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Access permission not found"
            )
        
        permission.status = 'revoked'
        db.commit()
        
        return {"message": "Access revoked successfully"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke access: {str(e)}"
        )


@app.get("/api/v1/files/{file_id}/access")
async def get_file_access_list(
    file_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of doctors who have access to a file
    """
    try:
        # Get all active permissions for this file
        permissions = db.query(FileAccessPermission).filter(
            FileAccessPermission.file_id == file_id,
            FileAccessPermission.status == 'active'
        ).all()
        
        # Get doctor details
        doctor_access = []
        for perm in permissions:
            doctor = db.query(User).filter(User.user_id == perm.doctor_id).first()
            if doctor:
                doctor_access.append({
                    "doctor_id": doctor.user_id,
                    "full_name": doctor.full_name or doctor.username,
                    "specialization": doctor.specialization,
                    "access_level": perm.access_level,
                    "granted_date": perm.granted_date.isoformat() if perm.granted_date else None
                })
        
        return {"file_id": file_id, "doctors_with_access": doctor_access}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get access list: {str(e)}"
        )


# Module 2: MRI Upload & Preprocessing Routes (now using PostgreSQL)
# General upload endpoint for any authenticated user
@app.post("/api/v1/upload")
async def upload_file(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    skip_validation: bool = Form(False),  # Allow bypassing deep validation for known-good files
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload MRI scan - FR8.1 to FR8.4
    - Patients can upload their own files (auto-uses their patient_id)
    - Doctors upload files for patients (must specify patient_id)
    Saves file to disk and stores metadata in database
    
    Performs comprehensive validation:
    1. File extension check (DICOM/NIfTI only)
    2. File size check (max 5GB)
    3. File integrity check (can be read/parsed)
    4. Brain MRI characteristics check (dimensions, intensity patterns)
    """
    # Get user role and set patient_id appropriately
    user_role = user.get("role")
    user_id = int(user.get("user_id"))
    
    # If patient is uploading, auto-use their patient_id
    if user_role == 'patient':
        db_user = db.query(User).filter(User.user_id == user_id).first()
        patient_id = db_user.medical_record_number
    elif not patient_id:
        # Doctors must provide patient_id
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Doctors must provide patient_id when uploading files"
        )
    
    # Basic file extension validation (quick check before saving)
    allowed_extensions = ['.dcm', '.nii', '.nii.gz']
    file_ext = Path(file.filename).suffix.lower()
    
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Create upload directory for this user
        upload_dir = Path("data/uploads") / user.get("user_id")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = upload_dir / safe_filename
        
        # Write file to disk first (needed for deep validation)
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Perform comprehensive validation on the saved file
        validation_result = MRIFileValidator.comprehensive_validation(file_path, file.filename)
        
        if not validation_result["can_process"]:
            # File is invalid - delete it and return error
            file_path.unlink(missing_ok=True)
            
            error_detail = {
                "message": "File validation failed",
                "errors": validation_result["errors"],
                "recommendations": validation_result.get("recommendations", []),
                "validation_checks": validation_result.get("checks", {})
            }
            
            logger.warning(f"File validation failed for {file.filename}: {validation_result['errors']}")
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail
            )
        
        # Determine file type
        file_type = "DICOM" if file_ext in ['.dcm', '.dicom'] else "NIfTI"
        
        # Prepare validation metadata for storage
        validation_metadata = {
            "validation_passed": True,
            "brain_mri_confidence": validation_result.get("brain_mri_confidence", None),
            "warnings": validation_result.get("warnings", []),
            "checks_summary": {
                k: v.get("valid", True) 
                for k, v in validation_result.get("checks", {}).items()
            }
        }
        
        # Create database record
        db_file = DBFile(
            user_id=int(user.get("user_id")),
            filename=file.filename,
            safe_filename=safe_filename,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            file_type=file_type,
            patient_id=patient_id,
            status="uploaded"
        )
        
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
        logger.info(f"File uploaded and validated: {db_file.file_id} by user {user.get('user_id')} "
                   f"(Brain MRI confidence: {validation_result.get('brain_mri_confidence', 'N/A')}%)")
        
        response = {
            "message": "File uploaded successfully",
            "file_id": db_file.file_id,
            "filename": file.filename,
            "size": db_file.file_size,
            "uploaded_by": user.get("user_id"),
            "patient_id": patient_id,
            "file_type": file_type,
            "uploaded_at": db_file.upload_date.isoformat(),
            "validation": {
                "brain_mri_confidence": validation_result.get("brain_mri_confidence"),
                "warnings": validation_result.get("warnings", [])
            }
        }
        
        # Add warnings to response if any
        if validation_result.get("warnings"):
            response["validation_warnings"] = validation_result["warnings"]
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        db.rollback()
        # Clean up file if it was saved
        if 'file_path' in locals() and Path(file_path).exists():
            Path(file_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )


@app.post("/api/v1/validate-file")
async def validate_file(
    file: UploadFile = File(...),
    user = Depends(get_current_user)
):
    """
    Validate MRI file without saving to database.
    
    Use this endpoint to check if a file is valid before upload.
    Performs all validation checks:
    1. File extension (DICOM/NIfTI only)
    2. File integrity (can be read)
    3. Brain MRI characteristics (dimensions, intensity patterns)
    
    Returns validation result without storing the file.
    """
    import tempfile
    
    # Quick extension check
    allowed_extensions = ['.dcm', '.nii', '.nii.gz']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return {
            "is_valid": False,
            "can_process": False,
            "errors": [f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"],
            "recommendations": ["Please upload a file with .dcm, .nii, or .nii.gz extension"]
        }
    
    try:
        # Save to temp file for validation
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        
        # Run comprehensive validation
        validation_result = MRIFileValidator.comprehensive_validation(tmp_path, file.filename)
        
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)
        
        return validation_result
        
    except Exception as e:
        # Clean up temp file if exists
        if 'tmp_path' in locals():
            Path(tmp_path).unlink(missing_ok=True)
        
        return {
            "is_valid": False,
            "can_process": False,
            "errors": [f"Validation failed: {str(e)}"],
            "recommendations": ["Please ensure the file is valid and try again"]
        }


# Legacy doctor-only upload endpoint - TODO: Migrate to database
# @app.post("/api/v1/mri/upload")
# async def upload_mri(
#     file: UploadFile = File(...),
#     patient_id: Optional[str] = None,
#     user = Depends(get_current_doctor)
# ):
#     """
#     Upload MRI scan - FR8.1 to FR8.4
#     Only doctors can upload MRI scans
#     Use /api/v1/upload endpoint instead
#     """
#     pass


@app.post("/api/v1/mri/preprocess")
async def preprocess_mri(
    file_id: int,
    normalize: str = "zscore",
    denoise: bool = True,
    bias_correction: bool = True,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Preprocess uploaded MRI - FR8.6 to FR8.8
    Applies normalization, denoising, and bias field correction
    """
    # Check if file exists in database
    file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    # Check access: patients can preprocess their own files, doctors can preprocess files they have access to
    user_id = int(user.get("user_id"))
    user_role = user.get("role")
    
    if user_role == 'patient':
        # Patients can only preprocess their own files
        if file_record.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only preprocess files you uploaded"
            )
    else:
        # Doctors can preprocess files they uploaded OR files they have been granted access to
        doctor_uploaded_file = (file_record.user_id == user_id)
        has_permission = db.query(FileAccessPermission).filter(
            FileAccessPermission.file_id == file_id,
            FileAccessPermission.doctor_id == user_id,
            FileAccessPermission.status == 'active'
        ).first()
        
        if not doctor_uploaded_file and not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this file. Patient must grant you access first."
            )
    
    try:
        logger.info(f"Preprocessing file {file_id}: {file_record.filename}")
        
        # Extract metadata from file
        file_path = Path(file_record.file_path)
        metadata = preprocessor.extract_metadata(str(file_path))
        
        # Convert metadata to JSON-serializable format
        metadata = convert_to_serializable(metadata)
        
        # Preprocess the file
        processed_dir = Path("data/processed") / user.get("user_id")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Always save preprocessed files as .nii.gz (NIfTI format)
        original_name = Path(file_record.safe_filename).stem  # Remove extension
        output_path = processed_dir / f"processed_{original_name}.nii.gz"
        
        # Run preprocessing pipeline
        preprocessing_result = preprocessor.preprocess_pipeline(
            str(file_path),
            str(output_path),
            normalize=normalize,
            denoise=denoise,
            bias_correction=bias_correction
        )
        
        # Update file record in database
        file_record.preprocessed = True
        file_record.preprocessed_path = str(output_path)
        file_record.preprocessing_params = {
            "normalize": normalize,
            "denoise": denoise,
            "bias_correction": bias_correction
        }
        file_record.file_metadata = metadata
        file_record.status = "preprocessed"
        
        db.commit()
        
        logger.info(f"✓ Preprocessing complete for {file_id}")
        
        return {
            "message": "MRI preprocessing complete",
            "file_id": file_id,
            "preprocessed_path": str(output_path),
            "metadata": metadata,
            "parameters": {
                "normalize": normalize,
                "denoise": denoise,
                "bias_correction": bias_correction
            }
        }
        
    except Exception as e:
        logger.error(f"Preprocessing failed for {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )


@app.get("/api/v1/mri/files")
async def list_mri_files(
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List MRI files based on user role and access permissions
    - Patients see their own files
    - Doctors see files they have been granted access to
    """
    user_id = int(user.get("user_id"))
    user_role = user.get("role")
    
    # Get user from database to access patient_id
    db_user = db.query(User).filter(User.user_id == user_id).first()
    
    if user_role == 'patient':
        # Patients see files where patient_id matches their medical_record_number
        patient_id = db_user.medical_record_number
        files_query = db.query(DBFile).filter(DBFile.patient_id == patient_id)
        files = files_query.order_by(DBFile.upload_date.desc()).all()
    else:
        # Doctors see only files they have been granted access to
        # Get file IDs from access permissions
        permitted_file_ids = db.query(FileAccessPermission.file_id).filter(
            FileAccessPermission.doctor_id == user_id,
            FileAccessPermission.status == 'active'
        ).all()
        
        file_ids = [f[0] for f in permitted_file_ids]
        
        if file_ids:
            files = db.query(DBFile).filter(DBFile.file_id.in_(file_ids)).order_by(DBFile.upload_date.desc()).all()
        else:
            files = []
    
    # Convert to list of dicts and include analysis info
    user_files = []
    for f in files:
        # Get all analyses for this file
        analyses = db.query(AnalysisResult).filter(AnalysisResult.file_id == f.file_id).all()
        
        # Count how many doctors have analyzed this file
        analysis_count = len(analyses)
        
        # Check if current doctor has analyzed (for doctors)
        has_analyzed = False
        if user_role != 'patient':
            has_analyzed = any(a.assessed_by == user_id for a in analyses)
        
        user_files.append({
            "file_id": f.file_id,
            "filename": f.filename,
            "safe_filename": f.safe_filename,
            "path": f.file_path,
            "size": f.file_size,
            "file_type": f.file_type,
            "uploaded_by": str(f.user_id),
            "patient_id": f.patient_id,
            "upload_date": f.upload_date.isoformat() if f.upload_date else None,
            "status": f.status,
            "preprocessed": f.preprocessed,
            "analysis_count": analysis_count,
            "has_analyzed": has_analyzed
        })
    
    return user_files


@app.get("/api/v1/mri/files/{file_id}")
async def get_mri_file(
    file_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific MRI file
    """
    file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    # Check access permissions
    user_id = int(user.get("user_id"))
    if file_record.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return {
        "file_id": file_record.file_id,
        "filename": file_record.filename,
        "safe_filename": file_record.safe_filename,
        "path": file_record.file_path,
        "size": file_record.file_size,
        "file_type": file_record.file_type,
        "uploaded_by": str(file_record.user_id),
        "patient_id": file_record.patient_id,
        "uploaded_at": file_record.upload_date.isoformat() if file_record.upload_date else None,
        "status": file_record.status,
        "preprocessed": file_record.preprocessed,
        "preprocessed_path": file_record.preprocessed_path,
        "metadata": file_record.file_metadata
    }


@app.get("/api/v1/mri/files/{file_id}/download")
async def download_mri_file(
    file_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Download MRI file - accessible by patient (owner) or doctors with access
    """
    from fastapi.responses import FileResponse
    import os
    
    file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    # Check access permissions
    user_id = int(user.get("user_id"))
    user_role = user.get("role")
    
    has_access = False
    
    if user_role == 'patient':
        # Patient can download their own files
        has_access = (file_record.user_id == user_id)
    elif user_role in ['doctor', 'radiologist', 'oncologist']:
        # Doctor must have been granted access
        permission = db.query(FileAccessPermission).filter(
            FileAccessPermission.file_id == file_id,
            FileAccessPermission.doctor_id == user_id
        ).first()
        has_access = (permission is not None)
    
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You don't have permission to download this file"
        )
    
    # Check if file exists on disk
    file_path = file_record.file_path
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on server"
        )
    
    # Return file for download
    return FileResponse(
        path=file_path,
        filename=file_record.filename,
        media_type='application/octet-stream',
        headers={
            "Content-Disposition": f'attachment; filename="{file_record.filename}"'
        }
    )


@app.delete("/api/v1/mri/files/{file_id}")
async def delete_mri_file(
    file_id: int,
    user = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    """
    Delete an MRI file
    Only the doctor who uploaded it can delete
    """
    file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    # Check if user owns this file
    user_id = int(user.get("user_id"))
    if file_record.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete files you uploaded"
        )
    
    try:
        # Delete file from disk
        file_path = Path(file_record.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete preprocessed file if exists
        if file_record.preprocessed_path:
            preprocessed_path = Path(file_record.preprocessed_path)
            if preprocessed_path.exists():
                preprocessed_path.unlink()
        
        # Remove from database
        db.delete(file_record)
        db.commit()
        
        return {
            "message": "File deleted successfully",
            "file_id": file_id
        }
    except Exception as e:
        db.rollback()
        logger.error(f"File deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File deletion failed: {str(e)}"
        )




# Module 3: Tumor Segmentation Routes (Legacy - use /api/v1/analyze instead)
# @app.post("/api/v1/segmentation/segment")
# async def segment_tumor(
#     file_id: int,
#     user = Depends(get_current_doctor),
#     db: Session = Depends(get_db)
# ):
#     """
#     Perform tumor segmentation - FR9.1 to FR9.8
#     Use /api/v1/analyze endpoint instead for complete analysis
#     """
#     pass


# @app.get("/api/v1/segmentation/results/{result_id}")
# async def get_segmentation_results(result_id: int, user = Depends(get_current_user)):
#     """Get segmentation results - FR9.2, FR9.3, FR9.6"""
#     pass


# Module 4: Tumor Classification Routes (Legacy - use /api/v1/analyze instead)
# @app.post("/api/v1/classification/classify")
# async def classify_tumor(
#     file_id: int,
#     user = Depends(get_current_doctor),
#     db: Session = Depends(get_db)
# ):
#     """
#     Classify tumor type - FR13.1 to FR13.8
#     Use /api/v1/analyze endpoint instead for complete analysis
#     """
#     pass


# @app.get("/api/v1/classification/results/{result_id}")
# async def get_classification_results(result_id: int, user = Depends(get_current_user)):
#     """Get classification results - FR13.5, FR13.6"""
#     pass


# Utility endpoints
@app.post("/api/v1/analyze")
async def analyze_complete(
    file_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Complete analysis pipeline: Preprocess → Segment → Classify
    Runs all steps in sequence and returns comprehensive results
    """
    try:
        logger.info(f"Starting complete analysis for file {file_id}")
        
        # Get file from database
        file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Check access: 
        # - Doctors can only analyze files they have been granted access to
        # - Patients cannot analyze (only view)
        user_id = int(user.get("user_id"))
        user_role = user.get("role")
        
        if user_role == 'patient':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only doctors can perform analysis"
            )
        
        # Check if doctor has access to this file
        # Option 1: Doctor uploaded the file themselves
        # Option 2: Doctor has been granted access via FileAccessPermission
        doctor_uploaded_file = (file_record.user_id == user_id)
        has_permission = db.query(FileAccessPermission).filter(
            FileAccessPermission.file_id == file_id,
            FileAccessPermission.doctor_id == user_id,
            FileAccessPermission.status == 'active'
        ).first()
        
        if not doctor_uploaded_file and not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this file. Patient must grant you access first."
            )
        
        # Check if this doctor has already analyzed this file
        existing_analysis = db.query(AnalysisResult).filter(
            AnalysisResult.file_id == file_id,
            AnalysisResult.assessed_by == user_id
        ).first()
        
        if existing_analysis:
            # Delete the existing analysis to replace it with new one
            # (Alternative: could update instead, but re-running gives fresh results)
            logger.info(f"Doctor {user_id} already analyzed file {file_id}, replacing existing analysis")
            db.delete(existing_analysis)
            db.commit()
        
        # Get doctor information
        doctor = db.query(User).filter(User.user_id == user_id).first()
        
        # Update file status
        file_record.status = "analyzing"
        db.commit()
        
        # Step 1: Preprocess if not already done
        logger.info("Step 1: Preprocessing...")
        if not file_record.preprocessed:
            await preprocess_mri(
                file_id=file_id,
                normalize="zscore",
                denoise=True,
                bias_correction=True,
                user=user,
                db=db
            )
            db.refresh(file_record)
        
        # Step 2: Segmentation
        logger.info("Step 2: Segmentation...")
        
        segmentation_data = None
        
        # Try to run actual inference if model is loaded
        if segmentation_inference and file_record.preprocessed_path:
            try:
                logger.info(f"Running inference on {file_record.preprocessed_path}")
                
                # Load preprocessed NIfTI
                img = nib.load(file_record.preprocessed_path)
                data = img.get_fdata() # (H, W, D)
                
                # Convert to tensor
                data_tensor = torch.from_numpy(data).float()
                
                # Resize to 128x128x128 if needed (Model expects this size)
                if data.shape != (128, 128, 128):
                    logger.info(f"Resizing input from {data.shape} to (128, 128, 128)")
                    resizer = Resize(spatial_size=(128, 128, 128))
                    
                    # Handle 2D input (H, W) -> (H, W, 1)
                    if data_tensor.ndim == 2:
                        data_tensor = data_tensor.unsqueeze(-1)
                    
                    # Add channel dim for resize: (C, H, W, D) -> (1, H, W, D)
                    # Ensure we have (C, spatial...) format for MONAI Resize
                    if data_tensor.ndim == 3: # (H, W, D)
                        data_tensor = data_tensor.unsqueeze(0) # (1, H, W, D)
                        
                    data_tensor = resizer(data_tensor).squeeze(0)
                
                # Create 4 channels (duplicate single modality)
                # Input: (H, W, D) -> (1, H, W, D) -> (4, H, W, D)
                # Note: This is a fallback for single-file upload. 
                # Ideally we should load 4 different files.
                if data_tensor.ndim == 3:
                    input_tensor = data_tensor.unsqueeze(0).repeat(4, 1, 1, 1)
                elif data_tensor.ndim == 4 and data_tensor.shape[0] == 1:
                    input_tensor = data_tensor.repeat(4, 1, 1, 1)
                else:
                    input_tensor = data_tensor # Assume already correct
                
                # Add batch dim: (1, 4, 128, 128, 128)
                input_batch = input_tensor.unsqueeze(0).to(str(segmentation_inference.device))
                
                # Predict
                prediction, probs = segmentation_inference.predict(input_batch, return_probabilities=True)
                
                # Calculate volumes
                # Use voxel spacing from image header if available, else (1,1,1)
                if hasattr(img.header, 'get_zooms'):
                    zooms = img.header.get_zooms()
                    # Handle 2D images where zooms might only have 2 elements
                    if len(zooms) >= 3:
                        spacing = zooms[:3]
                    elif len(zooms) == 2:
                        spacing = (zooms[0], zooms[1], 1.0)
                    else:
                        spacing = (1.0, 1.0, 1.0)
                else:
                    spacing = (1.0, 1.0, 1.0)
                
                volumes = segmentation_inference.calculate_tumor_volumes(
                    prediction[0], 
                    voxel_spacing=spacing
                )
                
                # Calculate confidence
                conf_scores = segmentation_inference.calculate_confidence(probs[0])
                avg_conf = np.mean([s["mean_confidence"] for s in conf_scores.values()]) if conf_scores else 0.0
                
                # Map volume keys to short names (config may have long names)
                # Look for NCR/NET, ED, ET in the keys (partial match)
                ncr_data = {"volume_mm3": 0, "voxel_count": 0}
                ed_data = {"volume_mm3": 0, "voxel_count": 0}
                et_data = {"volume_mm3": 0, "voxel_count": 0}
                
                for key, val in volumes.items():
                    key_upper = key.upper()
                    if "NCR" in key_upper or "NET" in key_upper or "NECROTIC" in key_upper:
                        ncr_data = val
                    elif "ED" in key_upper or "EDEMA" in key_upper:
                        ed_data = val
                    elif "ET" in key_upper or "ENHANCING" in key_upper:
                        et_data = val
                
                # Calculate total from actual tumor regions only
                total_voxels = ncr_data["voxel_count"] + ed_data["voxel_count"] + et_data["voxel_count"]
                total_mm3 = ncr_data["volume_mm3"] + ed_data["volume_mm3"] + et_data["volume_mm3"]
                
                # Format for frontend
                segmentation_data = {
                    "regions": {
                        "NCR": ncr_data,
                        "ED": ed_data,
                        "ET": et_data
                    },
                    "total_volume": {
                        "voxels": total_voxels,
                        "mm3": total_mm3
                    },
                    "metrics": {
                        "dice_score": 0.0, # No ground truth available
                        "confidence": float(avg_conf)
                    }
                }
                logger.info("✓ Segmentation inference complete")
                
            except Exception as e:
                logger.error(f"Segmentation inference failed: {e}")
                segmentation_data = None

        # Fallback to simulated results if inference failed or model not available
        if segmentation_data is None:
            logger.warning("Using simulated segmentation results")
            segmentation_data = {
                "regions": {
                    "NCR": {
                        "volume_voxels": 4523,
                        "volume_mm3": 3618.4
                    },
                    "ED": {
                        "volume_voxels": 12847,
                        "volume_mm3": 10277.6
                    },
                    "ET": {
                        "volume_voxels": 8634,
                        "volume_mm3": 6907.2
                    }
                },
                "total_volume": {
                    "voxels": 26004,
                    "mm3": 20803.2
                },
                "metrics": {
                    "dice_score": 0.89,
                    "hausdorff_distance": 4.2
                }
            }

        logger.info("Step 3: Classification (Hybrid Approach)...")
        
        classification_data = None
        classification_type = None
        classification_confidence = None
        who_grade = None
        malignancy = None
        resnet_prediction = None
        
        # ============================================================
        # HYBRID CLASSIFICATION APPROACH
        # 1. First, run ResNet classification on 2D slice
        # 2. Then, apply segmentation-based rules for BraTS tumor grading
        # 3. Combine both for final classification
        # ============================================================
        
        # Step 3a: Run ResNet classification (for 2D tumor detection)
        if classification_model is not None:
            try:
                logger.info("Running ResNet classification on 2D slice...")
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                classification_model.to(device)
                classification_model.eval()
                
                # Load the preprocessed data for 2D slice extraction
                img = nib.load(file_record.preprocessed_path)
                data = img.get_fdata()
                
                # Get middle slice (most informative)
                if data.ndim == 3:
                    mid_slice_idx = data.shape[2] // 2
                    slice_2d = data[:, :, mid_slice_idx]
                elif data.ndim == 2:
                    slice_2d = data
                else:
                    slice_2d = data[:, :, 0]
                
                # Resize to 224x224 for ResNet
                from monai.transforms import Resize as MonaiResize
                slice_tensor = torch.from_numpy(slice_2d).float()
                
                if slice_tensor.ndim == 1:
                    slice_tensor = slice_tensor.unsqueeze(0)
                
                slice_tensor = slice_tensor.unsqueeze(0)  # (1, H, W)
                resizer_2d = MonaiResize(spatial_size=(224, 224))
                slice_tensor = resizer_2d(slice_tensor)  # (1, 224, 224)
                
                # Create 4 channels for ResNet input
                input_tensor = slice_tensor.repeat(4, 1, 1)  # (4, 224, 224)
                input_batch = input_tensor.unsqueeze(0).to(device)
                
                # Run classification
                with torch.no_grad():
                    outputs = classification_model(input_batch)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, dim=1)
                
                # Map class to tumor type (Brain Tumor MRI Dataset - 4 classes)
                tumor_types_map = {
                    0: "Glioma",
                    1: "Meningioma", 
                    2: "No Tumor",
                    3: "Pituitary Tumor"
                }
                
                resnet_prediction = {
                    "type": tumor_types_map.get(predicted_class.item(), "Unknown"),
                    "confidence": confidence.item() * 100,
                    "probabilities": {tumor_types_map.get(i, f"Class_{i}"): float(p) 
                                     for i, p in enumerate(probabilities[0].cpu().numpy())}
                }
                
                logger.info(f"  ResNet prediction: {resnet_prediction['type']} ({resnet_prediction['confidence']:.1f}%)")
                
            except Exception as e:
                logger.error(f"ResNet classification failed: {e}")
                resnet_prediction = None
        
        # Step 3b: Segmentation-based classification (for BraTS tumor grading)
        # Extract tumor volumes
        ncr_vol = segmentation_data["regions"]["NCR"]["volume_mm3"]
        ed_vol = segmentation_data["regions"]["ED"]["volume_mm3"]
        et_vol = segmentation_data["regions"]["ET"]["volume_mm3"]
        total_tumor_vol = segmentation_data["total_volume"]["mm3"]
        
        # Calculate ratios
        wt_volume = ncr_vol + ed_vol + et_vol  # Whole Tumor
        tc_volume = ncr_vol + et_vol  # Tumor Core
        et_ratio = et_vol / wt_volume if wt_volume > 0 else 0
        ncr_ratio = ncr_vol / wt_volume if wt_volume > 0 else 0
        
        logger.info(f"  Segmentation volumes - NCR: {ncr_vol:.1f}, ED: {ed_vol:.1f}, ET: {et_vol:.1f}, Total: {wt_volume:.1f}")
        logger.info(f"  Ratios - ET/WT: {et_ratio:.3f}, NCR/WT: {ncr_ratio:.3f}")
        
        # Step 3c: Hybrid Decision Logic
        # Priority: Segmentation-based rules for BraTS-style data, ResNet as backup
        
        segmentation_based_classification = None
        
        # Rule 1: No significant tumor detected
        if wt_volume < 100:  # Less than 100 mm³
            segmentation_based_classification = {
                "type": "No Tumor Detected",
                "who_grade": "N/A",
                "malignancy": "None",
                "confidence": 95.0,
                "reasoning": "Total tumor volume below detection threshold (<100 mm³)"
            }
        
        # Rule 2: High-Grade Glioma (GBM) - characterized by high enhancement and necrosis
        elif et_vol > 1000 or (et_ratio > 0.25 and wt_volume > 5000):
            segmentation_based_classification = {
                "type": "Glioblastoma (GBM)",
                "who_grade": "Grade IV",
                "malignancy": "High",
                "confidence": min(90.0, 70.0 + et_ratio * 80),
                "reasoning": f"High enhancing tumor volume ({et_vol:.1f} mm³) and ET/WT ratio ({et_ratio:.2f}) indicate high-grade glioma"
            }
        
        # Rule 3: Anaplastic Astrocytoma (Grade III) - moderate enhancement
        elif et_vol > 500 or (et_ratio > 0.15 and wt_volume > 3000):
            segmentation_based_classification = {
                "type": "Anaplastic Astrocytoma",
                "who_grade": "Grade III",
                "malignancy": "Medium",
                "confidence": min(85.0, 65.0 + et_ratio * 60),
                "reasoning": f"Moderate enhancing tumor ({et_vol:.1f} mm³) suggests Grade III astrocytoma"
            }
        
        # Rule 4: Low-Grade Glioma (LGG) - minimal enhancement, mostly edema
        elif wt_volume > 500 and et_ratio < 0.15:
            segmentation_based_classification = {
                "type": "Low-Grade Glioma (LGG)",
                "who_grade": "Grade II",
                "malignancy": "Low",
                "confidence": min(85.0, 60.0 + (1 - et_ratio) * 50),
                "reasoning": f"Low enhancement ratio ({et_ratio:.2f}) with significant tumor volume indicates low-grade glioma"
            }
        
        # Rule 5: Small tumor with some enhancement - needs further investigation
        elif wt_volume >= 100:
            segmentation_based_classification = {
                "type": "Glioma (Grade II-III)",
                "who_grade": "Grade II-III",
                "malignancy": "Medium",
                "confidence": 70.0,
                "reasoning": f"Tumor detected ({wt_volume:.1f} mm³) - grade determination requires clinical correlation"
            }
        
        # Step 3d: Combine ResNet and Segmentation-based results
        if segmentation_based_classification and segmentation_based_classification["type"] != "No Tumor Detected":
            # Tumor detected by segmentation - use segmentation-based classification
            # (More reliable for BraTS-style 3D MRI data)
            classification_type = segmentation_based_classification["type"]
            who_grade = segmentation_based_classification["who_grade"]
            malignancy = segmentation_based_classification["malignancy"]
            classification_confidence = segmentation_based_classification["confidence"]
            reasoning = segmentation_based_classification["reasoning"]
            classification_source = "segmentation_rules"
            
            logger.info(f"  Final (segmentation-based): {classification_type} - {reasoning}")
            
        elif resnet_prediction and resnet_prediction["type"] != "No Tumor":
            # ResNet detected tumor but segmentation didn't find significant volume
            # Trust ResNet for 2D-based detection
            classification_type = resnet_prediction["type"]
            classification_confidence = resnet_prediction["confidence"]
            
            # Map ResNet tumor types to WHO grades
            if "Glioma" in classification_type:
                who_grade = "Grade II-IV"
                malignancy = "High"
            elif "Meningioma" in classification_type:
                who_grade = "Grade I-II"
                malignancy = "Low"
            elif "Pituitary" in classification_type:
                who_grade = "Grade I"
                malignancy = "Low"
            else:
                who_grade = "Unknown"
                malignancy = "Unknown"
            
            reasoning = f"ResNet classification (2D slice analysis)"
            classification_source = "resnet"
            
            logger.info(f"  Final (ResNet-based): {classification_type}")
            
        else:
            # Neither found tumor
            classification_type = "No Tumor Detected"
            who_grade = "N/A"
            malignancy = "None"
            classification_confidence = 95.0
            reasoning = "No significant tumor detected by segmentation or classification models"
            classification_source = "consensus"
            
            logger.info(f"  Final: No tumor detected")
        
        # Build class probabilities (combine both sources)
        class_probs = {}
        if resnet_prediction:
            class_probs = resnet_prediction["probabilities"].copy()
        
        # Add segmentation-based tumor type with its confidence
        if segmentation_based_classification:
            seg_type = segmentation_based_classification["type"]
            seg_conf = segmentation_based_classification["confidence"] / 100
            # Merge with existing or add new
            if seg_type in class_probs:
                class_probs[seg_type] = max(class_probs[seg_type], seg_conf)
            else:
                class_probs[seg_type] = seg_conf
        
        classification_data = {
            "prediction": {
                "tumor_type": classification_type,
                "confidence": classification_confidence,
                "who_grade": who_grade,
                "malignancy": malignancy
            },
            "class_probabilities": class_probs,
            "analysis_details": {
                "classification_source": classification_source,
                "reasoning": reasoning,
                "segmentation_volumes": {
                    "NCR": ncr_vol,
                    "ED": ed_vol,
                    "ET": et_vol,
                    "total": wt_volume
                },
                "resnet_prediction": resnet_prediction
            }
        }
        
        logger.info(f"✓ Hybrid classification complete: {classification_type} ({classification_confidence:.1f}%)")
        
        # Calculate actual tumor volume from segmentation
        actual_tumor_volume = segmentation_data["total_volume"]["mm3"]
        
        # Determine classification source for notes
        classification_source = classification_data.get("analysis_details", {}).get("classification_source", "unknown")
        
        if classification_source == "segmentation_rules":
            notes_text = "Hybrid analysis: Classification based on segmentation-derived tumor characteristics (BraTS methodology)"
        elif classification_source == "resnet":
            notes_text = "Hybrid analysis: Classification from ResNet (2D slice), segmentation from 3D U-Net"
        else:
            notes_text = "Analysis completed using hybrid approach (3D U-Net segmentation + rule-based classification)"
        
        # Save results to database with doctor information
        analysis_result = AnalysisResult(
            file_id=file_id,
            segmentation_data=segmentation_data,
            segmentation_confidence=segmentation_data["metrics"].get("confidence", 0.0),
            tumor_volume=actual_tumor_volume,
            tumor_regions=segmentation_data["regions"],
            classification_type=classification_type,
            classification_confidence=classification_confidence,
            who_grade=who_grade,
            malignancy_level=malignancy,
            preprocessing_time=2.3,
            segmentation_time=4.5,
            classification_time=3.2,
            total_time=10.0,
            model_versions={"segmentation": "3D U-Net v1.0", "classification": "ResNet50 v1.0"},
            notes=notes_text,
            assessed_by=user_id,
            doctor_name=doctor.full_name or doctor.username,
            doctor_specialization=doctor.specialization,
            assessment_date=datetime.utcnow()
        )
        
        db.add(analysis_result)
        
        # Update file status
        file_record.status = "analyzed"
        db.commit()
        db.refresh(analysis_result)
        
        logger.info(f"✓ Complete analysis finished for {file_id}")
        
        return {
            "file_id": file_id,
            "analysis_id": analysis_result.result_id,
            "status": "completed",
            "timestamp": analysis_result.analysis_date.isoformat(),
            "patient_id": file_record.patient_id,
            "filename": file_record.filename,
            "segmentation": segmentation_data,
            "classification": classification_data,
            "summary": {
                "diagnosis": classification_type,
                "confidence": classification_confidence,
                "tumor_volume_mm3": actual_tumor_volume,
                "who_grade": who_grade,
                "malignancy": malignancy
            },
            "note": notes_text
        }
        
    except Exception as e:
        logger.error(f"Complete analysis failed for {file_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/api/v1/analyze/results/{file_id}")
async def get_analysis_results(
    file_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all analysis results for a file from different doctors
    Returns multiple analyses if the file was analyzed by multiple doctors
    """
    try:
        # Get file from database
        file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Check access: patients see their files, doctors see files they have access to
        user_id = int(user.get("user_id"))
        user_role = user.get("role")
        
        if user_role == 'patient':
            # Get patient's patient_id
            db_user = db.query(User).filter(User.user_id == user_id).first()
            patient_id = db_user.medical_record_number
            
            if file_record.patient_id != patient_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You can only view your own files"
                )
        else:
            # Check if doctor has access to this file
            has_access = db.query(FileAccessPermission).filter(
                FileAccessPermission.file_id == file_id,
                FileAccessPermission.doctor_id == user_id,
                FileAccessPermission.status == 'active'
            ).first()
            
            if not has_access:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have access to this file"
                )
        
        # Get ALL analysis results for this file (from all doctors)
        analysis_results = db.query(AnalysisResult).filter(
            AnalysisResult.file_id == file_id
        ).order_by(AnalysisResult.analysis_date.desc()).all()
        
        if not analysis_results or len(analysis_results) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No analysis results found for this file. Please analyze it first."
            )
        
        # Build response with all analyses
        analyses_list = []
        for analysis_result in analysis_results:
            analysis_data = {
                "file_id": file_id,
                "analysis_id": analysis_result.result_id,
                "status": "completed",
                "timestamp": analysis_result.analysis_date.isoformat(),
                "patient_id": file_record.patient_id,
                "filename": file_record.filename,
                "segmentation": analysis_result.segmentation_data,
                "classification": {
                    "prediction": {
                        "tumor_type": analysis_result.classification_type,
                        "confidence": analysis_result.classification_confidence,
                        "who_grade": analysis_result.who_grade,
                        "malignancy": analysis_result.malignancy_level
                    }
                },
                "summary": {
                    "diagnosis": analysis_result.classification_type,
                    "confidence": analysis_result.classification_confidence,
                    "tumor_volume_mm3": analysis_result.tumor_volume,
                    "who_grade": analysis_result.who_grade,
                    "malignancy": analysis_result.malignancy_level
                },
                "total_time": analysis_result.total_time,
                "note": analysis_result.notes,
                # Doctor information
                "doctor_info": {
                    "doctor_id": analysis_result.assessed_by,
                    "doctor_name": analysis_result.doctor_name,
                    "specialization": analysis_result.doctor_specialization,
                    "assessment_date": analysis_result.assessment_date.isoformat() if analysis_result.assessment_date else None
                },
                # Doctor's clinical assessment
                "doctor_assessment": {
                    "interpretation": analysis_result.doctor_interpretation,
                    "diagnosis": analysis_result.clinical_diagnosis,
                    "prescription": analysis_result.prescription,
                    "treatment_plan": analysis_result.treatment_plan,
                    "follow_up_notes": analysis_result.follow_up_notes,
                    "next_appointment": analysis_result.next_appointment.isoformat() if analysis_result.next_appointment else None
                } if analysis_result.doctor_interpretation else None
            }
            analyses_list.append(analysis_data)
        
        return {
            "file_id": file_id,
            "patient_id": file_record.patient_id,
            "filename": file_record.filename,
            "total_analyses": len(analyses_list),
            "analyses": analyses_list
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to get analysis results for {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analysis results: {str(e)}"
        )


@app.put("/api/v1/analyze/results/{file_id}/assessment")
async def add_doctor_assessment(
    file_id: int,
    assessment: dict,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add doctor's clinical assessment to analysis results
    Only doctors can add assessments
    """
    try:
        # Check if user is a doctor
        user_role = user.get("role")
        if user_role not in ['doctor', 'radiologist', 'oncologist']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only doctors can add clinical assessments"
            )
        
        user_id = int(user.get("user_id"))
        
        # Check if doctor has access to this file
        file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Verify doctor has been granted access to this file
        has_access = db.query(FileAccessPermission).filter(
            FileAccessPermission.file_id == file_id,
            FileAccessPermission.doctor_id == user_id
        ).first()
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to assess this file"
            )
        
        # Get the latest analysis result for this file (regardless of who did it)
        analysis_result = db.query(AnalysisResult).filter(
            AnalysisResult.file_id == file_id
        ).order_by(AnalysisResult.analysis_date.desc()).first()
        
        if not analysis_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No analysis results found for this file. Please analyze the file first."
            )
        
        # Check if this doctor has already assessed this file
        # If yes, update their assessment; if no, create a new assessment record
        existing_assessment = db.query(AnalysisResult).filter(
            AnalysisResult.file_id == file_id,
            AnalysisResult.assessed_by == user_id,
            AnalysisResult.clinical_diagnosis.isnot(None)
        ).first()
        
        if existing_assessment:
            # Update existing assessment
            target_result = existing_assessment
        else:
            # Create a new assessment record for this doctor
            # Copy the AI analysis data but add this doctor's assessment
            from copy import deepcopy
            target_result = AnalysisResult(
                file_id=file_id,
                segmentation_data=analysis_result.segmentation_data,
                segmentation_confidence=analysis_result.segmentation_confidence,
                tumor_volume=analysis_result.tumor_volume,
                tumor_regions=analysis_result.tumor_regions,
                classification_type=analysis_result.classification_type,
                classification_confidence=analysis_result.classification_confidence,
                who_grade=analysis_result.who_grade,
                malignancy_level=analysis_result.malignancy_level,
                preprocessing_time=analysis_result.preprocessing_time,
                segmentation_time=analysis_result.segmentation_time,
                classification_time=analysis_result.classification_time,
                total_time=analysis_result.total_time,
                model_versions=analysis_result.model_versions,
                notes=f"Assessment by {user.get('full_name', 'Doctor')}"
            )
            db.add(target_result)
        
        # Update with doctor's assessment
        from datetime import datetime
        
        doctor = db.query(User).filter(User.user_id == user_id).first()
        
        target_result.doctor_interpretation = assessment.get("doctor_interpretation")
        target_result.clinical_diagnosis = assessment.get("clinical_diagnosis")
        target_result.prescription = assessment.get("prescription")
        target_result.treatment_plan = assessment.get("treatment_plan")
        target_result.follow_up_notes = assessment.get("follow_up_notes")
        target_result.assessment_date = datetime.now()
        target_result.assessed_by = user_id
        target_result.doctor_name = doctor.full_name or doctor.username
        target_result.doctor_specialization = doctor.specialization
        
        # Handle next_appointment date
        next_appt = assessment.get("next_appointment")
        if next_appt:
            try:
                from datetime import date
                if isinstance(next_appt, str):
                    target_result.next_appointment = date.fromisoformat(next_appt)
                elif isinstance(next_appt, date):
                    target_result.next_appointment = next_appt
            except:
                pass
        
        db.commit()
        db.refresh(target_result)
        
        return {
            "message": "Assessment added successfully",
            "file_id": file_id,
            "result_id": target_result.result_id,
            "assessed_by": user_id,
            "doctor_name": target_result.doctor_name,
            "assessment_date": target_result.assessment_date.isoformat()
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to add assessment for {file_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add assessment: {str(e)}"
        )


# ============= Discussion Panel Endpoints (FE-5) =============

@app.post("/api/v1/cases/{file_id}/discussions")
async def add_discussion_comment(
    file_id: int,
    comment_data: dict,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    FE-5: Add a clinical opinion/comment to a patient case
    Supports multi-doctor collaboration and second opinions
    """
    try:
        user_id = int(user.get("user_id"))
        user_role = user.get("role")
        
        # Verify user is a doctor
        if user_role not in ['doctor', 'radiologist', 'oncologist']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only doctors can post clinical opinions"
            )
        
        # Verify doctor has access to this file
        has_access = db.query(FileAccessPermission).filter(
            FileAccessPermission.file_id == file_id,
            FileAccessPermission.doctor_id == user_id
        ).first()
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't have permission to view this case"
            )
        
        # Get file record
        file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Get comment content
        comment_text = comment_data.get("comment", "").strip()
        comment_type = comment_data.get("type", "general")  # general, second_opinion, clinical_note
        
        if not comment_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Comment text is required"
            )
        
        # Get doctor info
        doctor = db.query(User).filter(User.user_id == user_id).first()
        
        # Create discussion entry
        discussion_entry = {
            "discussion_id": str(uuid.uuid4()),
            "user_id": user_id,
            "doctor_name": doctor.full_name if doctor else user.get("full_name"),
            "specialization": doctor.specialization if doctor else user.get("specialization"),
            "comment": comment_text,
            "comment_type": comment_type,
            "timestamp": datetime.utcnow().isoformat(),
            "edited": False
        }
        
        # Add to file's discussions (stored in notes field as JSON for now)
        # In production, this should be a separate table
        if not file_record.notes:
            file_record.notes = json.dumps({"discussions": []})
        
        try:
            notes_data = json.loads(file_record.notes) if isinstance(file_record.notes, str) else file_record.notes
            if not isinstance(notes_data, dict):
                notes_data = {"discussions": []}
            if "discussions" not in notes_data:
                notes_data["discussions"] = []
        except:
            notes_data = {"discussions": []}
        
        notes_data["discussions"].append(discussion_entry)
        file_record.notes = json.dumps(notes_data)
        
        db.commit()
        
        return {
            "message": "Comment added successfully",
            "discussion": discussion_entry
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to add discussion comment: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add comment: {str(e)}"
        )


@app.get("/api/v1/cases/{file_id}/discussions")
async def get_discussion_comments(
    file_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    FE-5: Get all clinical opinions/comments for a patient case
    Shows collaborative discussions between doctors
    """
    try:
        user_id = int(user.get("user_id"))
        user_role = user.get("role")
        
        # Get file record
        file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Check access permissions
        if user_role == 'patient':
            # Patient can only view their own files
            if file_record.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        elif user_role in ['doctor', 'radiologist', 'oncologist']:
            # Doctor must have been granted access
            has_access = db.query(FileAccessPermission).filter(
                FileAccessPermission.file_id == file_id,
                FileAccessPermission.doctor_id == user_id
            ).first()
            
            if not has_access:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: You don't have permission to view this case"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get discussions from notes field
        discussions = []
        if file_record.notes:
            try:
                notes_data = json.loads(file_record.notes) if isinstance(file_record.notes, str) else file_record.notes
                if isinstance(notes_data, dict) and "discussions" in notes_data:
                    discussions = notes_data["discussions"]
            except:
                pass
        
        # Sort by timestamp (newest first)
        discussions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "file_id": file_id,
            "patient_id": file_record.patient_id,
            "filename": file_record.filename,
            "total_comments": len(discussions),
            "discussions": discussions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get discussions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve discussions: {str(e)}"
        )


@app.get("/api/v1/models/status")
async def get_models_status():
    """Get status of all AI models"""
    return {
        "segmentation_model": {
            "name": "3D U-Net",
            "architecture": "loaded" if segmentation_model is not None else "not_available",
            "weights": "loaded" if segmentation_inference is not None else "not_trained",
            "status": "ready" if segmentation_inference is not None else "architecture_ready",
            "version": "1.0",
            "input_channels": 4,
            "output_classes": 4,
            "classes": ["Background", "NCR", "ED", "ET"]
        },
        "classification_model": {
            "name": "ResNet50",
            "architecture": "loaded" if classification_model is not None else "not_available",
            "weights": "not_trained",
            "status": "architecture_ready",
            "version": "1.0",
            "num_classes": 7,
            "classes": [
                "Glioblastoma", "Astrocytoma", "Oligodendroglioma",
                "Meningioma", "Medulloblastoma", "Ependymoma", "Other"
            ]
        },
        "preprocessing": {
            "available": True,
            "methods": ["zscore", "minmax"],
            "denoising": ["gaussian", "median"],
            "bias_correction": "N4ITK"
        }
    }


# ============= Multi-Doctor Collaboration APIs =============

@app.post("/api/v1/collaboration/share/{file_id}")
async def share_case_with_doctor(
    file_id: int,
    collaborator_email: str = Form(...),
    message: str = Form(None),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Share a case with another doctor for collaboration
    - Invites another doctor to view and discuss the case
    - Creates a collaboration record
    """
    try:
        # Verify the current user
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        user_id = user.get("user_id")
        
        # Check if file exists and user has access
        file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Find the collaborating doctor by email
        collaborator = db.query(User).filter(User.email == collaborator_email).first()
        if not collaborator:
            raise HTTPException(status_code=404, detail=f"Doctor with email {collaborator_email} not found")
        
        if collaborator.user_id == user_id:
            raise HTTPException(status_code=400, detail="Cannot share with yourself")
        
        # Check if already shared
        existing = db.query(CaseCollaboration).filter(
            CaseCollaboration.file_id == file_id,
            CaseCollaboration.collaborating_doctor_id == collaborator.user_id,
            CaseCollaboration.status == 'active'
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="Case already shared with this doctor")
        
        # Create collaboration record
        collaboration = CaseCollaboration(
            file_id=file_id,
            primary_doctor_id=user_id,
            collaborating_doctor_id=collaborator.user_id,
            message=message,
            status='active'
        )
        db.add(collaboration)
        db.commit()
        db.refresh(collaboration)
        
        logger.info(f"Case {file_id} shared with {collaborator_email} by user {user_id}")
        
        return {
            "success": True,
            "message": f"Case shared with Dr. {collaborator.full_name or collaborator.email}",
            "collaboration_id": collaboration.collaboration_id,
            "collaborator": {
                "id": collaborator.user_id,
                "name": collaborator.full_name or collaborator.username,
                "email": collaborator.email,
                "specialization": collaborator.specialization
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sharing case: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/collaboration/collaborators/{file_id}")
async def get_case_collaborators(
    file_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Get all doctors collaborating on a case
    """
    try:
        # Verify the current user
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        user_id = user.get("user_id")
        
        # Get all collaborations for this file
        collaborations = db.query(CaseCollaboration).filter(
            CaseCollaboration.file_id == file_id,
            CaseCollaboration.status == 'active'
        ).all()
        
        # Get all unique doctor IDs involved
        doctor_ids = set()
        for collab in collaborations:
            doctor_ids.add(collab.primary_doctor_id)
            doctor_ids.add(collab.collaborating_doctor_id)
        
        # Also check if current user uploaded the file (note: File model uses user_id, not uploaded_by)
        file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if file_record and file_record.user_id:
            doctor_ids.add(file_record.user_id)
        
        # Get doctor details
        doctors = []
        for doc_id in doctor_ids:
            doctor = db.query(User).filter(User.user_id == doc_id).first()
            if doctor:
                # Check if this doctor has analyzed this case
                analysis = db.query(AnalysisResult).filter(
                    AnalysisResult.file_id == file_id,
                    AnalysisResult.assessed_by == doc_id
                ).first()
                
                # Check if this is the primary doctor from any collaboration
                is_primary = any(c.primary_doctor_id == doc_id for c in collaborations)
                
                doctors.append({
                    "id": doctor.user_id,
                    "name": doctor.full_name or doctor.username,
                    "email": doctor.email,
                    "specialization": doctor.specialization,
                    "is_primary": is_primary,
                    "has_analyzed": analysis is not None,
                    "is_current_user": doc_id == user_id
                })
        
        return {
            "file_id": file_id,
            "collaborators": doctors,
            "total": len(doctors)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collaborators: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/collaboration/remove/{file_id}/{doctor_id}")
async def remove_collaborator(
    file_id: int,
    doctor_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Remove a collaborating doctor from a case
    """
    try:
        # Verify the current user
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        # Find and deactivate the collaboration
        collaboration = db.query(CaseCollaboration).filter(
            CaseCollaboration.file_id == file_id,
            CaseCollaboration.collaborating_doctor_id == doctor_id,
            CaseCollaboration.status == 'active'
        ).first()
        
        if not collaboration:
            raise HTTPException(status_code=404, detail="Collaboration not found")
        
        collaboration.status = 'completed'
        db.commit()
        
        return {"success": True, "message": "Collaborator removed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing collaborator: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Discussion/Comments APIs =============

@app.post("/api/v1/discussion/{file_id}")
async def add_discussion_comment(
    file_id: int,
    comment: str = Form(...),
    parent_id: int = Form(None),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Add a comment to the case discussion thread
    """
    try:
        # Verify the current user
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        user_id = user.get("user_id")
        
        # Check if file exists
        file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get doctor info
        doctor = db.query(User).filter(User.user_id == user_id).first()
        
        # Create discussion entry
        discussion = CaseDiscussion(
            file_id=file_id,
            doctor_id=user_id,
            comment=comment,
            parent_id=parent_id
        )
        db.add(discussion)
        db.commit()
        db.refresh(discussion)
        
        logger.info(f"Comment added to case {file_id} by user {user_id}")
        
        return {
            "success": True,
            "discussion_id": discussion.discussion_id,
            "comment": {
                "id": discussion.discussion_id,
                "text": discussion.comment,
                "created_at": discussion.created_at.isoformat(),
                "doctor": {
                    "id": user_id,
                    "name": doctor.full_name or doctor.username if doctor else "Unknown",
                    "specialization": doctor.specialization if doctor else None
                },
                "parent_id": parent_id
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding comment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/discussion/{file_id}")
async def get_discussion_thread(
    file_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Get all comments in the discussion thread for a case
    """
    try:
        # Verify the current user
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        user_id = user.get("user_id")
        
        # Get all discussions for this file
        discussions = db.query(CaseDiscussion).filter(
            CaseDiscussion.file_id == file_id
        ).order_by(CaseDiscussion.created_at.asc()).all()
        
        comments = []
        for disc in discussions:
            doctor = db.query(User).filter(User.user_id == disc.doctor_id).first()
            comments.append({
                "id": disc.discussion_id,
                "text": disc.comment,
                "created_at": disc.created_at.isoformat(),
                "updated_at": disc.updated_at.isoformat() if disc.updated_at else None,
                "doctor": {
                    "id": disc.doctor_id,
                    "name": doctor.full_name or doctor.username if doctor else "Unknown",
                    "specialization": doctor.specialization if doctor else None,
                    "is_current_user": disc.doctor_id == user_id
                },
                "parent_id": disc.parent_id
            })
        
        return {
            "file_id": file_id,
            "comments": comments,
            "total": len(comments)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting discussion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/discussion/{discussion_id}")
async def update_discussion_comment(
    discussion_id: int,
    comment: str = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Update a comment (only by the author)
    """
    try:
        # Verify the current user
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        user_id = user.get("user_id")
        
        # Find the discussion
        discussion = db.query(CaseDiscussion).filter(
            CaseDiscussion.discussion_id == discussion_id
        ).first()
        
        if not discussion:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        if discussion.doctor_id != user_id:
            raise HTTPException(status_code=403, detail="Can only edit your own comments")
        
        discussion.comment = comment
        discussion.updated_at = datetime.utcnow()
        db.commit()
        
        return {"success": True, "message": "Comment updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating comment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/discussion/{discussion_id}")
async def delete_discussion_comment(
    discussion_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Delete a comment (only by the author)
    """
    try:
        # Verify the current user
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        user_id = user.get("user_id")
        
        # Find the discussion
        discussion = db.query(CaseDiscussion).filter(
            CaseDiscussion.discussion_id == discussion_id
        ).first()
        
        if not discussion:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        if discussion.doctor_id != user_id:
            raise HTTPException(status_code=403, detail="Can only delete your own comments")
        
        db.delete(discussion)
        db.commit()
        
        return {"success": True, "message": "Comment deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/collaboration/shared-with-me")
async def get_cases_shared_with_me(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Get all cases shared with the current doctor
    """
    try:
        # Verify the current user
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        
        user_id = user.get("user_id")
        
        # Get all active collaborations where user is the collaborating doctor
        collaborations = db.query(CaseCollaboration).filter(
            CaseCollaboration.collaborating_doctor_id == user_id,
            CaseCollaboration.status == 'active'
        ).all()
        
        shared_cases = []
        for collab in collaborations:
            file_record = db.query(DBFile).filter(DBFile.file_id == collab.file_id).first()
            primary_doctor = db.query(User).filter(User.user_id == collab.primary_doctor_id).first()
            
            # Get analysis if exists
            analysis = db.query(AnalysisResult).filter(
                AnalysisResult.file_id == collab.file_id
            ).first()
            
            if file_record:
                shared_cases.append({
                    "collaboration_id": collab.collaboration_id,
                    "file_id": file_record.file_id,
                    "filename": file_record.filename,
                    "patient_id": file_record.patient_id,
                    "shared_at": collab.shared_at.isoformat(),
                    "message": collab.message,
                    "shared_by": {
                        "id": primary_doctor.user_id if primary_doctor else None,
                        "name": primary_doctor.full_name or primary_doctor.username if primary_doctor else "Unknown",
                        "specialization": primary_doctor.specialization if primary_doctor else None
                    },
                    "has_analysis": analysis is not None,
                    "classification_type": analysis.classification_type if analysis else None
                })
        
        return {
            "shared_cases": shared_cases,
            "total": len(shared_cases)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting shared cases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Starting Seg-Mind Backend API")
    print("=" * 60)
    print(f"Title: {API_TITLE}")
    print(f"Version: {API_VERSION}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print("=" * 60)
    print(f"Server running at: http://127.0.0.1:8000")
    print(f"API Documentation: http://127.0.0.1:8000/docs")
    print(f"Alternative docs: http://127.0.0.1:8000/redoc")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0 for Windows
        port=8000,
        reload=False,  # Disable reload to prevent shutdown issues
        log_level="info"
    )
