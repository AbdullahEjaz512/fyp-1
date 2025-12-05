"""
Module 1: User Management System
Handles user authentication, profiles, and role-based access control
"""

from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from enum import Enum


class UserRole(str, Enum):
    """User roles as per SRS Module 1"""
    DOCTOR = "doctor"
    RADIOLOGIST = "radiologist"
    ONCOLOGIST = "oncologist"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class UserBase(BaseModel):
    """Base user model - common fields"""
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=100)
    role: UserRole
    phone_number: Optional[str] = None
    

class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8)
    confirm_password: str
    
    def validate_passwords(self):
        """Ensure passwords match - FR4.4"""
        return self.password == self.confirm_password


class DoctorProfile(BaseModel):
    """
    Doctor Profile Model - FR1.1, FR1.3
    Professional information for doctors/radiologists
    """
    user_id: str
    full_name: str
    email: EmailStr
    phone_number: Optional[str] = None
    
    # Professional Information
    medical_license: str = Field(..., description="Medical license ID")
    specialization: str = Field(..., description="e.g., Radiology, Oncology, Neurology")
    institution: str
    department: str
    years_of_experience: int = Field(ge=0)
    
    # Profile metadata
    profile_picture_url: Optional[str] = None
    bio: Optional[str] = None
    
    # Statistics - FR1.4
    cases_analyzed: int = 0
    average_analysis_time: float = 0.0  # in minutes
    accuracy_rate: float = 0.0  # percentage
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "full_name": "Dr. Sarah Johnson",
                "email": "sarah.johnson@hospital.com",
                "phone_number": "+1-555-0123",
                "medical_license": "MD-12345-NY",
                "specialization": "Neuroradiology",
                "institution": "City General Hospital",
                "department": "Radiology",
                "years_of_experience": 8,
                "cases_analyzed": 250,
                "average_analysis_time": 15.5,
                "accuracy_rate": 94.2
            }
        }


class PatientProfile(BaseModel):
    """
    Patient Profile Model - FR2.1
    Patient information and MRI scan history
    """
    patient_id: str = Field(..., description="Unique patient identifier")
    full_name: str
    age: int = Field(ge=0, le=150)
    gender: str = Field(..., pattern="^(Male|Female|Other)$")
    date_of_birth: datetime
    
    # Contact information - HIPAA compliant
    contact_number: Optional[str] = None
    email: Optional[EmailStr] = None
    address: Optional[str] = None
    
    # Medical information
    medical_record_number: str
    referring_physician: Optional[str] = None
    
    # Scan history - FR2.2
    total_scans: int = 0
    last_scan_date: Optional[datetime] = None
    
    # Access control
    assigned_doctors: List[str] = []  # List of doctor user_ids with access
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "PT-2024-001",
                "full_name": "John Doe",
                "age": 45,
                "gender": "Male",
                "date_of_birth": "1979-03-15",
                "contact_number": "+1-555-0456",
                "medical_record_number": "MRN-456789",
                "total_scans": 3,
                "assigned_doctors": ["user_12345", "user_67890"]
            }
        }


class UserSession(BaseModel):
    """
    User Session Model - FR3.7
    Manages user authentication sessions
    """
    session_id: str
    user_id: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if session has expired - FR3.7"""
        return datetime.utcnow() > self.expires_at
    
    def is_idle_timeout(self, timeout_minutes: int = 120) -> bool:
        """
        Check for idle timeout - Table 2.23
        Default: 2 hours as per SRS
        """
        idle_threshold = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        return self.last_activity < idle_threshold


class ActivityLog(BaseModel):
    """
    Activity Log Model - FR1.3, FR16.7
    Tracks all user actions for audit purposes
    """
    log_id: str
    user_id: str
    action: str
    description: str
    resource_type: Optional[str] = None  # e.g., "patient", "scan", "report"
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "log_id": "log_abc123",
                "user_id": "user_12345",
                "action": "SCAN_UPLOADED",
                "description": "Uploaded MRI scan for patient PT-2024-001",
                "resource_type": "scan",
                "resource_id": "scan_xyz789",
                "timestamp": "2024-11-04T10:30:00Z"
            }
        }


class CaseCollaboration(BaseModel):
    """
    Case Collaboration Model - FE-4, FE-5 (Module 1)
    Multi-doctor collaboration on patient cases
    """
    case_id: str
    patient_id: str
    primary_doctor_id: str
    collaborating_doctors: List[str] = []  # List of user_ids
    
    # Discussion/Comments - FE-5
    discussions: List[dict] = []  # {user_id, comment, timestamp}
    
    # Shared access tracking - Table 2.24
    shared_at: datetime = Field(default_factory=datetime.utcnow)
    shared_by: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "case_123",
                "patient_id": "PT-2024-001",
                "primary_doctor_id": "user_12345",
                "collaborating_doctors": ["user_67890", "user_11111"],
                "discussions": [
                    {
                        "user_id": "user_67890",
                        "comment": "Observed significant tumor growth in T2-FLAIR",
                        "timestamp": "2024-11-04T14:20:00Z"
                    }
                ]
            }
        }


class PasswordReset(BaseModel):
    """Password Reset Model - FR5.1 to FR5.8"""
    email: EmailStr
    reset_token: Optional[str] = None
    token_expires: Optional[datetime] = None
    is_used: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def is_valid(self) -> bool:
        """Check if reset token is valid and not expired - FR5.3"""
        if self.is_used:
            return False
        if self.token_expires and datetime.utcnow() > self.token_expires:
            return False
        return True
