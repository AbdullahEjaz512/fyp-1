"""
Database configuration and models for Seg-Mind
PostgreSQL integration using SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, Text, TIMESTAMP, ForeignKey, CheckConstraint, BigInteger, text, Date
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import os

# Database connection URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@localhost:5432/segmind_db"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query debugging
    pool_pre_ping=True,  # Verify connections before using
    pool_size=10,
    max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


# ============= Database Models =============

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Profile fields (for all users)
    full_name = Column(String(255))
    phone_number = Column(String(50))
    
    # Doctor/Medical professional fields
    medical_license = Column(String(100))
    specialization = Column(String(100))
    institution = Column(String(255))
    department = Column(String(100))
    years_of_experience = Column(Integer, default=0)
    profile_picture_url = Column(Text)
    bio = Column(Text)
    
    # Patient fields
    date_of_birth = Column(TIMESTAMP)
    gender = Column(String(20))
    medical_record_number = Column(String(100))
    
    # Relationships
    files = relationship("File", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint(
            role.in_(['admin', 'doctor', 'radiologist', 'oncologist', 'patient']),
            name='check_user_role'
        ),
    )


class File(Base):
    """File model for uploaded MRI scans"""
    __tablename__ = "files"
    
    file_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    safe_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(BigInteger)
    upload_date = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    preprocessed = Column(Boolean, default=False)
    preprocessed_path = Column(Text)
    preprocessing_params = Column(JSONB)
    file_metadata = Column(JSONB)
    patient_id = Column(String(255))
    status = Column(String(50), default='uploaded', index=True)
    notes = Column(Text)  # For storing discussion/comments as JSON
    
    # Relationships
    user = relationship("User", back_populates="files")
    analysis_result = relationship("AnalysisResult", back_populates="file", uselist=False, cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint(
            file_type.in_(['DICOM', 'NIfTI']),
            name='check_file_type'
        ),
        CheckConstraint(
            status.in_(['uploaded', 'preprocessing', 'preprocessed', 'analyzing', 'analyzed', 'failed']),
            name='check_file_status'
        ),
    )


class AnalysisResult(Base):
    """Analysis result model for AI predictions"""
    __tablename__ = "analysis_results"
    
    result_id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey('files.file_id', ondelete='CASCADE'), nullable=False, index=True)
    analysis_date = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    
    # Segmentation results
    segmentation_data = Column(JSONB)
    segmentation_confidence = Column(Float)
    tumor_volume = Column(Float)
    tumor_regions = Column(JSONB)
    
    # Classification results
    classification_type = Column(String(100))
    classification_confidence = Column(Float)
    who_grade = Column(String(20))
    malignancy_level = Column(String(50))
    
    # Performance metrics
    preprocessing_time = Column(Float)
    segmentation_time = Column(Float)
    classification_time = Column(Float)
    total_time = Column(Float)
    
    # Additional metadata
    model_versions = Column(JSONB)
    notes = Column(Text)
    
    # Doctor assessment fields
    doctor_interpretation = Column(Text)
    clinical_diagnosis = Column(String(500))
    prescription = Column(Text)
    treatment_plan = Column(Text)
    follow_up_notes = Column(Text)
    next_appointment = Column(Date)
    assessment_date = Column(TIMESTAMP)
    assessed_by = Column(Integer, ForeignKey('users.user_id'))
    doctor_name = Column(String(255))
    doctor_specialization = Column(String(100))
    
    # Relationships
    file = relationship("File", back_populates="analysis_result")


class FileAccessPermission(Base):
    """Controls which doctors can access specific patient files"""
    __tablename__ = "file_access_permissions"
    
    permission_id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey('files.file_id', ondelete='CASCADE'), nullable=False, index=True)
    patient_id = Column(String(100), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    granted_date = Column(TIMESTAMP, default=datetime.utcnow)
    access_level = Column(String(50), default='view_and_analyze')  # 'view_only' or 'view_and_analyze'
    status = Column(String(50), default='active')  # 'active' or 'revoked'
    
    # Relationships
    file = relationship("File")
    doctor = relationship("User")


class CaseCollaboration(Base):
    """Multi-doctor collaboration on patient cases"""
    __tablename__ = "case_collaborations"
    
    collaboration_id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey('files.file_id', ondelete='CASCADE'), nullable=False, index=True)
    primary_doctor_id = Column(Integer, ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False)
    collaborating_doctor_id = Column(Integer, ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False)
    shared_at = Column(TIMESTAMP, default=datetime.utcnow)
    status = Column(String(50), default='active')  # 'active', 'completed', 'declined'
    message = Column(Text)  # Optional message when sharing
    
    # Relationships
    file = relationship("File")
    primary_doctor = relationship("User", foreign_keys=[primary_doctor_id])
    collaborating_doctor = relationship("User", foreign_keys=[collaborating_doctor_id])


class CaseDiscussion(Base):
    """Discussion thread for collaborative case review"""
    __tablename__ = "case_discussions"
    
    discussion_id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey('files.file_id', ondelete='CASCADE'), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False)
    comment = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    parent_id = Column(Integer, ForeignKey('case_discussions.discussion_id'), nullable=True)  # For replies
    
    # Relationships
    file = relationship("File")
    doctor = relationship("User")
    replies = relationship("CaseDiscussion", backref="parent", remote_side=[discussion_id])


class AuditLog(Base):
    """Audit log entries for sensitive data access and security events"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(128), nullable=True, index=True)
    role = Column(String(50), nullable=True)
    action = Column(String(255), nullable=False)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(255), nullable=True, index=True)
    method = Column(String(10), nullable=True)
    path = Column(String(255), nullable=True, index=True)
    status_code = Column(Integer, nullable=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    extra = Column(JSONB, nullable=True)


# ============= Database Helper Functions =============

def get_db():
    """
    Dependency for FastAPI to get database session
    Usage: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")


def test_connection():
    """Test database connection"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing database connection...")
    test_connection()
    print("\nInitializing database tables...")
    init_db()
