# Seg-Mind: AI-Powered Brain Tumor Analysis System

## 🧠 Project Overview

Seg-Mind is an intelligent automated system for brain tumor detection, segmentation, classification, and progression prediction from MRI scans. This project implements a comprehensive AI-powered medical imaging platform designed to assist radiologists and oncologists in brain tumor diagnosis.

### Key Features
- ✅ **Module 1**: User Management System with Firebase authentication
- ✅ **Module 2**: MRI Upload & Preprocessing with DICOM validation
- ✅ **Module 3**: 3D U-Net Tumor Segmentation (MONAI framework)
- ✅ **Module 4**: ResNet-based Tumor Classification
- ✅ **Module 5**: LSTM-based Tumor Growth Prediction
- ✅ **Module 6**: Explainable AI with Grad-CAM and SHAP
- ✅ **Module 7**: 2D/3D Visualization (slice viewer, multi-view, montage, 3D projection)
- ✅ **AI Assistant**: RAG-powered help, auto-report generation, similar cases search

---

## 📁 Project Structure

```
fyp/
├── backend/
│   └── app/
│       ├── models/          # Pydantic models
│       │   └── user.py      # User, Doctor, Patient models
│       ├── routers/         # API route handlers
│       ├── services/        # Business logic
│       │   ├── auth_service.py
│       │   └── mri_preprocessing.py
│       ├── utils/           # Utility functions
│       └── main.py          # FastAPI application
│
├── ml_models/
│   ├── segmentation/
│   │   └── unet3d.py        # 3D U-Net implementation
│   └── classification/
│       └── resnet_classifier.py  # ResNet classifier
│
├── data/
│   ├── raw/                 # Raw MRI data
│   ├── processed/           # Preprocessed data
│   └── uploads/             # User uploads
│
├── notebooks/               # Jupyter notebooks for experiments
├── tests/                   # Unit tests
├── logs/                    # Application logs
├── config.py                # Configuration file
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.13+ (currently using 3.13.1)
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- Windows/Linux/macOS

### 1. Clone and Navigate to Project
```powershell
cd C:\Users\SCM\Documents\fyp
```

### 2. Virtual Environment (Already Created)
```powershell
# Activate existing virtual environment
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
# Install all required packages
pip install -r requirements.txt
```

**Key Dependencies:**
- **Deep Learning**: PyTorch, MONAI, torchvision
- **Medical Imaging**: nibabel, SimpleITK, pydicom
- **Backend**: FastAPI, uvicorn
- **Data Processing**: numpy, pandas, scikit-learn
- **Visualization**: matplotlib, plotly
- **Explainable AI**: SHAP, grad-cam
- **Cloud**: Firebase Admin SDK, Google Cloud Storage

### 4. Environment Configuration
```powershell
# Copy environment template
copy .env.example .env

# Edit .env file with your credentials:
# - Firebase credentials
# - MongoDB connection string
# - Synapse auth token (already configured)
# - API keys
```

### 5. Dataset Setup
The BraTS 2023 dataset is already downloaded at:
```
D:\BraTs 2023\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
```

**Dataset Statistics:**
- Training samples: 1,251 cases
- Modalities: T1n, T1c, T2w, T2-FLAIR
- Segmentation labels: NCR/NET, ED, ET
- Size: ~26.3 GB

---

## 🎯 Module Implementations

### Module 1: User Management System
**Status**: ✅ Implemented

**Features:**
- User registration with email verification (FR4.1-FR4.8)
- Secure login with JWT tokens (FR3.1-FR3.8)
- Password reset functionality (FR5.1-FR5.8)
- Doctor and Patient profile management (FR1.1-FR1.7, FR2.1-FR2.8)
- Role-based access control (RBAC)
- Activity logging for audit trails
- Multi-doctor collaboration (FE-4, FE-5)

**Files:**
- `backend/app/models/user.py`: User data models
- `backend/app/services/auth_service.py`: Authentication logic

**Test:**
```powershell
python backend/app/services/auth_service.py
```

---

### Module 2: MRI Upload & Preprocessing
**Status**: ✅ Implemented

**Features:**
- DICOM and NIfTI file upload (FR8.1)
- File validation (format, size) (FR8.5)
- DICOM metadata extraction (FR8.5)
- Intensity normalization (Z-score, Min-Max) (FR8.6)
- Noise reduction (Gaussian, Median filters) (FR8.7)
- N4 bias field correction
- Multi-modal preprocessing pipeline (FR8.2)

**Files:**
- `backend/app/services/mri_preprocessing.py`: Preprocessing pipeline

**Test:**
```powershell
python backend/app/services/mri_preprocessing.py
```

---

### Module 3: Tumor Segmentation (3D U-Net)
**Status**: ✅ Implemented

**Features:**
- 3D U-Net architecture using MONAI (FR9.1)
- Multi-class segmentation (NCR/NET, ED, ET) (FR9.2)
- Confidence score calculation (FR9.3)
- Segmentation mask generation (NIfTI, DICOM) (FR9.4)
- Quality metrics (Dice score, Hausdorff distance) (FR9.6)
- BraTS dataset loader
- Training pipeline with early stopping
- Model checkpointing

**Model Architecture:**
- Input: 4 channels (T1n, T1c, T2w, T2f)
- Output: 4 classes (Background + 3 tumor regions)
- Channels: (16, 32, 64, 128, 256)
- Loss: Dice + Cross-Entropy
- Optimizer: Adam (lr=1e-4)

**Files:**
- `ml_models/segmentation/unet3d.py`: 3D U-Net implementation

**Test:**
```powershell
python ml_models/segmentation/unet3d.py
```

---

### Module 4: Tumor Classification (ResNet)
**Status**: ✅ Implemented

**Features:**
- ResNet50-based classifier (FR13.1)
- Tumor type classification (GBM, LGG, Meningioma) (FR13.2)
- WHO grade estimation (FR13.3)
- Malignancy level assessment (FR13.4)
- Confidence score display (FR13.5)
- Imaging characteristic extraction (FR13.6)
- Browser-based inference (ONNX.js) (FR13.7)
- Classification history tracking (FR13.8)

**Model Architecture:**
- Backbone: ResNet50 (pretrained on ImageNet)
- Input: 4-channel MRI slices
- Output: 4 tumor classes
- Classification head: FC layers with dropout
- Loss: Cross-Entropy
- Optimizer: Adam (lr=1e-4)

**Files:**
- `ml_models/classification/resnet_classifier.py`: ResNet implementation

**Test:**
```powershell
python ml_models/classification/resnet_classifier.py
```

---

### AI Assistant Module
**Status**: ✅ Implemented

**Features:**
- **RAG-Powered Documentation Help**: Semantic search over project docs using sentence-transformers + FAISS (no LLM API costs)
- **Automated Clinical Reports**: Templated report generation with Jinja2, includes AI predictions, metrics, and clinical disclaimers
- **PDF Export**: Professional PDF reports via reportlab with proper medical formatting
- **Similar Cases Search**: Find similar patient cases by tumor type and characteristics
- **Conversational Interface**: Chat-based UI for queries and report drafting

**Why This Matters:**
- **Practical AI Integration**: Goes beyond ML inference to provide decision support tools
- **Cost-Effective**: Uses open-source embeddings (no external API dependencies)
- **Recruiter-Friendly**: Demonstrates RAG, embeddings, responsible AI, API design, and full-stack skills
- **Production-Ready**: Auth-gated, error handling, extensible architecture

**Endpoints:**
- `POST /api/v1/assistant/chat`: Conversational help with doc search
- `POST /api/v1/assistant/report`: Generate text report from case data
- `POST /api/v1/assistant/report/pdf`: Generate downloadable PDF report
- `GET /api/v1/assistant/cases/{id}/similar`: Find similar cases

**Files:**
- `backend/app/routers/assistant.py`: Assistant API router
- `frontend/src/pages/AssistantPage.tsx`: Chat UI
- `frontend/src/services/assistant.service.ts`: Frontend service

**Demo:**
See `ASSISTANT_DEMO_SCRIPT.md` for a 60-90 second walkthrough

**Test:**
```powershell
# With API running and authenticated
$token = Get-Content test_token.txt
$headers = @{ Authorization = "Bearer $token" }
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/assistant/chat -Method POST -Headers $headers -ContentType "application/json" -Body (@{ message = "What is validation Dice?" } | ConvertTo-Json)
```

---

## 🔧 Running the Application

### Start Backend API
```powershell
# Navigate to backend directory
cd backend/app

# Run FastAPI server
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /`: API health check
- `POST /api/v1/auth/register`: User registration
- `POST /api/v1/auth/login`: User login
- `POST /api/v1/mri/upload`: Upload MRI scan
- `POST /api/v1/segmentation/segment`: Perform segmentation
- `POST /api/v1/classification/classify`: Classify tumor

**Access API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📊 Training the Models

### Train 3D U-Net Segmentation Model
```powershell
# Create training script (will be provided)
python train_segmentation.py --epochs 300 --batch-size 2 --lr 1e-4
```

### Train ResNet Classification Model
```powershell
# Create training script (will be provided)
python train_classification.py --epochs 100 --batch-size 16 --lr 1e-4
```

---

## 🧪 Testing

### Run All Tests
```powershell
pytest tests/ -v --cov=backend --cov=ml_models
```

### Test Individual Modules
```powershell
# Test authentication
pytest tests/test_auth.py -v

# Test preprocessing
pytest tests/test_preprocessing.py -v

# Test segmentation
pytest tests/test_segmentation.py -v
```

---

## 📈 Performance Metrics

### Expected Model Performance (based on BraTS benchmarks)

**Segmentation (3D U-Net):**
- Dice Score (WT): ~0.90
- Dice Score (TC): ~0.85
- Dice Score (ET): ~0.80
- Hausdorff Distance: <5mm

**Classification (ResNet):**
- Accuracy: ~92%
- Precision: ~90%
- Recall: ~89%
- F1-Score: ~90%

---

## 🔐 Security & Compliance

- **Authentication**: JWT tokens with 2-hour expiration
- **Password Security**: Bcrypt hashing
- **Data Encryption**: HTTPS (TLS 1.2+), AES-256 at rest
- **Access Control**: Role-based (Doctor, Admin)
- **Audit Logging**: All sensitive operations logged
- **HIPAA Compliance**: Patient data anonymization
- **Firebase Security**: Firestore rules, Storage access controls

---

## 📝 Configuration

All settings are centralized in `config.py`:

```python
# Key configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BRATS_DATASET_PATH = "D:/BraTs 2023/..."
API_PORT = 8000
MAX_UPLOAD_SIZE = 5GB
ACCESS_TOKEN_EXPIRE_MINUTES = 120
```

---

## 🛠️ Development Workflow

### 1. Create Feature Branch
```powershell
git checkout -b feature/new-module
```

### 2. Make Changes
```powershell
# Edit files
# Add tests
# Update documentation
```

### 3. Run Tests
```powershell
pytest tests/ -v
black backend/ ml_models/  # Format code
flake8 backend/ ml_models/  # Lint code
```

### 4. Commit and Push
```powershell
git add .
git commit -m "Add new feature"
git push origin feature/new-module
```

---

## 🚦 Current Status (90% Complete)

### ✅ Completed Modules
1. ✅ Module 1: User Management (auth, profiles, RBAC)
2. ✅ Module 2: MRI Preprocessing pipeline
3. ✅ Module 3: 3D U-Net Segmentation
4. ✅ Module 4: ResNet Classification
5. ✅ Module 5: LSTM Growth Prediction
6. ✅ Module 6: Explainable AI (Grad-CAM, SHAP)
7. ✅ Module 7: 2D/3D Visualization
8. ✅ AI Assistant (RAG, auto-reports, PDF export)
9. ✅ FastAPI backend with complete API
10. ✅ React frontend with full UI
11. ✅ PostgreSQL database integration
12. ✅ Multi-doctor collaboration features

### 🔄 In Progress
- Model training on full BraTS dataset
- Advanced visualization features (WebGL 3D)
- Production deployment preparation

### 📋 Next Steps (Final 10%)
- Complete model training
- Performance optimization
- Cloud deployment (AWS/GCP)
- Final testing and documentation
- Demo video and presentation

---

## 📚 Documentation

- **SRS Document**: Complete functional requirements
- **API Documentation**: http://localhost:8000/docs (when running)
- **Code Comments**: Inline documentation for all functions
- **Module Tests**: Test files demonstrate usage

---

## 🤝 Contributors

- **Your Name** - Lead Developer
- **Project Type**: Final Year Project (FYP)
- **Academic Year**: 2024-2025

---

## 📧 Support

For questions or issues:
- Email: your.email@university.edu
- GitHub Issues: [Create an issue]
- Documentation: See `/docs` folder

---

## 🎓 Academic Context

**Course**: Software Engineering / AI Capstone Project
**Modules Covered**:
- Artificial Intelligence / Image Processing
- Software Engineering (SDLC)
- Web Technologies
- Database Systems
- Cyber Security
- Project Management

**Objectives Achieved**:
- BO-1: 3D U-Net segmentation ✅
- BO-4: ResNet classification ✅
- BO-7: Cloud-based deployment (in progress)

---

## 📄 License

This project is for academic purposes only.

---

**Last Updated**: November 4, 2025
**Version**: 1.0.0 (30% Milestone)
