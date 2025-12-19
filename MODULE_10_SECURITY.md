# Module 10: Cloud Infrastructure and Security

## Overview

Module 10 implements comprehensive cloud security and deployment features:

- **FE-1**: Firebase Authentication with role-based access (Doctor/Admin)
- **FE-2**: Firebase Hosting + GCP infrastructure deployment
- **FE-3**: Encrypted HTTPS transmission + secure Firestore/Storage rules
- **FE-4**: Audit logging for patient data access and compliance

## Architecture

### Backend
- **Hybrid Authentication**: Firebase ID tokens + JWT fallback
- **Audit Middleware**: Logs all sensitive resource access
- **Security Router**: Admin endpoints for audit logs and Firebase role management
- **Postgres AuditLog Table**: Persistent audit trail

### Frontend
- **Firebase SDK**: Authentication UI and token management
- **HTTPS Enforcement**: All API calls over secure channels
- **Token Propagation**: Firebase tokens sent in Authorization header

### Infrastructure
- **Firebase Hosting**: Static frontend deployment with security headers
- **Cloud Run / App Engine**: Backend API deployment (GCP)
- **Cloud SQL (Postgres)**: Production database
- **Cloud Storage**: MRI file storage with access rules

## Features Implemented

### 1. Firebase Authentication (FE-1)

**Backend** ([backend/app/services/firebase_auth.py](backend/app/services/firebase_auth.py)):
- `FirebaseAuthClient`: Verifies Firebase ID tokens, syncs users to Postgres
- `set_role_claim()`: Set custom claims (doctor/admin/patient)
- Auto-sync Firebase users with local database

**Backend** ([backend/app/main.py](backend/app/main.py)):
- `get_current_user()`: Tries Firebase token first, falls back to JWT
- Transparent auth for existing endpoints

**Admin Endpoint**:
```http
POST /api/v1/security/firebase/set-role
{
  "email": "doctor@hospital.com",
  "role": "doctor"
}
```

### 2. Audit Logging (FE-4)

**Backend** ([backend/app/services/audit_service.py](backend/app/services/audit_service.py)):
- `AuditLogger`: Logs sensitive path access (files, analyses, auth)
- Records: user, role, path, IP, status, timestamp

**Backend** ([backend/app/database.py](backend/app/database.py)):
- `AuditLog` table: Persistent audit trail in Postgres

**Backend** ([backend/app/main.py](backend/app/main.py)):
- `audit_middleware`: Automatic logging on every request
- Configurable via `ENABLE_AUDIT_LOGS` environment variable

**Admin Endpoints**:
```http
GET /api/v1/security/audit-logs?skip=0&limit=100&user_id=123
GET /api/v1/security/audit-logs/stats
```

### 3. Security Rules (FE-3)

**Firebase Storage** ([storage.rules](storage.rules)):
- Doctor/Admin only read/write for scans/results
- 5MB limit on profile avatars
- Backend-only write for processed data

**Firestore** ([firestore.rules](firestore.rules)):
- Role-based access: `request.auth.token.role`
- Patient data readable only by doctors/admins
- Audit logs admin-only

### 4. Deployment Configuration (FE-2)

**Firebase Hosting** ([firebase.json](firebase.json)):
- Security headers: HSTS, X-Frame-Options, CSP
- SPA rewrites to `/index.html`
- Static asset caching

**Backend Deployment** (App Engine/Cloud Run):
- HTTPS enforced via GCP load balancer
- Cloud SQL Postgres connection
- Environment variables for secrets (`.env` → Secret Manager)

## Setup Instructions

### Prerequisites

1. **Firebase Project**:
   - Create project at [console.firebase.google.com](https://console.firebase.google.com)
   - Enable Authentication (Email/Password, Google)
   - Enable Firestore, Storage
   - Download service account JSON

2. **GCP Project**:
   - Enable Cloud Run, Cloud SQL, Secret Manager APIs
   - Create Cloud SQL Postgres instance
   - Configure IAM roles (Cloud Run Admin, SQL Client)

### Backend Configuration

1. **Environment Variables** (`.env`):
```bash
# Firebase
FIREBASE_CREDENTIALS_PATH=firebase-credentials.json
FIREBASE_STORAGE_BUCKET=seg-mind.appspot.com

# Audit Logging
ENABLE_AUDIT_LOGS=true

# Database (Cloud SQL)
DATABASE_URL=postgresql://user:pass@/segmind_db?host=/cloudsql/project:region:instance

# Security
SECRET_KEY=<your-production-secret>
```

2. **Deploy Backend to Cloud Run**:
```bash
# Build container
docker build -t gcr.io/seg-mind/backend .

# Push to Container Registry
docker push gcr.io/seg-mind/backend

# Deploy
gcloud run deploy seg-mind-api \
  --image gcr.io/seg-mind/backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL=postgresql://... \
  --set-env-vars FIREBASE_CREDENTIALS_PATH=/secrets/firebase.json \
  --add-cloudsql-instances project:region:instance
```

3. **Initialize Database**:
```bash
# SSH into Cloud SQL or use Cloud SQL Proxy
python -c "from app.database import init_db; init_db()"
```

### Frontend Configuration

1. **Firebase Config** (`frontend/src/config/firebase.ts`):
```typescript
export const firebaseConfig = {
  apiKey: "AIza...",
  authDomain: "seg-mind.firebaseapp.com",
  projectId: "seg-mind",
  storageBucket: "seg-mind.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc123"
};
```

2. **Build Frontend**:
```bash
cd frontend
npm install
npm run build  # Outputs to build/
```

3. **Deploy to Firebase Hosting**:
```bash
firebase login
firebase use seg-mind  # Your project ID
firebase deploy --only hosting
```

## Security Features

### HTTPS Enforcement
- Firebase Hosting: Automatic HTTPS
- Backend: Cloud Run enforces HTTPS
- API calls: `https://api.seg-mind.com`

### Data Encryption
- **In Transit**: TLS 1.3 (HTTPS)
- **At Rest**: Cloud SQL automatic encryption, Storage bucket encryption

### Role-Based Access Control (RBAC)
- **Frontend**: Firebase Auth custom claims (`role: 'doctor'`)
- **Backend**: `get_current_doctor()` dependency enforces roles
- **Database**: Row-level security via `FileAccessPermission` table

### Audit Trail
- **Who**: User ID + role
- **What**: HTTP method + path + resource
- **When**: UTC timestamp
- **Where**: IP address + User-Agent
- **Status**: HTTP status code

### Compliance
- **HIPAA**: Audit logs, encryption, access controls
- **GDPR**: User consent, data deletion (via admin endpoints)

## Testing

### Test Firebase Auth
```bash
# Get Firebase ID token (use Firebase Console or SDK)
TOKEN="eyJhbGciOiJSUzI1NiIsImtpZCI6..."

# Test backend endpoint
curl -H "Authorization: Bearer $TOKEN" \
  https://api.seg-mind.com/api/v1/auth/me
```

### Test Audit Logs
```bash
# Login as admin
TOKEN="admin_jwt_token"

# Get audit logs
curl -H "Authorization: Bearer $TOKEN" \
  https://api.seg-mind.com/api/v1/security/audit-logs?limit=50

# Get stats
curl -H "Authorization: Bearer $TOKEN" \
  https://api.seg-mind.com/api/v1/security/audit-logs/stats
```

### Test Security Rules
```bash
# Try to access protected Firestore data
firebase firestore:get /patients/PT-2025-00001 --token $DOCTOR_TOKEN
```

## Admin Tasks

### Set User Role
```bash
# Via API
curl -X POST https://api.seg-mind.com/api/v1/security/firebase/set-role \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email": "doctor@hospital.com", "role": "doctor"}'

# Via Firebase CLI
firebase auth:import users.json
```

### View Audit Logs
```bash
# Query Postgres directly
psql -h /cloudsql/... -U postgres -d segmind_db \
  -c "SELECT * FROM audit_logs WHERE user_id='123' ORDER BY created_at DESC LIMIT 100;"
```

### Revoke Access
```bash
# Revoke file access permission
curl -X DELETE https://api.seg-mind.com/api/v1/files/123/access/456 \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## Monitoring

### Cloud Logging
- Backend logs: Cloud Run → Cloud Logging
- Frontend: Firebase Performance Monitoring
- Audit logs: Postgres + Cloud Logging

### Alerts
- Failed login attempts: Cloud Monitoring alert
- High audit log volume: BigQuery + Data Studio dashboard
- Security incidents: Cloud Security Command Center

## Cost Estimates

### Firebase (Free Tier → Pay-as-you-go)
- **Authentication**: Free up to 50k MAU
- **Hosting**: 10GB storage, 360MB/day (free)
- **Firestore**: 1GB storage, 50k reads/day (free)
- **Storage**: 5GB storage, 1GB/day egress (free)

### GCP
- **Cloud Run**: $0.00002400/vCPU-second (scales to zero)
- **Cloud SQL**: ~$25/month (db-f1-micro)
- **Cloud Storage**: $0.020/GB/month

**Total**: ~$30-50/month for development, ~$200-500/month for production

## Documentation

- Firebase Auth: https://firebase.google.com/docs/auth
- Firestore Security: https://firebase.google.com/docs/firestore/security/get-started
- Cloud Run Deployment: https://cloud.google.com/run/docs/quickstarts
- HIPAA Compliance: https://cloud.google.com/security/compliance/hipaa

## Status

✅ **FE-1**: Firebase Authentication + role claims implemented  
✅ **FE-2**: Deployment configs (firebase.json, Cloud Run ready)  
✅ **FE-3**: Security rules (Firestore + Storage), HTTPS enforcement  
✅ **FE-4**: Audit logging (middleware + admin endpoints)  

**Next**: Frontend Firebase SDK integration (login/register UI)
