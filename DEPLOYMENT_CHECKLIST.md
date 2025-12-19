# GCP & Firebase Deployment Checklist

## Prerequisites Setup (15 min)

### 1. Create Firebase Project
```bash
# Go to https://console.firebase.google.com
# Click "Add project" → Enter "seg-mind" → Continue
# Disable Google Analytics (optional) → Create project
```

### 2. Install Firebase CLI
```powershell
npm install -g firebase-tools
firebase login
firebase init
```

Select:
- [x] Hosting
- [x] Firestore
- [x] Storage

Project: `seg-mind` (your project ID)

### 3. Enable Firebase Services
In Firebase Console:
- **Authentication** → Get Started → Enable Email/Password
- **Firestore Database** → Create Database → Start in production mode
- **Storage** → Get Started

### 4. Download Service Account Key
```
Firebase Console → Project Settings → Service Accounts
→ Generate New Private Key → Save as `firebase-credentials.json`
```

**IMPORTANT**: Add to `.gitignore`:
```
firebase-credentials.json
```

---

## Frontend Deployment (5 min)

### Build Frontend
```powershell
cd frontend
npm run build
# Output: dist/ folder
```

### Update firebase.json
```json
{
  "hosting": {
    "public": "frontend/dist",  // Change from "build" to "frontend/dist"
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "rewrites": [{"source": "**", "destination": "/index.html"}]
  }
}
```

### Deploy to Firebase Hosting
```powershell
firebase deploy --only hosting
```

**Result**: Frontend live at `https://seg-mind.web.app`

---

## Backend Deployment Options

### **Option A: Cloud Run (Recommended - Easiest)**

#### 1. Create Dockerfile
```dockerfile
# Save as: Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
ENV PORT=8080
EXPOSE 8080

# Run with uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### 2. Enable GCP APIs
```powershell
gcloud auth login
gcloud config set project seg-mind

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

#### 3. Create Cloud SQL Instance
```powershell
gcloud sql instances create segmind-db \
  --database-version=POSTGRES_14 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database
gcloud sql databases create segmind_db --instance=segmind-db

# Set password
gcloud sql users set-password postgres \
  --instance=segmind-db \
  --password=YOUR_SECURE_PASSWORD
```

#### 4. Store Secrets
```powershell
# Firebase credentials
gcloud secrets create firebase-credentials --data-file=firebase-credentials.json

# Database URL
echo "postgresql://postgres:YOUR_PASSWORD@/segmind_db?host=/cloudsql/seg-mind:us-central1:segmind-db" | gcloud secrets create database-url --data-file=-

# Secret key
echo "PRODUCTION_SECRET_KEY_CHANGE_THIS_$(openssl rand -hex 32)" | gcloud secrets create secret-key --data-file=-
```

#### 5. Build & Deploy to Cloud Run
```powershell
# Build container
gcloud builds submit --tag gcr.io/seg-mind/backend

# Deploy
gcloud run deploy seg-mind-api \
  --image gcr.io/seg-mind/backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --add-cloudsql-instances seg-mind:us-central1:segmind-db \
  --set-secrets="DATABASE_URL=database-url:latest,SECRET_KEY=secret-key:latest,FIREBASE_CREDENTIALS_PATH=firebase-credentials:latest" \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300
```

**Result**: Backend live at `https://seg-mind-api-xxx-uc.a.run.app`

#### 6. Initialize Database
```powershell
# Connect via Cloud SQL Proxy
gcloud sql connect segmind-db --user=postgres

# Or use Cloud Shell
gcloud run services proxy seg-mind-api --port=8080

# Run migrations
python -c "from backend.app.database import init_db; init_db()"
```

---

### **Option B: App Engine (Alternative)**

```powershell
# Create app.yaml
cat > app.yaml << EOF
runtime: python310
entrypoint: uvicorn backend.app.main:app --host 0.0.0.0 --port \$PORT

env_variables:
  DATABASE_URL: "postgresql://..."
  SECRET_KEY: "..."

instance_class: F2
automatic_scaling:
  max_instances: 10
  min_instances: 0
EOF

# Deploy
gcloud app deploy
```

---

## Post-Deployment Configuration

### 1. Update Frontend API URL
**frontend/src/services/api.ts**:
```typescript
export const API_BASE_URL = 
  import.meta.env.VITE_API_URL || 
  'https://seg-mind-api-xxx-uc.a.run.app';  // Your Cloud Run URL
```

Rebuild and redeploy:
```powershell
cd frontend
npm run build
firebase deploy --only hosting
```

### 2. Update CORS Origins
**backend/app/main.py**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://seg-mind.web.app",
        "https://seg-mind.firebaseapp.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Redeploy backend:
```powershell
gcloud builds submit --tag gcr.io/seg-mind/backend
gcloud run deploy seg-mind-api --image gcr.io/seg-mind/backend
```

### 3. Deploy Security Rules
```powershell
firebase deploy --only firestore:rules
firebase deploy --only storage:rules
```

### 4. Create Admin User
```powershell
# Via Cloud Run console or API
curl -X POST https://seg-mind-api-xxx-uc.a.run.app/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@seg-mind.com",
    "password": "SecureAdmin123",
    "full_name": "Admin User",
    "role": "admin",
    "medical_license": "ADMIN-001",
    "specialization": "System Administration",
    "institution": "Seg-Mind",
    "department": "IT"
  }'

# Set Firebase custom claim
firebase auth:import admin.json --project seg-mind
```

---

## Testing Deployment

### 1. Test Frontend
```
https://seg-mind.web.app
```
- ✅ Login page loads
- ✅ Assets load (no 404s)
- ✅ Security headers present (check DevTools → Network)

### 2. Test Backend
```powershell
# Health check
curl https://seg-mind-api-xxx-uc.a.run.app/health

# API docs
https://seg-mind-api-xxx-uc.a.run.app/docs
```

### 3. Test Authentication
```powershell
# Register test user
curl -X POST https://seg-mind-api-xxx-uc.a.run.app/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@test.com", "password": "Test1234", ...}'

# Login
curl -X POST https://seg-mind-api-xxx-uc.a.run.app/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@test.com", "password": "Test1234"}'
```

### 4. Test Audit Logs
```powershell
# Access protected resource (triggers audit log)
curl -H "Authorization: Bearer <admin_token>" \
  https://seg-mind-api-xxx-uc.a.run.app/api/v1/security/audit-logs
```

---

## Monitoring Setup (Optional)

### 1. Cloud Logging
```
GCP Console → Logging → Logs Explorer
Filter: resource.type="cloud_run_revision"
```

### 2. Cloud Monitoring
```
GCP Console → Monitoring → Dashboards
Create dashboard for:
- Request rate
- Error rate
- Response latency
- CPU/Memory usage
```

### 3. Uptime Checks
```
Monitoring → Uptime Checks → Create Check
URL: https://seg-mind-api-xxx-uc.a.run.app/health
Frequency: 5 minutes
```

---

## Cost Estimates

### Free Tier Limits
- **Firebase Hosting**: 10GB storage, 360MB/day transfer (free)
- **Firestore**: 1GB storage, 50k reads/day (free)
- **Cloud Run**: 2M requests/month (free)
- **Cloud SQL**: db-f1-micro ~$7/month (no free tier)

### Expected Monthly Costs
- **Development**: ~$10-20 (Cloud SQL + minimal traffic)
- **Production**: ~$50-200 (depends on traffic)

### Cost Optimization
- Use Cloud Run (scales to zero vs App Engine always-on)
- Use db-f1-micro for Cloud SQL (smallest instance)
- Enable Cloud CDN for static assets

---

## Troubleshooting

### Frontend won't load
```powershell
# Check build output
cd frontend && npm run build

# Verify firebase.json paths
cat firebase.json
```

### Backend 500 errors
```powershell
# Check logs
gcloud run services logs read seg-mind-api --limit=50

# Common issues:
# - DATABASE_URL not set correctly
# - Cloud SQL connection failed
# - Missing secrets
```

### Database connection fails
```powershell
# Test connection
gcloud sql connect segmind-db --user=postgres

# Check Cloud SQL instance running
gcloud sql instances list
```

### CORS errors
- Update `allow_origins` in backend/app/main.py
- Include both .web.app and .firebaseapp.com URLs
- Redeploy backend after changes

---

## Security Checklist

- [ ] Firebase credentials in Secret Manager (not in code)
- [ ] Database password secure (not default)
- [ ] SECRET_KEY randomly generated
- [ ] CORS origins restricted to your domains
- [ ] Firestore rules deployed and tested
- [ ] Storage rules deployed and tested
- [ ] HTTPS enforced (automatic with Firebase/Cloud Run)
- [ ] Audit logging enabled

---

## Next Steps After Deployment

1. **Monitor**: Set up alerts for errors/downtime
2. **Backup**: Enable automated Cloud SQL backups
3. **SSL Certificate**: Already handled by Firebase/Cloud Run
4. **Custom Domain** (optional): 
   ```
   Firebase Console → Hosting → Add custom domain
   ```
5. **CI/CD** (optional): Set up GitHub Actions for auto-deploy

---

## Quick Commands Reference

```powershell
# Deploy frontend only
firebase deploy --only hosting

# Deploy backend only
gcloud run deploy seg-mind-api --image gcr.io/seg-mind/backend

# Deploy security rules
firebase deploy --only firestore:rules,storage:rules

# View logs
gcloud run services logs read seg-mind-api --limit=100

# Connect to database
gcloud sql connect segmind-db --user=postgres

# List deployed services
gcloud run services list
firebase hosting:sites:list
```

---

**Estimated Total Time**: 1-2 hours (first deployment)

**Difficulty**: Medium (follow steps carefully)

**Cost**: ~$10-50/month
