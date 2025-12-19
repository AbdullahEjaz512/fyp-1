# 🆓 Free Deployment Guide - Zero Cost

## Recommended: Vercel + Railway (5 minutes)

### Step 1: Deploy Frontend to Vercel (2 min)

```powershell
# Install Vercel CLI
npm install -g vercel

# Navigate to frontend
cd frontend

# Deploy
vercel login  # Login with GitHub
vercel        # Follow prompts, press Enter for defaults
```

**Output**: `https://seg-mind.vercel.app` (your live URL)

**What you get FREE:**
- ✅ Unlimited deployments
- ✅ 100GB bandwidth/month
- ✅ Auto HTTPS
- ✅ Auto-deploy on git push
- ✅ Custom domain support

---

### Step 2: Deploy Backend to Railway (3 min)

#### A. Create Railway Account
1. Go to https://railway.app
2. Click "Login with GitHub"
3. Authorize Railway

#### B. Deploy Backend
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your `fyp` repository
4. Railway auto-detects Python and deploys!

#### C. Add PostgreSQL Database
1. In your project, click "New"
2. Select "Database" → "PostgreSQL"
3. Railway provides `DATABASE_URL` automatically

#### D. Set Environment Variables
Click on backend service → Variables tab:
```
SECRET_KEY=my-super-secret-key-change-this
ENABLE_AUDIT_LOGS=true
PORT=8000
```

#### E. Get Your Backend URL
```
Settings → Domains → Generate Domain
```
**Output**: `https://seg-mind-backend.railway.app`

**What you get FREE:**
- ✅ $5/month credit (500+ hours runtime)
- ✅ PostgreSQL database included
- ✅ Auto HTTPS
- ✅ Auto-deploy on git push
- ✅ Free for hobby projects

---

### Step 3: Connect Frontend to Backend (1 min)

Update `frontend/src/services/api.ts`:
```typescript
export const API_BASE_URL = 'https://seg-mind-backend.railway.app';
```

Redeploy frontend:
```powershell
cd frontend
vercel --prod
```

---

### Step 4: Initialize Database

#### Option A: Railway Console
1. Railway Dashboard → PostgreSQL → Connect
2. Opens psql terminal
3. Run:
```sql
-- Railway already connected to your database
\i /path/to/database/setup_database.sql
```

#### Option B: Python Script
```powershell
# Set DATABASE_URL locally
$env:DATABASE_URL="postgresql://user:pass@...railway.app/railway"

# Run migration
python -c "from backend.app.database import init_db; init_db()"
```

---

## Alternative: Render.com (All-in-One)

### One-Click Deploy with render.yaml

Create `render.yaml` in project root:

```yaml
services:
  # Backend API
  - type: web
    name: seg-mind-api
    env: python
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: segmind-db
          property: connectionString
      - key: SECRET_KEY
        generateValue: true
      - key: ENABLE_AUDIT_LOGS
        value: true
    healthCheckPath: /health

  # Frontend Static Site
  - type: web
    name: seg-mind-frontend
    env: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: frontend/dist
    routes:
      - type: rewrite
        source: /*
        destination: /index.html

databases:
  - name: segmind-db
    databaseName: segmind_db
    user: postgres
```

### Deploy to Render
```powershell
# 1. Push render.yaml to GitHub
git add render.yaml
git commit -m "Add Render config"
git push

# 2. Go to https://dashboard.render.com
# 3. New → Blueprint
# 4. Connect GitHub repo
# 5. Render deploys everything automatically!
```

**What you get FREE:**
- ✅ 750 hours/month (always-on)
- ✅ PostgreSQL 90 days free trial
- ✅ Auto SSL
- ✅ Auto-deploy on push

---

## Comparison Table

| Service | Frontend | Backend | Database | Free Limits | Best For |
|---------|----------|---------|----------|-------------|----------|
| **Vercel + Railway** | ✅ Unlimited | ✅ $5/month | ✅ Included | 500hrs backend | **Recommended** |
| **Render** | ✅ 100GB | ✅ 750hrs | ⚠️ 90 days | Then $7/month DB | All-in-one |
| **GitHub Pages + Fly.io** | ✅ 1GB | ✅ 160GB | ❌ Need Supabase | 3 VMs free | Static-heavy |
| **Netlify + Heroku** | ✅ 100GB | ⚠️ Deprecated | ❌ | Heroku ending free | Not recommended |

---

## Testing Your Deployment

### 1. Test Frontend
```
https://seg-mind.vercel.app
```
- ✅ Page loads
- ✅ No console errors
- ✅ HTTPS works

### 2. Test Backend Health
```powershell
curl https://seg-mind-backend.railway.app/health
```

Expected response:
```json
{"status": "healthy", "service": "seg-mind-api"}
```

### 3. Test API Docs
```
https://seg-mind-backend.railway.app/docs
```

### 4. Test Authentication
```powershell
# Register user
curl -X POST https://seg-mind-backend.railway.app/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@test.com",
    "password": "Test1234",
    "full_name": "Test User",
    "role": "doctor",
    "medical_license": "TEST-001",
    "specialization": "Testing",
    "institution": "Test Hospital",
    "department": "QA"
  }'

# Login
curl -X POST https://seg-mind-backend.railway.app/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@test.com", "password": "Test1234"}'
```

---

## Troubleshooting

### Backend won't start on Railway
**Check logs**: Railway Dashboard → Service → Logs tab

Common issues:
```powershell
# Missing dependencies
# Fix: Add to requirements.txt

# Port binding error
# Fix: Railway auto-sets PORT env var, use $PORT

# Database connection error
# Fix: Use Railway's DATABASE_URL (already set)
```

### Frontend API calls failing
**CORS error**: Add Railway URL to backend CORS:

`backend/app/main.py`:
```python
CORS_ORIGINS = [
    "http://localhost:5173",
    "https://seg-mind.vercel.app",
    "https://seg-mind-backend.railway.app"
]
```

Redeploy:
```powershell
git commit -am "Fix CORS"
git push  # Railway auto-deploys
```

### Database tables not created
```powershell
# SSH into Railway
railway run python

>>> from backend.app.database import init_db
>>> init_db()
>>> exit()
```

---

## Cost Breakdown

### Free Tier Limits

**Vercel Frontend:**
- Bandwidth: 100GB/month
- Builds: 6000 minutes/month
- **Cost**: $0 forever

**Railway Backend:**
- Runtime: $5 credit = ~500 hours
- Database: 1GB storage included
- **Cost**: $0 for low-traffic (~$5-10/month for production)

**Total for FYP Demo**: **$0/month** ✅

### When You'll Need to Pay

**Railway** (only if you exceed):
- 500 hours runtime/month (~17 hours/day)
- Then pay-as-you-go: $0.000463/GB-hour

**For FYP demonstration**: Completely free ✅

---

## Upgrade Options (Optional)

If your project becomes popular:

**Vercel Pro** ($20/month):
- Unlimited bandwidth
- Team collaboration
- Analytics

**Railway** (usage-based):
- Only pay for what you use
- Typically $5-20/month for active projects

**But for FYP**: Free tier is perfect! ✅

---

## Quick Commands Summary

```powershell
# Deploy frontend (Vercel)
cd frontend
vercel --prod

# Deploy backend (Railway)
# Just push to GitHub, Railway auto-deploys
git push

# View backend logs
railway logs

# Run database migration
railway run python -c "from backend.app.database import init_db; init_db()"

# Check deployment status
railway status
```

---

## Benefits of This Setup

✅ **No credit card required**  
✅ **Auto-deploy on git push**  
✅ **Free SSL certificates**  
✅ **Zero downtime deploys**  
✅ **Logs and monitoring included**  
✅ **Perfect for FYP demonstrations**  
✅ **Can scale to production if needed**  

---

## Next Steps

1. ✅ Deploy to Vercel + Railway (5 min)
2. ✅ Test all endpoints work
3. ✅ Create demo admin account
4. ✅ Show your supervisor! 🎉

---

**Total Time**: 5-10 minutes  
**Total Cost**: $0  
**Difficulty**: Very Easy  

Perfect for your FYP! 🚀
