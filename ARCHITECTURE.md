# Vercel Deployment Architecture

## Problem: Original Structure ❌

```
Vercel looks for entrypoint in:
├── app.py          ❌ Not found
├── src/app.py      ❌ Not found
├── api/app.py      ❌ Not found
├── api/main.py     ❌ Not found
└── ...             ❌ Not found

Actual app location:
└── backend/app/main.py  ⚠️ Not discoverable by Vercel
```

**Result**: Deployment fails with "No fastapi entrypoint found"

---

## Solution: New Structure ✅

```
Repository Root
│
├── api/                          ✅ NEW - Vercel standard location
│   ├── __init__.py              ✅ Package initialization
│   └── main.py                  ✅ Entrypoint (imports from backend)
│
├── backend/
│   └── app/
│       └── main.py              ⚠️ Original app (unchanged)
│
├── vercel.json                  ✅ Vercel configuration
├── runtime.txt                  ✅ Python version
├── pyproject.toml               ✅ Package metadata
└── .vercelignore                ✅ Deployment exclusions
```

---

## Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Vercel Cloud                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Vercel Detects │
                    │   Python Project │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Reads config   │
                    │  vercel.json     │
                    │  runtime.txt     │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Finds Entry    │
                    │  api/main.py ✅   │
                    └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       api/main.py                            │
│  ┌────────────────────────────────────────────────────┐     │
│  │ import sys                                          │     │
│  │ from pathlib import Path                           │     │
│  │                                                     │     │
│  │ # Add project root to path                         │     │
│  │ root_dir = Path(__file__).resolve().parent.parent  │     │
│  │ sys.path.insert(0, str(root_dir))                  │     │
│  │                                                     │     │
│  │ # Import actual app                                │     │
│  │ from backend.app.main import app  ←────────────┐   │     │
│  └────────────────────────────────────────────────┘   │     │
└───────────────────────────────────────────────────────┼─────┘
                                                        │
                                                        │
┌───────────────────────────────────────────────────────┼─────┐
│                  backend/app/main.py                  │     │
│  ┌────────────────────────────────────────────────────▼┐    │
│  │ from fastapi import FastAPI                         │    │
│  │                                                      │    │
│  │ app = FastAPI(                                      │    │
│  │     title="Seg-Mind API",                           │    │
│  │     version="1.0.0",                                │    │
│  │     description="Brain Tumor Analysis"              │    │
│  │ )                                                    │    │
│  │                                                      │    │
│  │ @app.get("/")                                       │    │
│  │ async def root():                                   │    │
│  │     return {"message": "API is running"}            │    │
│  │                                                      │    │
│  │ # ... rest of the application ...                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   FastAPI App    │
                    │     Running      │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  API Endpoints:  │
                    │  /health         │
                    │  /api/v1/auth/*  │
                    │  /api/v1/upload  │
                    │  /api/v1/analyze │
                    │  /docs           │
                    └──────────────────┘
```

---

## Configuration Files Explained

### 1. vercel.json
```json
{
  "version": 2,
  "builds": [{
    "src": "api/main.py",      ← Points to entrypoint
    "use": "@vercel/python"    ← Use Python builder
  }],
  "routes": [{
    "src": "/(.*)",            ← Route all requests
    "dest": "api/main.py"      ← To the entrypoint
  }]
}
```

### 2. runtime.txt
```
python-3.11    ← Specifies Python version
```

### 3. api/main.py (Bridge/Proxy)
```python
import sys
from pathlib import Path

# Add root to path for imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Import and export the actual app
from backend.app.main import app
__all__ = ["app"]
```

---

## Deployment Process

```
┌──────────────┐
│ Push to Git  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Vercel Detects   │
│ Push             │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Read Config      │
│ - vercel.json    │
│ - runtime.txt    │
│ - requirements   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Install Python   │
│ Dependencies     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Build Function   │
│ from api/main.py │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Deploy to        │
│ Vercel Edge      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ ✅ App Live!     │
│ your-app.vercel  │
└──────────────────┘
```

---

## File Size Optimization

```
Original Repository          Deployed to Vercel
├── ML Models (2GB)    →    ❌ Excluded (.vercelignore)
├── Training Data      →    ❌ Excluded (.vercelignore)
├── Test Files         →    ❌ Excluded (.vercelignore)
├── Frontend           →    ❌ Excluded (.vercelignore)
├── Python Code        →    ✅ Included
├── Config Files       →    ✅ Included
└── Requirements       →    ✅ Included (dependencies installed)
```

---

## Environment Variables Flow

```
Local Development (.env)
    ↓
    ├── DATABASE_URL=localhost:5432
    ├── SECRET_KEY=dev-key
    └── DEBUG_MODE=True

Production (Vercel Dashboard)
    ↓
    ├── DATABASE_URL=production-postgres-url
    ├── SECRET_KEY=secure-production-key
    └── DEBUG_MODE=False
```

---

## Key Benefits of This Architecture

✅ **Vercel Compatible**: Entrypoint in standard location
✅ **No Code Changes**: Original app unchanged
✅ **Maintainable**: Clear separation of concerns
✅ **Flexible**: Can easily switch deployment platforms
✅ **Documented**: Comprehensive guides provided

---

## Troubleshooting Guide

### Issue: Build Fails (Size Limit)
```
Problem: Dependencies > 250MB
Solution: 
  1. Use Vercel Pro (larger limits)
  2. Deploy to Railway/Google Cloud Run
  3. Use lighter alternatives (torch-cpu)
```

### Issue: Import Error
```
Problem: Module not found
Solution:
  1. Check requirements.txt has all dependencies
  2. Verify Python version matches (3.11)
  3. Check sys.path configuration in api/main.py
```

### Issue: Database Connection Failed
```
Problem: Can't connect to database
Solution:
  1. Verify DATABASE_URL in Vercel env vars
  2. Check database allows external connections
  3. Use connection pooling
```

---

## Summary

The solution creates a **bridge pattern** where:
- `api/main.py` is the **discoverable entrypoint** for Vercel
- `backend/app/main.py` is the **actual application** (unchanged)
- Configuration files tell Vercel **how to build and deploy**

This approach ensures **minimal changes** to existing code while making the application **Vercel-compatible**.

🎉 **Result**: Deployment now succeeds!
