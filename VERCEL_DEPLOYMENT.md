# Vercel Deployment Guide for Seg-Mind API

## Overview

This document explains how to deploy the Seg-Mind FastAPI backend to Vercel.

## Problem Resolution

**Issue**: Vercel couldn't find the FastAPI entrypoint because the app was located at `backend/app/main.py`, which is not one of Vercel's standard locations.

**Solution**: Created an entrypoint at `api/main.py` that Vercel can discover, which imports the actual FastAPI app from `backend/app/main.py`.

## File Structure

```
fyp-1/
├── api/
│   ├── __init__.py          # Package initialization
│   └── main.py              # Vercel entrypoint (imports from backend/app/main.py)
├── backend/
│   └── app/
│       └── main.py          # Actual FastAPI application
├── vercel.json              # Vercel configuration
├── pyproject.toml           # Python package configuration
└── requirements.txt         # Python dependencies
```

## Configuration Files

### 1. `api/main.py`
This is the entrypoint that Vercel discovers. It:
- Adds the project root to Python path
- Imports the FastAPI `app` instance from `backend/app/main.py`
- Exports the `app` for Vercel to use

### 2. `vercel.json`
Configures Vercel deployment:
- Specifies Python 3.11 runtime
- Points to `api/main.py` as the build source
- Routes all requests to the FastAPI app

### 3. `pyproject.toml`
Provides Python package metadata and configuration:
- Project information (name, version, description)
- Core dependencies (FastAPI, Uvicorn, etc.)
- App script entry point: `app = "backend.app.main:app"`

## Deployment Steps

### Deploy to Vercel

1. **Connect to GitHub**:
   - Go to [vercel.com](https://vercel.com)
   - Sign in with your GitHub account
   - Click "Add New Project"
   - Import the `AbdullahEjaz512/fyp-1` repository

2. **Configure Environment Variables** (if needed):
   - Set `DATABASE_URL` for PostgreSQL connection
   - Set `SECRET_KEY` for JWT authentication
   - Set any other environment variables from `.env.example`

3. **Deploy**:
   - Vercel will automatically detect the Python project
   - It will use `vercel.json` configuration
   - Build will install dependencies from `requirements.txt`
   - App will be available at `https://your-project.vercel.app`

### Environment Variables

Required environment variables (set in Vercel dashboard):
```bash
DATABASE_URL=postgresql://user:password@host:port/database
SECRET_KEY=your-secret-key-here
DEBUG_MODE=False
CORS_ORIGINS=https://your-frontend.vercel.app
```

## Testing the Deployment

After deployment, test these endpoints:

1. **Health Check**:
   ```bash
   curl https://your-project.vercel.app/health
   ```
   Expected: `{"status": "healthy", "service": "seg-mind-api"}`

2. **Root Endpoint**:
   ```bash
   curl https://your-project.vercel.app/
   ```
   Expected: API status and version information

3. **API Documentation**:
   - Swagger UI: `https://your-project.vercel.app/docs`
   - ReDoc: `https://your-project.vercel.app/redoc`

## Important Notes

### 1. Large Dependencies
The project uses heavy ML libraries (PyTorch, MONAI, etc.). Vercel has deployment size limits:
- Serverless Function: 50 MB compressed, 250 MB uncompressed
- If deployment fails due to size, consider:
  - Using Vercel Pro plan (larger limits)
  - Deploying ML models separately (e.g., AWS S3, Google Cloud Storage)
  - Loading models on-demand from external storage

### 2. Cold Starts
- First request after inactivity may be slow (cold start)
- ML models are loaded on startup, which takes time
- Consider keeping the instance warm with periodic health checks

### 3. Database
- Ensure PostgreSQL is accessible from Vercel's servers
- Use connection pooling for better performance
- Consider using Vercel Postgres or external managed database

### 4. File Uploads
- Uploaded files (MRI scans) are stored in `data/uploads/`
- On Vercel, filesystem is ephemeral (files are lost after function execution)
- Consider using:
  - Vercel Blob Storage
  - AWS S3
  - Google Cloud Storage
  - Azure Blob Storage

## Troubleshooting

### Issue: "No fastapi entrypoint found"
**Solution**: This is now fixed with the `api/main.py` entrypoint.

### Issue: "Build failed - dependencies too large"
**Solutions**:
1. Remove unnecessary dependencies from `requirements.txt`
2. Use lighter alternatives (e.g., `torch-cpu` instead of full `torch`)
3. Upgrade to Vercel Pro plan
4. Deploy to a different platform (AWS Lambda, Google Cloud Run, Railway)

### Issue: "Import error" during deployment
**Solution**: Ensure all dependencies are in `requirements.txt` and Python version matches (3.11)

### Issue: "Database connection failed"
**Solution**: 
1. Verify `DATABASE_URL` is set correctly in Vercel environment variables
2. Ensure database allows connections from Vercel's IP addresses
3. Use `psycopg2-binary` instead of `psycopg2`

## Alternative Deployment Options

If Vercel doesn't work due to size constraints or other issues:

### 1. Railway
- Better for large Python apps with ML dependencies
- Simple deployment from GitHub
- Generous free tier

### 2. Google Cloud Run
- Supports larger containers
- Good for ML workloads
- Auto-scaling

### 3. AWS Lambda with API Gateway
- Serverless option
- May need to optimize for size
- Use Lambda Layers for dependencies

### 4. Traditional VPS/Cloud VM
- DigitalOcean, Linode, AWS EC2
- Full control over environment
- No size limitations

## Support

For issues or questions:
1. Check Vercel deployment logs
2. Review the FastAPI app logs
3. Test locally first: `uvicorn api.main:app --reload`
4. Refer to [Vercel Python Documentation](https://vercel.com/docs/functions/serverless-functions/runtimes/python)

## Summary

The deployment is now configured for Vercel with:
- ✅ FastAPI entrypoint at `api/main.py`
- ✅ Vercel configuration in `vercel.json`
- ✅ Python package metadata in `pyproject.toml`
- ✅ All dependencies in `requirements.txt`

Your Seg-Mind API is ready to deploy! 🚀
