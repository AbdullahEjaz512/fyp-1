@echo off
cd /d C:\Users\SCM\Documents\fyp
echo ======================================
echo Starting Seg-Mind Backend Server...
echo ======================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Installing/checking dependencies...
pip install fastapi uvicorn python-multipart pyjwt python-jose[cryptography] passlib[bcrypt] --quiet
echo.
echo Starting backend server...
echo ======================================
python backend/app/main.py
pause
