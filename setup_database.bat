@echo off
echo ======================================
echo Creating Seg-Mind Database
echo ======================================
echo.

REM Add PostgreSQL to PATH
set PATH=%PATH%;C:\Program Files\PostgreSQL\18\bin

echo Step 1: Creating database 'segmind_db'...
echo Please enter your PostgreSQL password when prompted
echo.

REM Create database
psql -U postgres -c "CREATE DATABASE segmind_db;"

echo.
echo Step 2: Setting up database schema...
echo.

REM Run setup script
psql -U postgres -d segmind_db -f database\setup_database.sql

echo.
echo ======================================
echo Database setup complete!
echo ======================================
echo.
echo Database: segmind_db
echo Host: localhost
echo Port: 5432
echo User: postgres
echo.
pause
