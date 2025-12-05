# Seg-Mind Database Setup - PowerShell Script
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Creating Seg-Mind Database" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Add PostgreSQL to PATH
$env:Path += ";C:\Program Files\PostgreSQL\18\bin"

# Set password environment variable
$env:PGPASSWORD = "postgres123"

Write-Host "Step 1: Creating database 'segmind_db'..." -ForegroundColor Yellow
Write-Host ""

# Create database
$createDb = psql -U postgres -c "CREATE DATABASE segmind_db;" 2>&1

if ($LASTEXITCODE -eq 0 -or $createDb -like "*already exists*") {
    Write-Host "✓ Database created or already exists" -ForegroundColor Green
} else {
    Write-Host "✗ Error creating database: $createDb" -ForegroundColor Red
}

Write-Host ""
Write-Host "Step 2: Setting up database schema..." -ForegroundColor Yellow
Write-Host ""

# Run setup script
psql -U postgres -d segmind_db -f database\setup_database.sql

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Database setup complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Database Details:" -ForegroundColor White
Write-Host "  Database: segmind_db" -ForegroundColor White
Write-Host "  Host: localhost" -ForegroundColor White
Write-Host "  Port: 5432" -ForegroundColor White
Write-Host "  User: postgres" -ForegroundColor White
Write-Host ""

# Clear password from environment
$env:PGPASSWORD = ""

Read-Host "Press Enter to continue"
