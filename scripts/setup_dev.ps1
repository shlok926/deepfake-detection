# ==============================================================================
# ENTERPRISE FORENSICS PLATFORM WINDOWS SETUP SCRIPT (scripts/setup_dev.ps1)
# ==============================================================================

Write-Host "=== Starting AI Forensics Platform Windows Dev Setup ===" -ForegroundColor Cyan

# 1. Verify python is installed
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python interpreter not found. Please install Python 3.10+ and add it to your PATH."
    Exit 1
}

# 2. Initialize directories
Write-Host "Auto-generating developer folder structure..." -ForegroundColor Yellow
python app/utils/bootstrap.py

# 3. Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment in './venv'..." -ForegroundColor Yellow
    python -m venv venv
}

# 4. Install dependencies
Write-Host "Installing packages and dependencies..." -ForegroundColor Yellow
& venv/Scripts/pip install --upgrade pip
& venv/Scripts/pip install -r requirements.txt
& venv/Scripts/pip install black isort flake8 mypy pre-commit pytest pytest-cov httpx

# 5. Setup Git Pre-commit hook
if (Test-Path ".git") {
    Write-Host "Configuring git pre-commit hooks..." -ForegroundColor Yellow
    & venv/Scripts/pre-commit install
}

# 6. Copy local environment template
if (-not (Test-Path ".env")) {
    Write-Host "Creating local .env from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

Write-Host "=== Dev environment successfully initialized! ===" -ForegroundColor Green
Write-Host "Run 'venv/Scripts/activate' to start working." -ForegroundColor Green
