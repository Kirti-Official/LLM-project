
# VERDICT - Windows Setup Script

$ErrorActionPreference = "Stop"

Write-Host "`n====================================" -ForegroundColor Cyan
Write-Host "       VERDICT PROJECT SETUP" -ForegroundColor Cyan
Write-Host "====================================`n" -ForegroundColor Cyan

# Move to project root
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Locate Python 3.11
$PyLauncher = Get-Command py -ErrorAction SilentlyContinue

if ($PyLauncher) {
    Write-Host "Checking Python 3.11..."
    & py -3.11 --version

    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 is required. Please install Python 3.11."
    }

    $CreateVenv = { & py -3.11 -m venv verdict_env }
}
else {
    Write-Host "Checking installed Python..."
    & python --version

    if ($LASTEXITCODE -ne 0) {
        throw "Python is not installed or not available in PATH."
    }

    $CreateVenv = { & python -m venv verdict_env }
}

# Create virtual environment if missing
if (-not (Test-Path ".\verdict_env\Scripts\python.exe")) {
    Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
    & $CreateVenv

    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed."
    }
}
else {
    Write-Host "`nExisting virtual environment detected."
}

$VenvPython = ".\verdict_env\Scripts\python.exe"

# Upgrade pip
Write-Host "`nUpgrading pip..."
& $VenvPython -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {
    throw "Pip upgrade failed."
}

# Install project dependencies
Write-Host "`nInstalling VERDICT dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

# Verify dependencies
Write-Host "`nChecking dependency compatibility..."
& $VenvPython -m pip check

if ($LASTEXITCODE -ne 0) {
    throw "Dependency compatibility check failed."
}

# Check Ollama separately
Write-Host "`nChecking Ollama installation..."

$Ollama = Get-Command ollama -ErrorAction SilentlyContinue

if ($Ollama) {
    Write-Host "Ollama is installed." -ForegroundColor Green
    Write-Host "`nAvailable Ollama models:"
    & ollama list
}
else {
    Write-Host "Ollama is not installed." -ForegroundColor Yellow
    Write-Host "Install Ollama separately before running LLM generation."
}

Write-Host "`n====================================" -ForegroundColor Green
Write-Host "       SETUP COMPLETED" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green

Write-Host "`nTo activate the environment, run:"
Write-Host ".\verdict_env\Scripts\Activate.ps1" -ForegroundColor Cyan

Write-Host "`nFor LLM inference, ensure Ollama has the model:"
Write-Host "llama3.1:8b" -ForegroundColor Cyan
