# PowerShell script to install PyTorch with CUDA support in surveyeval conda environment
# This script installs PyTorch 2.x with CUDA 12.1 support

Write-Host "Installing PyTorch with CUDA support in surveyeval environment..." -ForegroundColor Green

# Activate conda environment and install PyTorch with CUDA
conda activate surveyeval

# Check CUDA version (if NVIDIA drivers are installed)
Write-Host "`nChecking CUDA availability..." -ForegroundColor Yellow
python -c "import subprocess; result = subprocess.run(['nvidia-smi'], capture_output=True, text=True); print(result.stdout if result.returncode == 0 else 'nvidia-smi not found - CUDA drivers may not be installed')"

Write-Host "`nInstalling PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "`nVerifying installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

Write-Host "`nInstallation complete!" -ForegroundColor Green

