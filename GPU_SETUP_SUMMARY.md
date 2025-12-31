# GeoAdapterStep2.py GPU Setup Summary

## Changes Made

### 1. Resolution Configuration ✅
- **Default resolution set to 256x256** for training
  - Updated `TrainConfig.resolution` default from 1024 to 256
  - Command-line argument already defaults to 256
  - Both are now consistent

### 2. GPU Device Handling ✅
- **Improved GPU detection and error handling**:
  - Added explicit GPU availability checks in `load_sdxl_pipeline()`
  - Added GPU verification at the start of both training functions
  - Clear error messages if CUDA is requested but not available
  - Informative messages showing which GPU is being used

### 3. Current Status
- **PyTorch CPU-only version detected**: `2.8.0+cpu`
- **CUDA not available** - needs PyTorch with CUDA support

## Next Steps: Install PyTorch with CUDA

### Option 1: Use the provided PowerShell script
```powershell
.\install_pytorch_gpu.ps1
```

### Option 2: Manual installation
```powershell
conda activate surveyeval
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 3: Check your CUDA version first
If you have a different CUDA version (e.g., 11.8), use the appropriate index URL:
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- CUDA 12.4: `--index-url https://download.pytorch.org/whl/cu124`

## Verification

After installation, verify GPU support:
```powershell
conda activate surveyeval
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Running GeoAdapterStep2.py

Once PyTorch with CUDA is installed, the script will automatically:
- ✅ Use GPU when `--device cuda` is specified (default)
- ✅ Train at 256x256 resolution (default)
- ✅ Show clear messages about which device is being used

Example command:
```powershell
conda activate surveyeval
python GeoAdapterStep2.py --mode ffhq --device cuda --resolution 256 --batch-size 2 --epochs 1
```

## Notes

- The script defaults to `--device cuda` and `--resolution 256`
- If CUDA is not available, it will raise a clear error message
- The `load_sdxl_pipeline()` function will fall back to CPU with a warning if CUDA is requested but unavailable
- Training functions will raise an error if CUDA is requested but unavailable (to prevent silent CPU fallback during training)

