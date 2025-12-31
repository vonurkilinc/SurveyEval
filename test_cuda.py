#!/usr/bin/env python3
"""Test CUDA availability and provide diagnostic information."""
import torch
import sys

print("=" * 60)
print("CUDA Diagnostic Test")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\n✓ CUDA is working!")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test GPU computation
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.matmul(x, y)
        print(f"\n✓ GPU computation test: SUCCESS")
        print(f"Result tensor device: {z.device}")
    except Exception as e:
        print(f"\n✗ GPU computation test: FAILED")
        print(f"Error: {e}")
else:
    print(f"\n✗ CUDA is NOT available")
    print("\nPossible reasons:")
    print("1. NVIDIA drivers are not installed or outdated")
    print("2. CUDA runtime libraries are missing")
    print("3. GPU is not properly connected")
    print("\nTo fix:")
    print("1. Install/update NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
    print("2. Restart your computer after installing drivers")
    print("3. Verify with: nvidia-smi (in admin PowerShell)")

print("=" * 60)

