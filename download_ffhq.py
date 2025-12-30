#!/usr/bin/env python3
"""
download_ffhq.py
----------------
Downloads FFHQ (Flickr-Faces-HQ) dataset at 1024x1024 resolution.

The FFHQ dataset contains 70,000 high-quality PNG images at 1024x1024 resolution.
Total size: ~89.1 GB

This script uses the official download method from NVIDIA's FFHQ repository.

Usage:
    python download_ffhq.py --output-dir ./ffhq_1024
    python download_ffhq.py --output-dir ./ffhq_1024 --metadata
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_dependencies():
    """Check and install required dependencies."""
    required_packages = ['requests', 'tqdm']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", *missing, "-q"
            ])
            print("✓ Dependencies installed")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {', '.join(missing)}")
            print(f"Please install manually: pip install {' '.join(missing)}")
            return False
    
    return True


def pre_download_metadata(output_dir: Path):
    """Pre-download metadata file using gdown to avoid issues with official script."""
    metadata_path = output_dir / "ffhq-dataset-v2.json"
    license_path = output_dir / "LICENSE.txt"
    
    # Check if metadata already exists
    if metadata_path.exists() and license_path.exists():
        # Verify file size (should be ~268 MB)
        if metadata_path.stat().st_size > 200000000:  # At least 200 MB
            print("✓ Metadata file already exists")
            return True
    
    print("\nPre-downloading metadata file (this helps avoid download issues)...")
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
    
    try:
        # Download metadata JSON
        if not metadata_path.exists() or metadata_path.stat().st_size < 200000000:
            print("Downloading ffhq-dataset-v2.json (~255 MB)...")
            gdown.download(
                'https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA',
                str(metadata_path),
                quiet=False
            )
        
        # Download LICENSE
        if not license_path.exists():
            print("Downloading LICENSE.txt...")
            gdown.download(
                'https://drive.google.com/uc?id=1SHafCugkpMZzYhbgOz0zCuYiy-hb9lYX',
                str(license_path),
                quiet=False
            )
        
        print("✓ Metadata pre-download completed")
        return True
    except Exception as e:
        print(f"⚠ Warning: Failed to pre-download metadata: {e}")
        print("  Will try using official script method...")
        return False


def clone_and_run_official_script(output_dir: Path, download_images: bool, download_metadata: bool):
    """Clone the official FFHQ repo and run their download script."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-download metadata to avoid issues
    if download_images:
        pre_download_metadata(output_dir)
    
    # Check if git is available
    if not shutil.which('git'):
        print("✗ Git is not installed or not in PATH")
        print("  Please install Git or use manual download method")
        return False
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "ffhq-dataset"
        
        print("\nCloning official FFHQ repository...")
        try:
            subprocess.check_call([
                'git', 'clone', '--depth', '1',
                'https://github.com/NVlabs/ffhq-dataset.git',
                str(repo_dir)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print("✓ Repository cloned")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to clone repository: {e}")
            return False
        
        # Check if download script exists
        download_script = repo_dir / "download_ffhq.py"
        if not download_script.exists():
            print("✗ download_ffhq.py not found in repository")
            return False
        
        # Install dependencies
        if not check_dependencies():
            return False
        
        # Run download script (it downloads to current directory, so we change to output_dir)
        success = True
        original_cwd = Path.cwd()
        
        try:
            # Change to output directory so files are downloaded there
            os.chdir(output_dir)
            
            cmd = [sys.executable, str(download_script)]
            
            if download_images:
                print("\nDownloading FFHQ 1024x1024 images (~89.1 GB)...")
                print("This may take a while depending on your internet connection...")
                print("Note: The script will first download metadata, then the images.")
                # Increase retry attempts for more reliability
                try:
                    subprocess.check_call(cmd + ['--images', '--num_attempts', '20'])
                    print("✓ Images download completed")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Failed to download images: {e}")
                    print("\nTip: The download may have partially completed.")
                    print("     You can retry - the script will resume from where it left off.")
                    success = False
            
            if download_metadata:
                print("\nDownloading FFHQ metadata...")
                try:
                    subprocess.check_call(cmd + ['--json'])
                    print("✓ Metadata download completed")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Failed to download metadata: {e}")
                    success = False
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Download FFHQ (Flickr-Faces-HQ) dataset at 1024x1024 resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download images only
  python download_ffhq.py --output-dir ./ffhq_1024
  
  # Download images and metadata
  python download_ffhq.py --output-dir ./ffhq_1024 --metadata
  
  # Download metadata only
  python download_ffhq.py --output-dir ./ffhq_1024 --metadata-only
        """
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ffhq_1024",
        help="Output directory for downloaded files (default: ./ffhq_1024)"
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Also download metadata JSON file (~254 MB)"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only metadata, skip images"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    
    download_images = not args.metadata_only
    download_metadata = args.metadata or args.metadata_only
    
    print("=" * 70)
    print("FFHQ Dataset Downloader")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    if download_images:
        print(f"Dataset size: ~89.1 GB (images)")
    if download_metadata:
        print(f"Metadata size: ~254 MB")
    print("=" * 70)
    
    # Check available space
    try:
        total, used, free = shutil.disk_usage(output_dir)
        free_gb = free / (1024**3)
        if download_images and free_gb < 100:
            print(f"\n⚠ Warning: Only {free_gb:.1f} GB free space available.")
            print("  FFHQ dataset requires ~90 GB. Download may fail.")
            response = input("  Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                return
    except Exception:
        pass
    
    # Run download
    success = clone_and_run_official_script(output_dir, download_images, download_metadata)
    
    print("\n" + "=" * 70)
    if success:
        print("✓ Download completed successfully!")
        print(f"Files saved to: {output_dir}")
        if download_images:
            images_dir = output_dir / "images1024x1024"
            if images_dir.exists():
                image_count = len(list(images_dir.glob("*.png")))
                print(f"  Found {image_count} images in {images_dir}")
    else:
        print("✗ Download encountered issues. See messages above.")
        print("\nManual download instructions:")
        print("  1. git clone https://github.com/NVlabs/ffhq-dataset.git")
        print("  2. cd ffhq-dataset")
        print("  3. python download_ffhq.py --images --dest", str(output_dir))
        if download_metadata:
            print("  4. python download_ffhq.py --json --dest", str(output_dir))
    print("=" * 70)


if __name__ == "__main__":
    main()

