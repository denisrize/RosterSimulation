"""
Data Download Module

Handles downloading required datasets from external hosting (Zenodo).
"""

import os
import sys
import zipfile
import tempfile
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from typing import Optional, Callable

# Zenodo record URL - update this with your actual Zenodo record ID
ZENODO_RECORD_ID = "18493146"
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_DOWNLOAD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

# Required data files and their expected locations within the datasets/ folder
REQUIRED_FILES = [
    "rider_features.csv",
    "trueskill_leader.csv",
    "equal_weight_trueskill_team.csv",
    "position_trueskill_team.csv",
    # "time_lag_trueskill_team.csv",
]

# Package data directory
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_UTILS_DIR)
_DATA_DIR = os.path.join(_PACKAGE_DIR, "data")
_DATASETS_DIR = os.path.join(_DATA_DIR, "datasets")


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Display download progress."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()  # New line when complete


def check_data_status() -> dict:
    """
    Check which data files are present and which are missing.
    
    Returns:
        Dictionary with status of each required file
    """
    status = {
        "datasets_dir": _DATASETS_DIR,
        "files": {},
        "all_present": True,
        "missing_count": 0,
    }
    
    for filename in REQUIRED_FILES:
        filepath = os.path.join(_DATASETS_DIR, filename)
        exists = os.path.exists(filepath)
        size_mb = 0
        if exists:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        status["files"][filename] = {
            "exists": exists,
            "path": filepath,
            "size_mb": round(size_mb, 2) if exists else None,
        }
        
        if not exists:
            status["all_present"] = False
            status["missing_count"] += 1
    
    return status


def download_from_url(url: str, dest_path: str, 
                      progress_callback: Optional[Callable] = None) -> bool:
    """
    Download a file from URL to destination path.
    
    Args:
        url: URL to download from
        dest_path: Local path to save to
        progress_callback: Optional progress callback function
        
    Returns:
        True if successful, False otherwise
    """
    try:
        callback = progress_callback or _progress_hook
        urlretrieve(url, dest_path, callback)
        return True
    except (URLError, HTTPError) as e:
        print(f"\n  Error downloading: {e}")
        return False


def download_data(force: bool = False, verbose: bool = True) -> bool:
    """
    Download required datasets from Zenodo.
    
    Args:
        force: Re-download even if files exist
        verbose: Print progress messages
        
    Returns:
        True if all files are available after operation
    """
    status = check_data_status()
    
    if status["all_present"] and not force:
        if verbose:
            print("All required data files are already present.")
            print(f"Location: {_DATASETS_DIR}")
        return True
    
    # Ensure datasets directory exists
    os.makedirs(_DATASETS_DIR, exist_ok=True)
    
    if verbose:
        print(f"Data directory: {_DATASETS_DIR}")
        print(f"Zenodo record: {ZENODO_RECORD_ID}")
        print()
    
    # Try to download individual files
    # Zenodo file download URL format: https://zenodo.org/records/{id}/files/{filename}?download=1
    success_count = 0
    error_count = 0
    
    for filename in REQUIRED_FILES:
        filepath = os.path.join(_DATASETS_DIR, filename)
        
        if os.path.exists(filepath) and not force:
            if verbose:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"[OK] {filename} ({size_mb:.1f} MB) - already exists")
            success_count += 1
            continue
        
        if verbose:
            print(f"[..] Downloading {filename}...")
        
        url = f"{ZENODO_DOWNLOAD_URL}/{filename}?download=1"
        
        if download_from_url(url, filepath):
            success_count += 1
            if verbose:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"     Downloaded ({size_mb:.1f} MB)")
        else:
            error_count += 1
            if verbose:
                print(f"     FAILED - Please download manually from Zenodo")
    
    if verbose:
        print()
        print(f"Summary: {success_count}/{len(REQUIRED_FILES)} files available")
        
        if error_count > 0:
            print()
            print("Some files could not be downloaded automatically.")
            print("Please download manually from:")
            print(f"  https://zenodo.org/records/{ZENODO_RECORD_ID}")
            print()
            print(f"Place the CSV files in:")
            print(f"  {_DATASETS_DIR}")
    
    return error_count == 0


def print_data_status() -> None:
    """Print formatted status of data files."""
    status = check_data_status()
    
    print("Data Status")
    print("=" * 60)
    print(f"Location: {status['datasets_dir']}")
    print()
    
    for filename, info in status["files"].items():
        if info["exists"]:
            print(f"  [OK] {filename} ({info['size_mb']} MB)")
        else:
            print(f"  [MISSING] {filename}")
    
    print()
    if status["all_present"]:
        print("All required datasets are available.")
    else:
        print(f"Missing {status['missing_count']} file(s).")
        print()
        print("To download missing files, run:")
        print("  python main.py download-data")
        print()
        print("Or download manually from Zenodo:")
        print(f"  https://zenodo.org/records/{ZENODO_RECORD_ID}")
