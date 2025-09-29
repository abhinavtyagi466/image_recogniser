# 🔧 Windows Installation Fix Guide

## Issue: Windows Long Path Support Error

The error you're encountering is due to Windows Long Path support not being enabled. Here are multiple solutions:

## Solution 1: Enable Windows Long Path Support (Recommended)

### For Windows 10/11:

1. **Open Group Policy Editor:**
   - Press `Win + R`, type `gpedit.msc`, press Enter
   - Navigate to: `Computer Configuration > Administrative Templates > System > Filesystem`
   - Find "Enable Win32 long paths"
   - Double-click and set to "Enabled"
   - Click OK and restart your computer

2. **Alternative Registry Method:**
   - Press `Win + R`, type `regedit`, press Enter
   - Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Find `LongPathsEnabled` and set its value to `1`
   - Restart your computer

## Solution 2: Use Conda Instead of Pip

```bash
# Install Miniconda first from: https://docs.conda.io/en/latest/miniconda.html

# Create new environment
conda create -n image_recognizer python=3.9
conda activate image_recognizer

# Install packages via conda
conda install tensorflow pytorch torchvision -c pytorch
conda install opencv pillow numpy pandas matplotlib scikit-learn
pip install ultralytics transformers fastapi uvicorn requests tqdm
```

## Solution 3: Install Packages Individually

```bash
# Activate your virtual environment
.venv\Scripts\activate

# Install packages one by one with shorter paths
pip install --no-cache-dir tensorflow-cpu==2.13.0
pip install --no-cache-dir torch==2.0.1
pip install --no-cache-dir torchvision==0.15.2
pip install --no-cache-dir transformers==4.30.2
pip install --no-cache-dir ultralytics==8.0.196
pip install --no-cache-dir opencv-python==4.8.0.74
pip install --no-cache-dir pillow==10.0.0
pip install --no-cache-dir numpy==1.24.3
pip install --no-cache-dir pandas==2.0.3
pip install --no-cache-dir matplotlib==3.7.2
pip install --no-cache-dir fastapi==0.100.0
pip install --no-cache-dir uvicorn==0.22.0
pip install --no-cache-dir requests==2.31.0
pip install --no-cache-dir tqdm==4.65.0
pip install --no-cache-dir scikit-learn==1.3.0
```

## Solution 4: Face Recognition Installation (Windows Specific)

Face recognition requires additional steps on Windows:

```bash
# Install Visual Studio Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then install in this order:
pip install cmake
pip install dlib
pip install face-recognition
```

## Solution 5: Use Pre-compiled Wheels

```bash
# Download pre-compiled wheels from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/

# Install using pip with local files
pip install path\to\downloaded\wheel.whl
```

## Quick Fix Commands

Run these commands in PowerShell as Administrator:

```powershell
# Enable long paths via PowerShell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Restart required after this
```

## Alternative: Use Docker

If all else fails, use Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "scripts/runengine.py"]
```

## Verification

After installation, verify with:

```python
import tensorflow as tf
import torch
import cv2
import ultralytics
print("All packages installed successfully!")
```

## Troubleshooting

1. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

2. **Use shorter path:**
   ```bash
   # Move project to C:\dev\ instead of long OneDrive path
   ```

3. **Use --no-deps flag:**
   ```bash
   pip install --no-deps tensorflow-cpu
   ```

Choose the solution that works best for your system!
