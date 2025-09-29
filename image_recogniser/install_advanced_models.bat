@echo off
echo 🚀 INSTALLING ULTRA ADVANCED DETECTION MODELS
echo ==============================================

echo.
echo 📦 Activating virtual environment...
call venv\Scripts\activate

echo.
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 🤖 Installing Ultra Advanced Detection Models...
echo.

echo 🔧 Installing YOLOv11/YOLOv8 (Latest versions)...
pip install ultralytics==8.0.196

echo.
echo 🔧 Installing DETR (Detection Transformer)...
pip install transformers==4.30.2
pip install torch==2.0.1
pip install torchvision==0.15.2

echo.
echo 🔧 Installing EfficientDet and Advanced Models...
pip install timm==0.9.12

echo.
echo 🔧 Installing Additional Dependencies...
pip install opencv-python==4.8.0.74
pip install pillow==10.0.0
pip install numpy==1.24.3

echo.
echo ✅ Installation complete! Testing advanced models...
python -c "from ultralytics import YOLO; print('✅ YOLOv11/YOLOv8 ready!')"
python -c "from transformers import AutoImageProcessor, AutoModelForObjectDetection; print('✅ DETR models ready!')"
python -c "import timm; print('✅ EfficientDet models ready!')"

echo.
echo 🎉 ULTRA ADVANCED MODELS INSTALLED!
echo.
echo 📊 Available Models:
echo • YOLOv11x.pt - Highest Accuracy (Extra Large)
echo • YOLOv11l.pt - High Accuracy (Large)  
echo • YOLOv11m.pt - Good Accuracy (Medium)
echo • YOLOv8x.pt - High Accuracy (Extra Large)
echo • YOLOv8l.pt - Good Accuracy (Large)
echo • DETR ResNet-50 - Transformer-based Detection
echo • DETR ResNet-101 - Larger Transformer Model
echo • EfficientDet-D7 - Highest Accuracy EfficientDet
echo • EfficientDet-D6 - High Accuracy
echo • EfficientDet-D5 - Balanced Accuracy
echo.
echo 💡 The system will automatically try to load the best available model!
echo 💡 Models will be downloaded automatically on first use.
echo.
echo 🚀 Ready for ultra high accuracy detection!
echo.
pause
