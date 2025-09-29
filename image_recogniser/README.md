# 🚀 Advanced Image Recognizer

A comprehensive image recognition system that combines **YOLOv8** for product detection, **face_recognition** for face identification, and **CLIP** for AI-powered image analysis and text-based search.

## ✨ Features

- **🛍️ Product Detection**: YOLOv8-based object detection with custom training support
- **👤 Face Recognition**: Real-time face identification and registration
- **🤖 AI Overview**: CLIP-powered natural language image descriptions
- **🔍 Text Search**: Search images using natural language queries
- **🖥️ GUI Interface**: User-friendly desktop application
- **🌐 REST API**: FastAPI-based web service for programmatic access
- **📊 Multi-modal Analysis**: Combined analysis using all three systems

## 📁 Project Structure

```
image_recogniser/
├── data/
│   ├── train/                 # Training data
│   │   ├── faces/            # Face training images
│   │   └── [categories]/     # Product categories
│   ├── test/                 # Test data
│   └── yolo_dataset/         # Auto-generated YOLO dataset
├── models/
│   ├── face_encodings/       # Face recognition database
│   ├── yolo_products.pt      # Custom YOLO model
│   └── cnn_classifier.h5     # CNN classification model
├── scripts/
│   ├── product_detect.py     # YOLOv8 integration
│   ├── face_detect.py        # Face recognition
│   ├── clip_search.py        # CLIP integration
│   ├── runengine.py          # Unified GUI
│   ├── api_server.py         # FastAPI server
│   ├── train.py              # Training pipeline
│   ├── preprocess.py         # Image preprocessing
│   └── model.py              # CNN model architecture
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### 1. Environment Setup

```bash
# Navigate to project directory
cd image_recogniser

# Activate virtual environment (if using venv)
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Additional Setup (Optional)

#### For Face Recognition (Windows):
```bash
# Install Visual Studio Build Tools if not already installed
# Then install dlib and face_recognition
pip install cmake
pip install dlib
pip install face_recognition
```

#### For CUDA Support (GPU acceleration):
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start

### 1. Train Initial Models

```bash
cd scripts

# Train both CNN and YOLO models
python train.py --model both --epochs 50

# Or train specific models
python train.py --model cnn    # CNN only
python train.py --model yolo   # YOLO only
```

### 2. Run GUI Application

```bash
cd scripts
python runengine.py
```

### 3. Start API Server

```bash
cd scripts
python api_server.py
```

The API will be available at: `http://localhost:8000`

## 📖 Usage Guide

### GUI Application

1. **Upload Image**: Click "Upload Image" to select an image file
2. **Analyze**: Click "Analyze Image" to run all three recognition systems
3. **View Results**: See combined results in the text area:
   - Products detected with confidence scores
   - Faces recognized or unknown faces
   - AI-generated image overview
4. **Register Faces**: When unknown faces are detected, enter the person's name
5. **Retrain**: Click "Retrain Models" to update models with new data

### API Endpoints

#### Product Detection
```bash
curl -X POST "http://localhost:8000/detect_products" \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```

#### Face Recognition
```bash
curl -X POST "http://localhost:8000/recognize_face" \
  -F "file=@image.jpg" \
  -F "tolerance=0.6"
```

#### Image Overview
```bash
curl -X POST "http://localhost:8000/overview" \
  -F "file=@image.jpg"
```

#### Combined Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@image.jpg"
```

#### Text Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "person with laptop", "image_paths": ["path1.jpg", "path2.jpg"]}'
```

### Face Management

#### Register New Face
```bash
curl -X POST "http://localhost:8000/register_face" \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "image_path": "/path/to/face.jpg"}'
```

#### Get Known Faces
```bash
curl "http://localhost:8000/known_faces"
```

#### Remove Face
```bash
curl -X DELETE "http://localhost:8000/faces/John%20Doe"
```

## 🎯 Training Custom Models

### YOLO Training

1. **Prepare Dataset**: Organize images in `data/train/[category]/` folders
2. **Run Training**: `python train.py --model yolo --epochs 100`
3. **Custom Dataset**: The system automatically converts your data to YOLO format

### Face Recognition Training

1. **Add Face Images**: Place images in `data/train/faces/[person_name]/`
2. **Register Faces**: Use the GUI or API to register faces
3. **Auto-retrain**: The system automatically updates when new faces are added

### CNN Training

1. **Organize Data**: Place images in category folders under `data/train/`
2. **Run Training**: `python train.py --model cnn`
3. **Data Augmentation**: Automatic augmentation is applied during training

## 🔧 Configuration

### Model Parameters

#### YOLO Detection
- **Confidence Threshold**: Adjust detection sensitivity (default: 0.5)
- **Model Size**: Choose between nano, small, medium, large models
- **Input Size**: Configure image input size (default: 640x640)

#### Face Recognition
- **Tolerance**: Adjust recognition strictness (default: 0.6)
- **Encoding Method**: Uses HOG-based face detection
- **Database**: Face encodings stored in `models/face_encodings/`

#### CLIP Analysis
- **Model**: Uses `openai/clip-vit-base-patch32`
- **Custom Phrases**: Provide custom descriptive phrases for analysis
- **Similarity Threshold**: Minimum similarity for text-image matching

### Performance Optimization

#### GPU Acceleration
```python
# Enable CUDA for PyTorch models
device = "cuda" if torch.cuda.is_available() else "cpu"
```

#### Memory Management
- Use smaller batch sizes for limited memory
- Enable model quantization for deployment
- Use CPU-only versions for lightweight deployment

## 📊 Supported Formats

### Input Images
- **Formats**: PNG, JPG, JPEG, BMP, GIF, TIFF
- **Size**: Any size (automatically resized)
- **Color**: RGB, Grayscale (converted to RGB)

### Output Formats
- **JSON**: API responses
- **Images**: Annotated detection results
- **Text**: Natural language descriptions

## 🐛 Troubleshooting

### Common Issues

#### 1. Face Recognition Not Working
```bash
# Install required dependencies
pip install cmake dlib face_recognition
```

#### 2. CUDA Out of Memory
```python
# Reduce batch size in training
batch_size = 8  # Instead of 16
```

#### 3. Model Loading Errors
```bash
# Check model paths
ls models/
# Re-train models if missing
python train.py --model both
```

#### 4. API Server Issues
```bash
# Check port availability
netstat -an | findstr :8000
# Use different port
uvicorn api_server:app --port 8001
```

### Performance Tips

1. **Use GPU**: Enable CUDA for faster processing
2. **Batch Processing**: Process multiple images together
3. **Model Optimization**: Use smaller models for faster inference
4. **Caching**: Cache model predictions for repeated images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For YOLOv8 implementation
- **OpenAI**: For CLIP model
- **Face Recognition**: For face_recognition library
- **Hugging Face**: For transformers library
- **FastAPI**: For web framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `http://localhost:8000/docs`
3. Open an issue on GitHub
4. Check the logs for detailed error messages

---

**Happy Image Recognition! 🎉**#   i m a g e _ r e c o g n i s e r  
 