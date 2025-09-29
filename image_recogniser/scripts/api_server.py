import os
import json
import base64
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io

# Import our custom modules
from product_detect import ProductDetector
from face_detect_fallback import FaceRecognizer
from clip_search import CLIPSearch

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Image Recognizer API",
    description="API for YOLOv8 product detection, face recognition, and CLIP image analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detection modules
product_detector = ProductDetector()
face_recognizer = FaceRecognizer()
clip_search = CLIPSearch()

# Pydantic models for request/response
class ProductDetectionResponse(BaseModel):
    objects: List[Dict]
    total_detections: int
    model_used: str

class FaceRecognitionResponse(BaseModel):
    faces: List[Dict]
    total_faces: int

class ImageOverviewResponse(BaseModel):
    description: str
    top_matches: List[List]
    analysis_phrases: int

class TextSearchRequest(BaseModel):
    query: str
    image_paths: List[str]

class TextSearchResponse(BaseModel):
    query: str
    results: List[Dict]
    total_images: int

class FaceRegistrationRequest(BaseModel):
    name: str
    image_path: str

class CombinedAnalysisResponse(BaseModel):
    products: ProductDetectionResponse
    faces: FaceRecognitionResponse
    overview: ImageOverviewResponse
    image_path: str

def save_uploaded_file(upload_file: UploadFile, upload_dir: str = "uploads") -> str:
    """Save uploaded file and return the path"""
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    file_extension = os.path.splitext(upload_file.filename)[1]
    import uuid
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        content = upload_file.file.read()
        buffer.write(content)
    
    return file_path

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced Image Recognizer API",
        "version": "1.0.0",
        "endpoints": {
            "detect_products": "/detect_products",
            "recognize_face": "/recognize_face",
            "overview": "/overview",
            "search": "/search",
            "register_face": "/register_face",
            "combined_analysis": "/analyze",
            "known_faces": "/known_faces",
            "supported_products": "/supported_products"
        }
    }

@app.post("/detect_products", response_model=ProductDetectionResponse)
async def detect_products(
    file: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    """
    Detect products in an uploaded image using YOLOv8
    
    Args:
        file: Image file to analyze
        confidence: Confidence threshold for detections (0.0-1.0)
    
    Returns:
        ProductDetectionResponse with detected objects
    """
    try:
        # Save uploaded file
        image_path = save_uploaded_file(file)
        
        # Detect products
        results = product_detector.detect_products(image_path, confidence)
        
        # Clean up uploaded file
        os.remove(image_path)
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return ProductDetectionResponse(
            objects=results.get("objects", []),
            total_detections=results.get("total_detections", 0),
            model_used=results.get("model_used", "unknown")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize_face", response_model=FaceRecognitionResponse)
async def recognize_face(
    file: UploadFile = File(...),
    tolerance: float = Form(0.6)
):
    """
    Recognize faces in an uploaded image
    
    Args:
        file: Image file to analyze
        tolerance: Face recognition tolerance (lower = more strict)
    
    Returns:
        FaceRecognitionResponse with recognized faces
    """
    try:
        # Save uploaded file
        image_path = save_uploaded_file(file)
        
        # Recognize faces
        results = face_recognizer.recognize_face(image_path, tolerance)
        
        # Clean up uploaded file
        os.remove(image_path)
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return FaceRecognitionResponse(
            faces=results.get("faces", []),
            total_faces=results.get("total_faces", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/overview", response_model=ImageOverviewResponse)
async def get_image_overview(
    file: UploadFile = File(...),
    custom_phrases: Optional[str] = Form(None)
):
    """
    Generate AI overview of an uploaded image using CLIP
    
    Args:
        file: Image file to analyze
        custom_phrases: Optional comma-separated custom phrases for analysis
    
    Returns:
        ImageOverviewResponse with image description
    """
    try:
        # Save uploaded file
        image_path = save_uploaded_file(file)
        
        # Parse custom phrases if provided
        phrases = None
        if custom_phrases:
            phrases = [phrase.strip() for phrase in custom_phrases.split(",")]
        
        # Generate overview
        results = clip_search.get_image_overview(image_path, phrases)
        
        # Clean up uploaded file
        os.remove(image_path)
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return ImageOverviewResponse(
            description=results.get("description", "No description available"),
            top_matches=results.get("top_matches", []),
            analysis_phrases=results.get("analysis_phrases", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=TextSearchResponse)
async def search_with_text(request: TextSearchRequest):
    """
    Search images using text query
    
    Args:
        request: TextSearchRequest with query and image paths
    
    Returns:
        TextSearchResponse with similarity scores
    """
    try:
        results = clip_search.search_with_text(request.query, request.image_paths)
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return TextSearchResponse(
            query=results.get("query", ""),
            results=results.get("results", []),
            total_images=results.get("total_images", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register_face")
async def register_face(request: FaceRegistrationRequest):
    """
    Register a new face in the database
    
    Args:
        request: FaceRegistrationRequest with name and image path
    
    Returns:
        Success message
    """
    try:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        success = face_recognizer.register_face(request.name, request.image_path)
        
        if success:
            return {"message": f"Successfully registered face for {request.name}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to register face")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=CombinedAnalysisResponse)
async def combined_analysis(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    tolerance: float = Form(0.6)
):
    """
    Perform combined analysis using all three systems
    
    Args:
        file: Image file to analyze
        confidence: Product detection confidence threshold
        tolerance: Face recognition tolerance
    
    Returns:
        CombinedAnalysisResponse with all analysis results
    """
    try:
        # Save uploaded file
        image_path = save_uploaded_file(file)
        
        # Run all analyses
        product_results = product_detector.detect_products(image_path, confidence)
        face_results = face_recognizer.recognize_face(image_path, tolerance)
        overview_results = clip_search.get_image_overview(image_path)
        
        # Clean up uploaded file
        os.remove(image_path)
        
        # Check for errors
        if "error" in product_results:
            raise HTTPException(status_code=500, detail=f"Product detection error: {product_results['error']}")
        if "error" in face_results:
            raise HTTPException(status_code=500, detail=f"Face recognition error: {face_results['error']}")
        if "error" in overview_results:
            raise HTTPException(status_code=500, detail=f"Overview error: {overview_results['error']}")
        
        return CombinedAnalysisResponse(
            products=ProductDetectionResponse(
                objects=product_results.get("objects", []),
                total_detections=product_results.get("total_detections", 0),
                model_used=product_results.get("model_used", "unknown")
            ),
            faces=FaceRecognitionResponse(
                faces=face_results.get("faces", []),
                total_faces=face_results.get("total_faces", 0)
            ),
            overview=ImageOverviewResponse(
                description=overview_results.get("description", "No description available"),
                top_matches=overview_results.get("top_matches", []),
                analysis_phrases=overview_results.get("analysis_phrases", 0)
            ),
            image_path=image_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/known_faces")
async def get_known_faces():
    """Get list of known faces in the database"""
    try:
        faces = face_recognizer.get_known_faces()
        return {"known_faces": faces, "total_faces": len(faces)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported_products")
async def get_supported_products():
    """Get list of supported product categories"""
    try:
        products = product_detector.get_product_categories()
        return {"supported_products": products, "total_categories": len(products)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/faces/{face_name}")
async def remove_face(face_name: str):
    """Remove a face from the database"""
    try:
        success = face_recognizer.remove_face(face_name)
        if success:
            return {"message": f"Successfully removed face: {face_name}"}
        else:
            raise HTTPException(status_code=404, detail=f"Face '{face_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "modules": {
            "product_detector": product_detector.model is not None,
            "face_recognizer": len(face_recognizer.known_names) >= 0,
            "clip_search": clip_search.model is not None
        }
    }

def main():
    """Run the API server"""
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
