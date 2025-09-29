import os
import json
import requests
import base64
from typing import Dict, List, Optional

class FreeVisionAPI:
    """
    Completely FREE Image Recognition using Hugging Face Inference API
    No credit card required!
    """
    
    def __init__(self):
        self.hf_api_url = "https://api-inference.huggingface.co/models"
        self.api_key = self.get_hf_api_key()
        
        # FREE models for image recognition
        self.models = {
            "object_detection": "facebook/detr-resnet-50",
            "image_classification": "google/vit-base-patch16-224",
            "face_detection": "microsoft/table-transformer-detection",
            "text_detection": "microsoft/table-transformer-structure-recognition"
        }
        
        if self.api_key:
            print("✅ FREE Vision API initialized (Hugging Face)")
        else:
            print("⚠️ Hugging Face API key not found")
    
    def get_hf_api_key(self) -> Optional[str]:
        """Get Hugging Face API key"""
        try:
            # Try environment variable first
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            
            if not api_key:
                # Try to read from file
                if os.path.exists('../models/hf_api_key.txt'):
                    with open('../models/hf_api_key.txt', 'r') as f:
                        api_key = f.read().strip()
            
            return api_key
        except:
            return None
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            return None
    
    def detect_objects_free(self, image_path: str) -> Dict:
        """Detect objects using FREE Hugging Face API"""
        try:
            if not self.api_key:
                return {"error": "API key not found"}
            
            # Encode image
            image_base64 = self.encode_image(image_path)
            if not image_base64:
                return {"error": "Failed to encode image"}
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Object Detection
            payload = {
                "inputs": image_base64
            }
            
            model_url = f"{self.hf_api_url}/{self.models['object_detection']}"
            response = requests.post(model_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return self.parse_hf_results(result)
            else:
                print(f"❌ Hugging Face API error: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Hugging Face API error: {e}")
            return {"error": str(e)}
    
    def classify_image_free(self, image_path: str) -> Dict:
        """Classify image using FREE Hugging Face API"""
        try:
            if not self.api_key:
                return {"error": "API key not found"}
            
            # Encode image
            image_base64 = self.encode_image(image_path)
            if not image_base64:
                return {"error": "Failed to encode image"}
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Image Classification
            payload = {
                "inputs": image_base64
            }
            
            model_url = f"{self.hf_api_url}/{self.models['image_classification']}"
            response = requests.post(model_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return self.parse_classification_results(result)
            else:
                print(f"❌ Hugging Face API error: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Hugging Face API error: {e}")
            return {"error": str(e)}
    
    def parse_hf_results(self, result: List) -> Dict:
        """Parse Hugging Face object detection results"""
        try:
            parsed_result = {
                "objects": [],
                "labels": [],
                "faces": [],
                "text": [],
                "summary": ""
            }
            
            if isinstance(result, list):
                for item in result:
                    if "label" in item and "score" in item:
                        parsed_result["objects"].append({
                            "name": item["label"],
                            "confidence": item["score"],
                            "box": item.get("box", {})
                        })
            
            # Generate summary
            if parsed_result["objects"]:
                obj_names = [obj["name"] for obj in parsed_result["objects"][:5]]
                parsed_result["summary"] = f"Detected objects: {', '.join(obj_names)}"
            else:
                parsed_result["summary"] = "No objects detected"
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Error parsing HF results: {e}")
            return {"error": str(e)}
    
    def parse_classification_results(self, result: List) -> Dict:
        """Parse Hugging Face classification results"""
        try:
            parsed_result = {
                "labels": [],
                "summary": ""
            }
            
            if isinstance(result, list):
                for item in result:
                    if "label" in item and "score" in item:
                        parsed_result["labels"].append({
                            "description": item["label"],
                            "confidence": item["score"]
                        })
            
            # Generate summary
            if parsed_result["labels"]:
                top_labels = [label["description"] for label in parsed_result["labels"][:3]]
                parsed_result["summary"] = f"Image contains: {', '.join(top_labels)}"
            else:
                parsed_result["summary"] = "No classification available"
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Error parsing classification results: {e}")
            return {"error": str(e)}
    
    def get_comprehensive_analysis(self, image_path: str) -> Dict:
        """Get comprehensive image analysis using FREE APIs"""
        try:
            analysis = {
                "objects": {},
                "classification": {},
                "summary": "",
                "api_used": "Hugging Face (FREE)"
            }
            
            # Object Detection
            print("🔍 Detecting objects...")
            objects_result = self.detect_objects_free(image_path)
            analysis["objects"] = objects_result
            
            # Image Classification
            print("🏷️ Classifying image...")
            classification_result = self.classify_image_free(image_path)
            analysis["classification"] = classification_result
            
            # Generate overall summary
            summary_parts = []
            
            if "summary" in objects_result:
                summary_parts.append(objects_result["summary"])
            
            if "summary" in classification_result:
                summary_parts.append(classification_result["summary"])
            
            analysis["summary"] = " | ".join(summary_parts)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            return {"error": str(e)}

# Alternative FREE APIs
class ReplicateFreeAPI:
    """
    Replicate API - FREE tier (1000 requests/month)
    No credit card required for free tier
    """
    
    def __init__(self):
        self.api_url = "https://api.replicate.com/v1/predictions"
        self.api_key = self.get_replicate_api_key()
        
        if self.api_key:
            print("✅ Replicate FREE API initialized")
        else:
            print("⚠️ Replicate API key not found")
    
    def get_replicate_api_key(self) -> Optional[str]:
        """Get Replicate API key"""
        try:
            if os.path.exists('../models/replicate_api_key.txt'):
                with open('../models/replicate_api_key.txt', 'r') as f:
                    return f.read().strip()
        except:
            pass
        return None
    
    def detect_objects_replicate(self, image_url: str) -> Dict:
        """Detect objects using Replicate FREE API"""
        try:
            if not self.api_key:
                return {"error": "API key not found"}
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "version": "yolov8-object-detection",
                "input": {
                    "image": image_url
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 201:
                result = response.json()
                return {"status": "processing", "id": result["id"]}
            else:
                print(f"❌ Replicate API error: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Replicate API error: {e}")
            return {"error": str(e)}

# Setup functions
def setup_free_apis():
    """Setup FREE APIs"""
    print("🆓 Setting up FREE Image Recognition APIs")
    print("No credit card required!")
    
    # Check existing API keys
    free_vision = FreeVisionAPI()
    replicate_api = ReplicateFreeAPI()
    
    if free_vision.api_key:
        print("✅ Hugging Face API key found - FREE object detection available")
    else:
        print("⚠️ Hugging Face API key not found")
        print("Get FREE API key from: https://huggingface.co/settings/tokens")
    
    if replicate_api.api_key:
        print("✅ Replicate API key found - FREE YOLOv8 detection available")
    else:
        print("⚠️ Replicate API key not found")
        print("Get FREE API key from: https://replicate.com/account/api-tokens")

if __name__ == "__main__":
    # Test the FREE Vision API
    setup_free_apis()
    
    # Test with an image
    free_vision = FreeVisionAPI()
    if free_vision.api_key:
        # Replace with your test image path
        test_image = "test_image.jpg"
        if os.path.exists(test_image):
            print("🔍 Testing FREE image recognition...")
            result = free_vision.get_comprehensive_analysis(test_image)
            print("Analysis Results:")
            print(json.dumps(result, indent=2))
        else:
            print("Test image not found. Please provide a valid image path.")
