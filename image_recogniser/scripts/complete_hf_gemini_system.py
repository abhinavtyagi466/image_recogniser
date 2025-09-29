#!/usr/bin/env python3
"""
Complete Hugging Face + Gemini Integration System
Handles both HF Vision and Gemini API for superior results
"""

import os
import requests
import json
from typing import Dict, Optional
from datetime import datetime

class CompleteHFGeminiSystem:
    """
    Complete system integrating Hugging Face Vision + Gemini API
    """
    
    def __init__(self):
        self.hf_api_url = "https://api-inference.huggingface.co/models"
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        # Get API keys
        self.hf_api_key = self.get_hf_api_key()
        self.gemini_api_key = self.get_gemini_api_key()
        
        # Working HF models
        self.hf_models = {
            "object_detection": "facebook/detr-resnet-50",
            "image_classification": "google/vit-base-patch16-224",
            "image_captioning": "Salesforce/blip-image-captioning-base"
        }
        
        # Status
        self.hf_available = self.validate_hf_api()
        self.gemini_available = self.gemini_api_key is not None
        
        print("🚀 Complete HF + Gemini System initialized!")
        print(f"✅ Hugging Face: {'Available' if self.hf_available else 'Not Available'}")
        print(f"✅ Gemini API: {'Available' if self.gemini_available else 'Not Available'}")
    
    def get_hf_api_key(self) -> Optional[str]:
        """Get Hugging Face API key"""
        try:
            if os.path.exists('../models/hf_api_key.txt'):
                with open('../models/hf_api_key.txt', 'r') as f:
                    return f.read().strip()
        except:
            pass
        return None
    
    def get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key"""
        try:
            if os.path.exists('../models/google_api_key.txt'):
                with open('../models/google_api_key.txt', 'r') as f:
                    return f.read().strip()
        except:
            pass
        return None
    
    def validate_hf_api(self) -> bool:
        """Validate Hugging Face API key"""
        if not self.hf_api_key:
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.hf_api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.hf_api_url}/gpt2"
            payload = {"inputs": "test"}
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            return response.status_code == 200
            
        except:
            return False
    
    def analyze_with_hf(self, image_path: str) -> Dict:
        """Analyze image using Hugging Face models"""
        if not self.hf_available:
            return {"error": "Hugging Face API not available"}
        
        try:
            print("🌐 Running HF Vision analysis...")
            
            analysis = {
                "objects": {"error": "Not analyzed"},
                "classification": {"error": "Not analyzed"},
                "caption": {"error": "Not analyzed"}
            }
            
            # Object Detection
            try:
                print("🔍 Detecting objects...")
                objects_result = self.call_hf_model(self.hf_models["object_detection"], image_path)
                if objects_result.get("success"):
                    analysis["objects"] = {
                        "success": True,
                        "objects": self.parse_objects(objects_result["data"]),
                        "summary": f"Detected {len(self.parse_objects(objects_result['data']))} objects"
                    }
            except Exception as e:
                analysis["objects"] = {"error": str(e)}
            
            # Image Classification
            try:
                print("🏷️ Classifying image...")
                classification_result = self.call_hf_model(self.hf_models["image_classification"], image_path)
                if classification_result.get("success"):
                    classifications = self.parse_classifications(classification_result["data"])
                    analysis["classification"] = {
                        "success": True,
                        "classifications": classifications,
                        "summary": f"Image classified as: {classifications[0]['label']}" if classifications else "Classification failed"
                    }
            except Exception as e:
                analysis["classification"] = {"error": str(e)}
            
            # Image Captioning
            try:
                print("📝 Generating caption...")
                caption_result = self.call_hf_model(self.hf_models["image_captioning"], image_path)
                if caption_result.get("success"):
                    caption = self.parse_caption(caption_result["data"])
                    analysis["caption"] = {
                        "success": True,
                        "caption": caption,
                        "summary": f"Caption: {caption}"
                    }
            except Exception as e:
                analysis["caption"] = {"error": str(e)}
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def call_hf_model(self, model_name: str, image_path: str) -> Dict:
        """Call Hugging Face model"""
        try:
            headers = {
                "Authorization": f"Bearer {self.hf_api_key}"
            }
            
            url = f"{self.hf_api_url}/{model_name}"
            
            with open(image_path, "rb") as image_file:
                response = requests.post(url, headers=headers, data=image_file, timeout=30)
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            elif response.status_code == 503:
                return {"error": "Model loading, please wait"}
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def parse_objects(self, data: list) -> list:
        """Parse object detection results"""
        try:
            objects = []
            if isinstance(data, list):
                for item in data:
                    if "label" in item and "score" in item:
                        objects.append({
                            "label": item["label"],
                            "confidence": item["score"],
                            "bbox": item.get("box", {})
                        })
            return objects
        except:
            return []
    
    def parse_classifications(self, data: list) -> list:
        """Parse classification results"""
        try:
            classifications = []
            if isinstance(data, list):
                for item in data:
                    if "label" in item and "score" in item:
                        classifications.append({
                            "label": item["label"],
                            "confidence": item["score"]
                        })
            return classifications
        except:
            return []
    
    def parse_caption(self, data: list) -> str:
        """Parse caption results"""
        try:
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
            return ""
        except:
            return ""
    
    def generate_gemini_response(self, user_message: str, hf_analysis: Dict, local_analysis: Dict = None) -> str:
        """Generate response using Gemini API with HF analysis"""
        if not self.gemini_available:
            return "Gemini API not available"
        
        try:
            # Build context from HF analysis
            context_parts = []
            
            if hf_analysis.get("objects", {}).get("success"):
                objects = hf_analysis["objects"]["objects"]
                if objects:
                    obj_labels = [obj["label"] for obj in objects[:5]]
                    context_parts.append(f"HF Objects detected: {', '.join(obj_labels)}")
            
            if hf_analysis.get("classification", {}).get("success"):
                classifications = hf_analysis["classification"]["classifications"]
                if classifications:
                    top_label = classifications[0]["label"]
                    context_parts.append(f"HF Classification: {top_label}")
            
            if hf_analysis.get("caption", {}).get("success"):
                caption = hf_analysis["caption"]["caption"]
                if caption:
                    context_parts.append(f"HF Caption: {caption}")
            
            # Add local analysis if available
            if local_analysis:
                if "faces" in local_analysis:
                    faces = local_analysis["faces"]
                    if isinstance(faces, dict) and "faces" in faces:
                        face_list = faces["faces"]
                        if face_list:
                            face_names = [f.get("name", "Unknown") for f in face_list]
                            context_parts.append(f"Local Faces: {', '.join(face_names)}")
                
                if "products" in local_analysis:
                    products = local_analysis["products"]
                    if isinstance(products, dict) and "objects" in products:
                        objects = products["objects"]
                        if objects:
                            obj_labels = [obj.get("label", "Unknown") for obj in objects]
                            context_parts.append(f"Local Objects: {', '.join(obj_labels)}")
            
            # Build prompt
            context_text = "\n".join(context_parts) if context_parts else "No analysis available"
            
            prompt = f"""
You are a helpful AI assistant analyzing an image with both cloud-based and local analysis.

ANALYSIS RESULTS:
{context_text}

User question: {user_message}

Please provide a comprehensive, natural response based on the analysis above.
Be conversational and helpful.
"""
            
            # Call Gemini API
            url = f"{self.gemini_url}/gemini-2.5-flash:generateContent?key={self.gemini_api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1000,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        return candidate["content"]["parts"][0]["text"].strip()
                    elif "finishReason" in candidate:
                        if candidate["finishReason"] == "MAX_TOKENS":
                            return "I can see the image and understand your question, but my response was cut short. Could you ask a more specific question?"
                        else:
                            return "I'm having trouble processing the response, but I'm here to help!"
                    else:
                        return "I'm having trouble processing the response, but I'm here to help!"
                else:
                    return "I couldn't generate a response. Please try again."
            else:
                return f"Gemini API error: {response.status_code}"
                
        except Exception as e:
            return f"I encountered an error: {e}"
    
    def analyze_image_complete(self, image_path: str, user_message: str, local_analysis: Dict = None) -> Dict:
        """Complete image analysis with HF + Gemini"""
        try:
            print("🚀 Starting complete analysis...")
            
            # Step 1: HF Analysis (if available)
            hf_analysis = {}
            if self.hf_available:
                hf_analysis = self.analyze_with_hf(image_path)
            else:
                print("⚠️ Hugging Face not available - using local analysis only")
            
            # Step 2: Generate Gemini response
            if self.gemini_available:
                response = self.generate_gemini_response(user_message, hf_analysis, local_analysis)
            else:
                response = "Gemini API not available"
            
            # Step 3: Combine results
            result = {
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "hf_analysis": hf_analysis,
                "local_analysis": local_analysis,
                "gemini_response": response,
                "systems_used": {
                    "hugging_face": self.hf_available,
                    "gemini": self.gemini_available,
                    "local_models": local_analysis is not None
                }
            }
            
            print("✅ Complete analysis finished!")
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict:
        """Get complete system status"""
        return {
            "hugging_face_available": self.hf_available,
            "gemini_available": self.gemini_available,
            "hf_api_key_valid": self.hf_api_key is not None,
            "gemini_api_key_valid": self.gemini_api_key is not None,
            "models_available": len(self.hf_models),
            "status": "ready" if (self.hf_available or self.gemini_available) else "needs_api_keys"
        }

# Test function
def test_complete_system():
    """Test the complete system"""
    try:
        print("🧪 Testing Complete HF + Gemini System...")
        
        system = CompleteHFGeminiSystem()
        status = system.get_system_status()
        print(f"System Status: {status}")
        
        if not status["gemini_available"]:
            print("❌ Gemini API not available")
            return False
        
        # Test with sample data
        test_image = "../data/test/bhains/bhains.png"
        test_local_analysis = {
            "faces": {"faces": [{"name": "John", "confidence": 0.95}]},
            "products": {"objects": [{"label": "laptop", "confidence": 0.89}]}
        }
        
        if os.path.exists(test_image):
            result = system.analyze_image_complete(test_image, "What do you see in this image?", test_local_analysis)
            print(f"Result: {json.dumps(result, indent=2)}")
            return True
        else:
            print("❌ Test image not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_complete_system()
