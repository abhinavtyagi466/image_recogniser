#!/usr/bin/env python3
"""
Clean Gemini Engine - Simplified and Working
Focuses on what actually works: Local models + Gemini API
"""

import os
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime

class CleanGeminiEngine:
    """
    Clean Gemini Engine - Simple and Working
    Uses local analysis + Gemini API for responses
    """
    
    def __init__(self):
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.api_key = self.get_google_api_key()
        
        if self.api_key:
            print("🚀 Clean Gemini Engine initialized!")
        else:
            print("❌ Google API key not found")
    
    def get_google_api_key(self) -> Optional[str]:
        """Get Google API key"""
        try:
            if os.path.exists('../models/google_api_key.txt'):
                with open('../models/google_api_key.txt', 'r') as f:
                    return f.read().strip()
        except:
            pass
        return None
    
    def generate_response(self, user_message: str, image_context: Dict = None) -> str:
        """Generate response using Gemini API"""
        try:
            # Build context from image analysis
            context_text = ""
            if image_context:
                context_text = self.build_context_text(image_context)
            
            # Create prompt
            prompt = f"""
You are a helpful AI assistant for an image recognition system.

{context_text}

User message: {user_message}

Please provide a helpful, natural response based on the image analysis above.
Be conversational and friendly.
"""
            
            # Call Gemini API
            response = self.call_gemini_api(prompt)
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"I encountered an error: {e}"
    
    def build_context_text(self, image_context: Dict) -> str:
        """Build context text from image analysis"""
        try:
            context_parts = []
            
            # Add face information
            if "faces" in image_context:
                faces = image_context["faces"]
                if isinstance(faces, dict) and "faces" in faces:
                    face_list = faces["faces"]
                    if face_list:
                        face_names = [f.get("name", "Unknown") for f in face_list]
                        context_parts.append(f"Faces detected: {', '.join(face_names)}")
                    else:
                        context_parts.append("No faces detected")
            
            # Add object information
            if "products" in image_context:
                products = image_context["products"]
                if isinstance(products, dict) and "objects" in products:
                    objects = products["objects"]
                    if objects:
                        object_labels = [obj.get("label", "Unknown") for obj in objects]
                        context_parts.append(f"Objects detected: {', '.join(object_labels)}")
                    else:
                        context_parts.append("No objects detected")
            
            # Add overview information
            if "overview" in image_context:
                overview = image_context["overview"]
                if isinstance(overview, dict) and "description" in overview:
                    description = overview["description"]
                    if description:
                        context_parts.append(f"Image description: {description}")
            
            return "\n".join(context_parts) if context_parts else "No image analysis available"
            
        except Exception as e:
            print(f"❌ Error building context: {e}")
            return "Image analysis available but could not be processed"
    
    def call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API"""
        try:
            url = f"{self.base_url}/gemini-2.5-flash:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 100000,
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
                print(f"❌ Gemini API error: {response.status_code}")
                return f"API error: {response.status_code}"
                
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return f"I encountered an error: {e}"
    
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            "gemini_api_available": self.api_key is not None,
            "status": "ready" if self.api_key else "missing_api_key"
        }

# Test function
def test_clean_gemini():
    """Test the clean Gemini engine"""
    try:
        print("🧪 Testing Clean Gemini Engine...")
        
        engine = CleanGeminiEngine()
        status = engine.get_status()
        print(f"Status: {status}")
        
        if not status["gemini_api_available"]:
            print("❌ Gemini API not available")
            return False
        
        # Test with sample context
        test_context = {
            "faces": {"faces": [{"name": "John", "confidence": 0.95}]},
            "products": {"objects": [{"label": "laptop", "confidence": 0.89}]},
            "overview": {"description": "Person at desk with laptop"}
        }
        
        response = engine.generate_response("What do you see in this image?", test_context)
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_clean_gemini()
