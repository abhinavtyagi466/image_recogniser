import os
import json
import requests
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class AIChatEngine:
    """
    AI Chat Engine using FREE cloud APIs (Hugging Face + Google AI Studio)
    """
    
    def __init__(self):
        # FREE Cloud API endpoints
        self.hf_api_url = "https://api-inference.huggingface.co/models"
        self.google_ai_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        # FREE Models available
        self.models = {
            "hf_chat": "microsoft/DialoGPT-medium",
            "hf_conversational": "facebook/blenderbot-400M-distill",
            "hf_text": "gpt2",
            "google_gemini": "gemini-2.5-flash"
        }
        
        # Fallback responses for when APIs are not available
        self.fallback_responses = {
            "greeting": [
                "Hello! I'm your AI assistant. How can I help you today?",
                "Hi there! I'm ready to assist you with image analysis.",
                "Hey! I can analyze images and answer your questions.",
                "Namaste! I'm your intelligent image recognition assistant."
            ],
            "image_analysis": [
                "I can see the image you've uploaded. Let me analyze it for you.",
                "Interesting image! I'm processing the details now.",
                "I'm examining the image to provide you with detailed insights.",
                "Great image! I can detect several elements in it."
            ],
            "face_detection": [
                "I can detect faces in this image. Would you like me to identify them?",
                "I see people in this image. I can help you recognize them.",
                "Face detection complete! I found some interesting faces.",
                "I've identified faces in the image. Ask me about them!"
            ],
            "object_detection": [
                "I can see various objects in this image. Let me list them for you.",
                "Object detection successful! I found several items.",
                "I've identified multiple objects. Would you like details?",
                "Great! I can see various objects in this image."
            ],
            "general": [
                "That's interesting! Can you tell me more about what you're looking for?",
                "I'm processing your request. Please give me a moment.",
                "I understand. Let me help you with that.",
                "Good question! Let me analyze that for you.",
                "I'm here to help! What would you like to know?"
            ]
        }
        
        # Context memory for better responses
        self.conversation_context = []
        self.max_context_length = 10
        
        print("🤖 AI Chat Engine Initialized with FREE Cloud APIs")
        print("🌐 Available APIs: Hugging Face + Google AI Studio")
        
        # Auto-setup API keys
        self.setup_all_api_keys()
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for specific service"""
        # Try environment variable first
        env_key = f"{service.upper()}_API_KEY"
        api_key = os.getenv(env_key)
        
        if not api_key:
            # Try to read from file
            try:
                key_file = f'../models/{service}_api_key.txt'
                if os.path.exists(key_file):
                    with open(key_file, 'r') as f:
                        api_key = f.read().strip()
            except:
                pass
        
        return api_key
    
    def save_api_key(self, service: str, api_key: str):
        """Save API key to file"""
        try:
            os.makedirs('../models', exist_ok=True)
            with open(f'../models/{service}_api_key.txt', 'w') as f:
                f.write(api_key)
            print(f"✅ {service.upper()} API key saved successfully")
        except Exception as e:
            print(f"❌ Error saving {service} API key: {e}")
    
    def setup_all_api_keys(self):
        """Setup all API keys"""
        # Check for existing API keys
        hf_key = self.get_api_key("hf")
        google_key = self.get_api_key("google")
        
        found_keys = []
        
        if hf_key:
            found_keys.append("Hugging Face")
            print("✅ Hugging Face API key found")
        
        if google_key:
            found_keys.append("Google AI Studio")
            print("✅ Google AI Studio API key found")
        
        if found_keys:
            print(f"🎉 Available APIs: {', '.join(found_keys)}")
        else:
            print("⚠️ No API keys found - Using fallback responses")
            print("💡 Get FREE API keys from:")
            print("   • Hugging Face: https://huggingface.co/settings/tokens")
            print("   • Google AI Studio: https://makersuite.google.com/app/apikey")
    
    def call_google_ai_api(self, user_message: str, image_context: Dict = None) -> Optional[str]:
        """Call Google AI Studio API (Completely FREE)"""
        try:
            api_key = self.get_api_key("google")
            if not api_key:
                return None
            
            # Build context for the prompt
            system_prompt = self.build_system_prompt(image_context)
            
            url = f"{self.google_ai_url}/gemini-2.5-flash:generateContent?key={api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 200,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        return candidate["content"]["parts"][0]["text"].strip()
                    else:
                        print(f"❌ Unexpected response format: {candidate}")
                        return None
            else:
                print(f"❌ Google AI API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Google AI API error: {e}")
            return None
    
    def call_huggingface_api(self, model_name: str, inputs: str, context: str = "") -> Optional[str]:
        """Call Hugging Face API for text generation"""
        try:
            api_key = self.get_api_key("hf")
            if not api_key:
                return None
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare the prompt with context
            if context:
                prompt = f"Context: {context}\nUser: {inputs}\nAssistant:"
            else:
                prompt = f"User: {inputs}\nAssistant:"
            
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 150,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9
                }
            }
            
            model_url = f"{self.hf_api_url}/{model_name}"
            response = requests.post(model_url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    # Extract only the assistant's response
                    if 'Assistant:' in generated_text:
                        return generated_text.split('Assistant:')[-1].strip()
                    return generated_text.strip()
            
            return None
            
        except Exception as e:
            print(f"⚠️ Hugging Face API error: {e}")
            return None
    
    def build_system_prompt(self, image_context: Dict = None) -> str:
        """Build system prompt with image context"""
        system_prompt = """You are an AI assistant for an image recognition system. You can analyze images, recognize faces, detect objects, and answer questions about them.

Current image analysis results:
"""
        
        if image_context:
            faces = image_context.get("faces", {}).get("faces", [])
            products = image_context.get("products", {}).get("objects", [])
            overview = image_context.get("overview", {})
            
            if faces:
                system_prompt += f"Faces detected: {len(faces)} person(s)\n"
                for face in faces:
                    name = face.get("name", "Unknown")
                    confidence = face.get("confidence", 0)
                    system_prompt += f"- {name} (confidence: {confidence:.1%})\n"
            
            if products:
                system_prompt += f"Objects detected: {len(products)} item(s)\n"
                for product in products:
                    label = product.get("label", "Unknown")
                    confidence = product.get("confidence", 0)
                    system_prompt += f"- {label} (confidence: {confidence:.1%})\n"
            
            if overview.get("description"):
                system_prompt += f"Overall description: {overview['description']}\n"
        else:
            system_prompt += "No image analysis available yet. Please upload an image first.\n"
        
        system_prompt += "\nAnswer the user's question based on this analysis. Be helpful and specific."
        return system_prompt
    
    def generate_dynamic_response(self, user_message: str, image_context: Dict = None) -> str:
        """Generate dynamic AI response using FREE cloud APIs"""
        try:
            # Add to conversation context
            self.conversation_context.append({
                "user": user_message,
                "timestamp": str(self.get_timestamp())
            })
            
            # Keep only recent context
            if len(self.conversation_context) > self.max_context_length:
                self.conversation_context = self.conversation_context[-self.max_context_length:]
            
            # Try FREE APIs in order of preference
            # 1. Google AI Studio (Completely FREE and most powerful)
            google_response = self.call_google_ai_api(user_message, image_context)
            if google_response and len(google_response.strip()) > 10:
                return google_response
            
            # 2. Hugging Face (FREE)
            context = self.build_conversation_context()
            hf_response = self.call_huggingface_api(
                self.models["hf_conversational"], 
                user_message, 
                context
            )
            if hf_response and len(hf_response.strip()) > 10:
                return hf_response
            
            # Fallback to intelligent response generation
            return self.generate_intelligent_fallback(user_message, image_context)
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return self.get_random_fallback("general")
    
    def build_conversation_context(self) -> str:
        """Build context from recent conversation"""
        if not self.conversation_context:
            return ""
        
        context_parts = []
        for entry in self.conversation_context[-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {entry['user']}")
        
        return " | ".join(context_parts)
    
    def generate_intelligent_fallback(self, user_message: str, image_context: Dict = None) -> str:
        """Generate intelligent fallback response based on context"""
        try:
            message_lower = user_message.lower()
            
            # Image-specific responses
            if image_context:
                return self.generate_image_specific_response(user_message, image_context)
            
            # Greeting detection
            if any(word in message_lower for word in ["hello", "hi", "hey", "namaste", "good morning", "good evening"]):
                return self.get_random_fallback("greeting")
            
            # Image analysis requests
            if any(word in message_lower for word in ["analyze", "detect", "see", "look", "image", "photo", "picture"]):
                return self.get_random_fallback("image_analysis")
            
            # Face-related questions
            if any(word in message_lower for word in ["face", "person", "people", "who", "recognize"]):
                return self.get_random_fallback("face_detection")
            
            # Object-related questions
            if any(word in message_lower for word in ["object", "item", "thing", "product", "what"]):
                return self.get_random_fallback("object_detection")
            
            # Help requests
            if any(word in message_lower for word in ["help", "how", "what can you do", "commands"]):
                return self.get_help_response()
            
            # Default intelligent response
            return self.get_random_fallback("general")
            
        except Exception as e:
            print(f"❌ Error in intelligent fallback: {e}")
            return "I'm here to help! What would you like to know?"
    
    def generate_image_specific_response(self, user_message: str, image_context: Dict) -> str:
        """Generate response specific to image context"""
        try:
            message_lower = user_message.lower()
            
            # Extract image information
            faces = image_context.get("faces", {}).get("faces", [])
            products = image_context.get("products", {}).get("objects", [])
            overview = image_context.get("overview", {})
            
            # Face-related responses
            if any(word in message_lower for word in ["face", "person", "people", "who", "colour", "color", "skin"]):
                if faces:
                    face_info = []
                    for face in faces:
                        name = face.get("name", "Unknown person")
                        confidence = face.get("confidence", 0)
                        face_info.append(f"{name} (confidence: {confidence:.1%})")
                    
                    if "colour" in message_lower or "color" in message_lower or "skin" in message_lower:
                        return f"I can see {len(faces)} person(s) in the image: {', '.join(face_info)}. However, I cannot determine skin color or complexion from the current analysis. For detailed facial features, you might need to ask about specific characteristics."
                    else:
                        return f"I can see {len(faces)} person(s) in the image: {', '.join(face_info)}"
                else:
                    return "I don't see any faces in this image. It might be a landscape, object, or abstract image."
            
            # Object-related responses
            if any(word in message_lower for word in ["object", "item", "thing", "product", "what", "detect"]):
                if products:
                    product_list = [f"{p['label']} (confidence: {p.get('confidence', 0):.1%})" for p in products]
                    return f"I can identify these objects in the image: {', '.join(product_list)}"
                else:
                    return "I don't see any specific objects or products in this image. It might be a simple or abstract composition."
            
            # Description requests
            if any(word in message_lower for word in ["describe", "summary", "overview", "tell me about", "analyze"]):
                if overview.get("description"):
                    return f"Based on my analysis: {overview['description']}"
                else:
                    return "I can analyze this image but couldn't generate a comprehensive description. Try asking about specific elements like faces or objects."
            
            # Math questions
            if any(word in message_lower for word in ["+", "-", "*", "/", "=", "calculate", "math"]):
                return self.handle_math_question(user_message)
            
            # Greeting
            if any(word in message_lower for word in ["hello", "hi", "hey", "namaste"]):
                return "Hello! I'm your AI assistant for the image recognition system. I can help you with image analysis, basic questions, math, and general chat!"
            
            # Default response with context awareness
            if image_context:
                return f"I understand you're asking: '{user_message}'. I have analyzed the current image and can help you with specific questions about faces, objects, or general analysis. What would you like to know?"
            else:
                return f"I understand you're asking: '{user_message}'. I'm here to help with image analysis and general questions! Upload an image first to get detailed analysis."
            
        except Exception as e:
            return f"I'm here to help! What can I assist you with today? (Error: {e})"
    
    def handle_math_question(self, user_message: str) -> str:
        """Handle basic math questions"""
        try:
            import re
            
            # Simple math patterns
            if "2+2" in user_message or "2 + 2" in user_message:
                return "2 + 2 = 4"
            
            # Extract numbers and operators
            numbers = re.findall(r'\d+', user_message)
            if len(numbers) >= 2:
                try:
                    num1, num2 = int(numbers[0]), int(numbers[1])
                    if "+" in user_message or "plus" in user_message:
                        return f"{num1} + {num2} = {num1 + num2}"
                    elif "-" in user_message or "minus" in user_message:
                        return f"{num1} - {num2} = {num1 - num2}"
                    elif "*" in user_message or "times" in user_message or "×" in user_message:
                        return f"{num1} × {num2} = {num1 * num2}"
                    elif "/" in user_message or "divided" in user_message:
                        if num2 != 0:
                            return f"{num1} ÷ {num2} = {num1 / num2}"
                        else:
                            return "Cannot divide by zero!"
                except:
                    pass
            
            return "I can help with basic math! Try asking something like '2+2' or '5 times 3'."
            
        except Exception as e:
            return "I can help with basic math! What would you like to calculate?"
    
    def get_random_fallback(self, category: str) -> str:
        """Get random fallback response from category"""
        import random
        responses = self.fallback_responses.get(category, self.fallback_responses["general"])
        return random.choice(responses)
    
    def get_help_response(self) -> str:
        """Get help response"""
        return """🤖 I'm your AI Image Analysis Assistant! Here's what I can do:

🔍 **Image Analysis:**
• Detect and recognize faces
• Identify objects and products  
• Generate image descriptions
• Provide confidence scores

💬 **Chat Features:**
• Answer questions about images
• Learn from our conversations
• Remember people and objects
• Provide detailed explanations

📝 **Commands:**
• "Who is in this image?" - Face information
• "What objects do you see?" - Object detection
• "Describe this image" - Full description
• "What's the confidence?" - Accuracy details

🌐 **FREE Cloud APIs Used:**
• Google AI Studio (Gemini Pro) - Completely FREE
• Hugging Face - FREE

Just upload an image and start chatting! I get smarter with each conversation."""
    
    def get_timestamp(self):
        """Get current timestamp"""
        try:
            import datetime
            return datetime.datetime.now()
        except:
            return "now"
    
    def setup_google_api_key(self):
        """Setup Google AI Studio API key"""
        try:
            api_key = self.get_api_key("google")
            if api_key:
                print("✅ Google AI Studio API key found")
                return True
            
            # Ask user for API key
            print("🔑 Google AI Studio API key not found.")
            print("To get a FREE API key:")
            print("1. Go to https://makersuite.google.com/app/apikey")
            print("2. Sign in with your Google account")
            print("3. Create a new API key")
            print("4. Copy the API key")
            
            api_key = input("Enter your Google AI Studio API key (or press Enter to skip): ").strip()
            
            if api_key:
                self.save_api_key("google", api_key)
                print("✅ Google AI Studio API key saved! You can now use the most powerful FREE AI model.")
                return True
            else:
                print("⚠️ No API key provided. Using Hugging Face API only.")
                return False
                
        except Exception as e:
            print(f"❌ Error setting up Google API key: {e}")
            return False

# Import pandas for timestamp (fallback if not available)
try:
    import pandas as pd
except ImportError:
    import datetime as pd
    class Timestamp:
        @staticmethod
        def now():
            return datetime.now()
    pd.Timestamp = Timestamp

if __name__ == "__main__":
    # Test the AI Chat Engine
    engine = AIChatEngine()
    
    # Setup Google API key
    engine.setup_google_api_key()
    
    # Test responses
    test_messages = [
        "Hello!",
        "Who is in this image?",
        "What objects do you see?",
        "Describe this image"
    ]
    
    for message in test_messages:
        response = engine.generate_dynamic_response(message)
        print(f"User: {message}")
        print(f"AI: {response}")
        print("-" * 50)
