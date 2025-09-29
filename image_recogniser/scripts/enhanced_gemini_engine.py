import os
import requests
import json
from typing import Dict, List, Optional
from enhanced_rag_system import EnhancedRAGSystem

class EnhancedGeminiEngine:
    """
    Enhanced Gemini Engine with Instruction Fine-tuning and RAG
    """
    
    def __init__(self):
        self.api_key = self.get_google_api_key()
        self.rag_system = EnhancedRAGSystem()
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        print("🚀 Enhanced Gemini Engine Initialized!")
    
    def get_google_api_key(self) -> str:
        """Get Google API key"""
        # Try environment variable first
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            # Try to read from file
            try:
                key_file = '../models/google_api_key.txt'
                if os.path.exists(key_file):
                    with open(key_file, 'r') as f:
                        api_key = f.read().strip()
            except:
                pass
        
        if not api_key:
            print("⚠️ Google API key not found!")
            print("💡 Get FREE API key from: https://makersuite.google.com/app/apikey")
        
        return api_key
    
    def generate_instruction_tuned_response(self, user_message: str, image_context: Dict = None) -> str:
        """Generate response using instruction fine-tuning and RAG"""
        try:
            if not self.api_key:
                return "Google API key not configured. Please set up your API key."
            
            # 1. Determine context type
            context_type = self.rag_system.determine_context_type(user_message, image_context)
            
            # 2. Retrieve relevant knowledge using RAG
            relevant_contexts = self.rag_system.retrieve_relevant_context(
                user_message, context_type
            )
            
            # 3. Build instruction-tuned prompt
            instruction_prompt = self.rag_system.build_instruction_prompt(
                context_type, image_context, relevant_contexts, user_message
            )
            
            # 4. Call Gemini with enhanced prompt
            response = self.call_gemini_with_instructions(
                instruction_prompt, user_message, context_type
            )
            
            # 5. Update knowledge base with new interaction
            self.rag_system.update_knowledge_base(user_message, response, image_context)
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating instruction-tuned response: {e}")
            # Fallback to simple response
            try:
                simple_response = self.call_gemini_with_instructions(
                    "You are a helpful AI assistant. Respond naturally and conversationally to the user's message.",
                    user_message, "conversational"
                )
                return simple_response
            except:
                return f"I apologize, but I encountered an error: {e}"
    
    def call_gemini_with_instructions(self, instruction_prompt: str, user_message: str, 
                                    context_type: str) -> str:
        """Call Gemini API with instruction-tuned prompt"""
        try:
            url = f"{self.base_url}/gemini-2.5-flash:generateContent?key={self.api_key}"
            
            # Build the complete prompt
            complete_prompt = f"""
{instruction_prompt}

USER MESSAGE: {user_message}

Please provide a helpful, accurate, and engaging response based on the instructions above.
"""
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": complete_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 300,
                    "topP": 0.8,
                    "topK": 40,
                    "stopSequences": []
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        return candidate["content"]["parts"][0]["text"].strip()
                    else:
                        print(f"❌ Unexpected response format: {candidate}")
                        # Try to extract any text from the response
                        if "text" in str(candidate):
                            return "I'm having trouble processing the response, but I'm here to help! What would you like to know?"
                        return "I couldn't generate a response. Please try again."
                else:
                    return "I couldn't generate a response. Please try again."
            else:
                print(f"❌ Gemini API error: {response.status_code} - {response.text}")
                return f"API error: {response.status_code}"
                
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return f"I encountered an error while processing your request: {e}"
    
    def generate_contextual_response(self, user_message: str, image_context: Dict = None) -> str:
        """Generate contextual response with enhanced understanding"""
        try:
            # Analyze user intent
            intent = self.rag_system._determine_user_intent(user_message)
            
            # Get relevant conversation history
            recent_history = self.rag_system._get_recent_conversation_history()
            
            # Build contextual prompt
            contextual_prompt = f"""
You are an AI assistant for an image recognition system. You have access to:

CONVERSATION HISTORY:
{recent_history}

CURRENT IMAGE CONTEXT:
{json.dumps(image_context, indent=2) if image_context else "No image context available"}

USER INTENT: {intent}

USER MESSAGE: {user_message}

Instructions:
1. Maintain conversational flow with the user
2. Reference previous interactions when relevant
3. Provide helpful and accurate information
4. Be engaging and friendly
5. Ask clarifying questions when needed
6. Learn from the conversation context

Respond naturally and helpfully.
"""
            
            return self.call_gemini_with_instructions(
                contextual_prompt, user_message, "conversational"
            )
            
        except Exception as e:
            print(f"❌ Error generating contextual response: {e}")
            return f"I apologize, but I encountered an error: {e}"
    
    def generate_face_analysis_response(self, user_message: str, face_context: Dict) -> str:
        """Generate specialized face analysis response"""
        try:
            faces = face_context.get("faces", [])
            known_faces = [f for f in faces if f.get("name", "Unknown") != "Unknown"]
            unknown_faces = [f for f in faces if f.get("name", "Unknown") == "Unknown"]
            
            face_prompt = f"""
You are an expert face recognition AI. Analyze the provided face data:

KNOWN FACES: {known_faces}
UNKNOWN FACES: {unknown_faces}
TOTAL FACES: {len(faces)}

USER QUESTION: {user_message}

Instructions:
1. Identify known faces with confidence scores
2. Describe unknown faces with physical attributes
3. Provide age estimation if possible
4. Analyze facial expressions and emotions
5. Suggest actions for unknown faces
6. Be specific and helpful

Respond with structured analysis.
"""
            
            return self.call_gemini_with_instructions(
                face_prompt, user_message, "face_analysis"
            )
            
        except Exception as e:
            print(f"❌ Error generating face analysis response: {e}")
            return f"I encountered an error analyzing faces: {e}"
    
    def generate_object_detection_response(self, user_message: str, object_context: Dict) -> str:
        """Generate specialized object detection response"""
        try:
            objects = object_context.get("objects", [])
            categories = list(set([obj.get("label", "") for obj in objects]))
            
            object_prompt = f"""
You are an expert object detection AI. Analyze the provided object data:

DETECTED OBJECTS: {objects}
OBJECT CATEGORIES: {categories}
TOTAL OBJECTS: {len(objects)}

USER QUESTION: {user_message}

Instructions:
1. Categorize objects by type
2. Estimate object sizes and positions
3. Identify brand names if visible
4. Suggest potential uses or contexts
5. Flag any unusual objects
6. Analyze object relationships

Respond with detailed analysis.
"""
            
            return self.call_gemini_with_instructions(
                object_prompt, user_message, "object_detection"
            )
            
        except Exception as e:
            print(f"❌ Error generating object detection response: {e}")
            return f"I encountered an error analyzing objects: {e}"
    
    def generate_image_understanding_response(self, user_message: str, image_context: Dict) -> str:
        """Generate specialized image understanding response"""
        try:
            description = image_context.get("description", "")
            elements = image_context.get("elements", "")
            
            image_prompt = f"""
You are a comprehensive image analysis AI. Analyze the provided image data:

IMAGE DESCRIPTION: {description}
DETECTED ELEMENTS: {elements}

USER QUESTION: {user_message}

Instructions:
1. Provide scene understanding and context
2. Analyze lighting, composition, and mood
3. Identify potential activities or scenarios
4. Suggest related questions or follow-ups
5. Generate creative descriptions
6. Identify cultural or contextual elements

Respond with comprehensive analysis.
"""
            
            return self.call_gemini_with_instructions(
                image_prompt, user_message, "image_understanding"
            )
            
        except Exception as e:
            print(f"❌ Error generating image understanding response: {e}")
            return f"I encountered an error analyzing the image: {e}"
    
    def test_api_connection(self) -> bool:
        """Test if the API connection is working"""
        try:
            if not self.api_key:
                print("❌ No API key found")
                return False
            
            # Simple test request
            url = f"{self.base_url}/gemini-2.5-flash:generateContent?key={self.api_key}"
            payload = {
                "contents": [{
                    "parts": [{
                        "text": "Hello, this is a test message."
                    }]
                }],
                "generationConfig": {
                    "maxOutputTokens": 10
                }
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print("✅ Gemini API connection successful!")
                return True
            else:
                print(f"❌ Gemini API connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Gemini API connection error: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get system status information"""
        return {
            "api_key_configured": bool(self.api_key),
            "rag_system_initialized": bool(self.rag_system),
            "knowledge_base_loaded": bool(self.rag_system.knowledge_base),
            "interaction_history_count": len(self.rag_system.interaction_history),
            "instruction_templates_count": len(self.rag_system.instruction_templates)
        }

if __name__ == "__main__":
    # Test the Enhanced Gemini Engine
    engine = EnhancedGeminiEngine()
    print("✅ Enhanced Gemini Engine created successfully!")
    
    # Test API connection
    if engine.test_api_connection():
        print("✅ API connection test passed!")
    else:
        print("❌ API connection test failed!")
    
    # Test system status
    status = engine.get_system_status()
    print(f"System Status: {status}")
    
    # Test instruction-tuned response
    test_response = engine.generate_instruction_tuned_response(
        "Hello, how are you?", 
        None
    )
    print(f"Test Response: {test_response}")
