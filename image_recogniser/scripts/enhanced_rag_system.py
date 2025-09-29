import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib

class EnhancedRAGSystem:
    """
    Enhanced RAG System for Gemini API Integration
    """
    
    def __init__(self):
        self.knowledge_base_path = "../models/knowledge_base.json"
        self.vector_store_path = "../models/vector_store.json"
        self.interaction_history_path = "../models/interaction_history.json"
        
        # Load existing data
        self.knowledge_base = self.load_knowledge_base()
        self.vector_store = self.load_vector_store()
        self.interaction_history = self.load_interaction_history()
        
        # Instruction templates for different contexts
        self.instruction_templates = {
            "face_analysis": """
            You are an expert face recognition AI. Analyze the provided image context:
            
            FACES DETECTED: {faces}
            CONFIDENCE SCORES: {confidence}
            KNOWN FACES: {known_faces}
            UNKNOWN FACES: {unknown_faces}
            
            Instructions:
            1. Identify known faces with high confidence (>80%)
            2. Describe unknown faces with physical attributes (age, gender, expression)
            3. Provide age estimation if possible
            4. Analyze facial expressions and emotions
            5. Suggest actions for unknown faces (register/ignore)
            6. Update face recognition patterns
            
            Respond in structured format with clear sections.
            """,
            
            "object_detection": """
            You are an expert object detection AI. Analyze the provided image context:
            
            OBJECTS DETECTED: {objects}
            CONFIDENCE SCORES: {confidence}
            OBJECT CATEGORIES: {categories}
            SPATIAL RELATIONSHIPS: {spatial}
            
            Instructions:
            1. Categorize objects by type (electronics, furniture, personal items, etc.)
            2. Estimate object sizes and positions
            3. Identify brand names if visible
            4. Suggest potential uses or contexts
            5. Flag any unusual or suspicious objects
            6. Analyze object interactions and relationships
            
            Provide detailed analysis with actionable insights.
            """,
            
            "image_understanding": """
            You are a helpful AI assistant analyzing an image. Here's what I can see:
            
            IMAGE DESCRIPTION: {description}
            DETECTED ELEMENTS: {elements}
            SCENE CONTEXT: {scene_context}
            MOOD/ATMOSPHERE: {mood}
            
            Please provide a natural, conversational response about what you see in the image. 
            Be helpful, accurate, and engaging. If the user asks specific questions, answer them directly.
            If no specific elements are detected, still provide a helpful response based on the image description.
            
            Keep your response conversational and friendly, not overly technical.
            """,
            
            "conversational": """
            You are a friendly and helpful AI assistant. 
            
            CONVERSATION HISTORY: {history}
            CURRENT IMAGE CONTEXT: {image_context}
            USER INTENT: {intent}
            
            Please respond naturally and conversationally. Be friendly, helpful, and engaging.
            If the user greets you, respond warmly. If they ask questions, answer helpfully.
            Keep your responses natural and not overly formal or technical.
            
            Be yourself and have a natural conversation!
            """
        }
        
        print("🧠 Enhanced RAG System Initialized!")
    
    def load_knowledge_base(self) -> Dict:
        """Load knowledge base from file"""
        try:
            if os.path.exists(self.knowledge_base_path):
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self.create_default_knowledge_base()
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            return self.create_default_knowledge_base()
    
    def create_default_knowledge_base(self) -> Dict:
        """Create default knowledge base structure"""
        return {
            "face_patterns": {
                "known_faces": {},
                "face_analysis_patterns": {
                    "high_confidence": "Known person, suggest greeting or identification",
                    "medium_confidence": "Possible match, ask for confirmation",
                    "low_confidence": "Unknown person, suggest registration or ignore"
                },
                "face_attributes": {
                    "age_groups": ["child", "teen", "adult", "elderly"],
                    "expressions": ["happy", "sad", "neutral", "surprised", "angry"],
                    "accessories": ["glasses", "hat", "mask", "jewelry"]
                }
            },
            "object_patterns": {
                "electronics": {
                    "laptop": {
                        "brands": ["Dell", "HP", "MacBook", "Lenovo", "Asus"],
                        "contexts": ["office", "home", "cafe", "library"],
                        "actions": ["work", "study", "entertainment", "presentation"]
                    },
                    "phone": {
                        "brands": ["iPhone", "Samsung", "Google", "OnePlus"],
                        "contexts": ["hand", "table", "pocket", "charging"],
                        "actions": ["calling", "texting", "browsing", "photography"]
                    }
                },
                "furniture": {
                    "chair": {
                        "types": ["office", "dining", "lounge", "recliner"],
                        "materials": ["wood", "metal", "plastic", "fabric"],
                        "contexts": ["office", "home", "restaurant", "outdoor"]
                    }
                }
            },
            "conversation_patterns": {
                "greetings": ["Hello", "Hi", "Good morning", "Good evening", "Namaste"],
                "questions": ["Who is this?", "What do you see?", "Can you identify?", "Tell me about"],
                "requests": ["Analyze this", "Tell me about", "Identify", "Describe"],
                "responses": {
                    "positive": ["Great!", "Excellent!", "Perfect!", "Wonderful!"],
                    "negative": ["I'm not sure", "Let me try again", "I need more information"],
                    "helpful": ["I can help with", "Let me assist you", "Here's what I found"]
                }
            },
            "context_patterns": {
                "office": ["desk", "computer", "meeting", "work", "business"],
                "home": ["living room", "kitchen", "bedroom", "family", "relaxing"],
                "outdoor": ["park", "street", "nature", "sunlight", "fresh air"],
                "social": ["party", "gathering", "friends", "celebration", "event"]
            }
        }
    
    def load_vector_store(self) -> Dict:
        """Load vector store for similarity search"""
        try:
            if os.path.exists(self.vector_store_path):
                with open(self.vector_store_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return {}
    
    def load_interaction_history(self) -> List:
        """Load interaction history"""
        try:
            if os.path.exists(self.interaction_history_path):
                with open(self.interaction_history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            print(f"❌ Error loading interaction history: {e}")
            return []
    
    def save_knowledge_base(self):
        """Save knowledge base to file"""
        try:
            os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            print("✅ Knowledge base saved successfully")
        except Exception as e:
            print(f"❌ Error saving knowledge base: {e}")
    
    def save_interaction_history(self):
        """Save interaction history"""
        try:
            os.makedirs(os.path.dirname(self.interaction_history_path), exist_ok=True)
            with open(self.interaction_history_path, 'w', encoding='utf-8') as f:
                json.dump(self.interaction_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving interaction history: {e}")
    
    def calculate_similarity(self, query: str, content: str) -> float:
        """Calculate similarity between query and content"""
        try:
            # Simple word-based similarity
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words or not content_words:
                return 0.0
            
            intersection = query_words.intersection(content_words)
            union = query_words.union(content_words)
            
            return len(intersection) / len(union) if union else 0.0
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            return 0.0
    
    def retrieve_relevant_context(self, query: str, context_type: str = "general") -> List[Dict]:
        """Retrieve relevant context from knowledge base using RAG"""
        relevant_contexts = []
        
        try:
            # Search in different sections based on context type
            search_sections = []
            
            if context_type == "face_analysis":
                search_sections = ["face_patterns"]
            elif context_type == "object_detection":
                search_sections = ["object_patterns"]
            elif context_type == "conversational":
                search_sections = ["conversation_patterns", "context_patterns"]
            else:
                search_sections = ["face_patterns", "object_patterns", "conversation_patterns", "context_patterns"]
            
            # Search in relevant sections
            for section in search_sections:
                if section in self.knowledge_base:
                    section_data = self.knowledge_base[section]
                    self._search_in_section(query, section_data, section, relevant_contexts)
            
            # Search in interaction history
            self._search_in_interaction_history(query, relevant_contexts)
            
            # Sort by similarity and return top results
            relevant_contexts.sort(key=lambda x: x["similarity"], reverse=True)
            return relevant_contexts[:5]  # Top 5 most relevant
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return []
    
    def _search_in_section(self, query: str, section_data: Dict, section_name: str, results: List):
        """Search within a specific section of knowledge base"""
        try:
            for key, value in section_data.items():
                if isinstance(value, dict):
                    self._search_in_section(query, value, f"{section_name}.{key}", results)
                elif isinstance(value, str):
                    similarity = self.calculate_similarity(query, value)
                    if similarity > 0.1:  # Threshold for relevance
                        results.append({
                            "content": value,
                            "similarity": similarity,
                            "source": f"{section_name}.{key}",
                            "type": "knowledge_base"
                        })
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            similarity = self.calculate_similarity(query, item)
                            if similarity > 0.1:
                                results.append({
                                    "content": item,
                                    "similarity": similarity,
                                    "source": f"{section_name}.{key}",
                                    "type": "knowledge_base"
                                })
        except Exception as e:
            print(f"❌ Error searching in section {section_name}: {e}")
    
    def _search_in_interaction_history(self, query: str, results: List):
        """Search in interaction history"""
        try:
            for interaction in self.interaction_history[-50:]:  # Last 50 interactions
                user_msg = interaction.get("user_message", "")
                ai_response = interaction.get("ai_response", "")
                
                # Check similarity with user message
                user_similarity = self.calculate_similarity(query, user_msg)
                if user_similarity > 0.2:
                    results.append({
                        "content": f"Previous similar question: {user_msg}",
                        "similarity": user_similarity,
                        "source": "interaction_history",
                        "type": "user_question",
                        "response": ai_response
                    })
                
                # Check similarity with AI response
                response_similarity = self.calculate_similarity(query, ai_response)
                if response_similarity > 0.2:
                    results.append({
                        "content": f"Previous similar response: {ai_response}",
                        "similarity": response_similarity,
                        "source": "interaction_history",
                        "type": "ai_response"
                    })
        except Exception as e:
            print(f"❌ Error searching interaction history: {e}")
    
    def determine_context_type(self, user_message: str, image_context: Dict = None) -> str:
        """Determine the type of context for the user message"""
        message_lower = user_message.lower()
        
        # If there's image context, prioritize image understanding
        if image_context and any(key in image_context for key in ["faces", "products", "overview"]):
            # Face-related keywords (specific face questions) - be more specific
            face_keywords = ["face", "person", "people", "who", "recognize", "identify", "age", "gender", "expression"]
            if any(keyword in message_lower for keyword in face_keywords) and not any(word in message_lower for word in ["what", "see", "anything", "something"]):
                return "face_analysis"
            
            # Object-related keywords (specific object questions)
            object_keywords = ["object", "item", "thing", "product", "detect", "identify"]
            if any(keyword in message_lower for keyword in object_keywords) and not any(word in message_lower for word in ["what", "see", "anything", "something"]):
                return "object_detection"
            
            # General image questions (default for image context)
            image_keywords = ["describe", "analyze", "overview", "summary", "tell me about", "explain", "see in", "what's in", "what", "see", "anything", "something", "human"]
            if any(keyword in message_lower for keyword in image_keywords):
                return "image_understanding"
            
            # Default to image understanding if image context exists
            return "image_understanding"
        
        # Default to conversational for general chat
        return "conversational"
    
    def build_instruction_prompt(self, context_type: str, image_context: Dict = None, 
                               relevant_contexts: List[Dict] = None, user_message: str = "") -> str:
        """Build instruction-tuned prompt"""
        try:
            # Get base instruction template
            base_instruction = self.instruction_templates.get(context_type, 
                self.instruction_templates["conversational"])
            
            # Format with image context
            if image_context:
                faces = image_context.get("faces", {}).get("faces", [])
                objects = image_context.get("products", {}).get("objects", [])
                overview = image_context.get("overview", {})
                
                # Separate known and unknown faces
                known_faces = [f for f in faces if f.get("name", "Unknown") != "Unknown"]
                unknown_faces = [f for f in faces if f.get("name", "Unknown") == "Unknown"]
                
                # Extract object categories
                categories = list(set([obj.get("label", "") for obj in objects]))
                
                # Build spatial relationships
                spatial = self._analyze_spatial_relationships(objects)
                
                # Determine scene context
                scene_context = self._determine_scene_context(faces, objects, overview)
                
                # Determine mood/atmosphere
                mood = self._determine_mood_atmosphere(faces, objects, overview)
                
                base_instruction = base_instruction.format(
                    faces=faces,
                    confidence=[f.get("confidence", 0) for f in faces],
                    known_faces=known_faces,
                    unknown_faces=unknown_faces,
                    objects=objects,
                    categories=categories,
                    spatial=spatial,
                    description=overview.get("description", ""),
                    elements=f"{faces} + {objects}",
                    scene_context=scene_context,
                    mood=mood,
                    history=self._get_recent_conversation_history(),
                    image_context=image_context,
                    intent=self._determine_user_intent(user_message)
                )
            
            # Add RAG context
            if relevant_contexts:
                rag_context = "\n\nRELEVANT KNOWLEDGE BASE:\n"
                for ctx in relevant_contexts[:3]:  # Top 3 most relevant
                    rag_context += f"- {ctx['content']} (Source: {ctx['source']})\n"
                    if ctx.get('response'):
                        rag_context += f"  Previous response: {ctx['response']}\n"
                
                base_instruction += rag_context
            
            return base_instruction
            
        except Exception as e:
            print(f"❌ Error building instruction prompt: {e}")
            return self.instruction_templates["conversational"]
    
    def _analyze_spatial_relationships(self, objects: List[Dict]) -> str:
        """Analyze spatial relationships between objects"""
        if len(objects) < 2:
            return "Single object detected"
        
        relationships = []
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                # Simple spatial analysis based on bounding boxes
                if obj1.get("bbox") and obj2.get("bbox"):
                    rel = self._determine_spatial_relation(obj1["bbox"], obj2["bbox"])
                    if rel:
                        relationships.append(f"{obj1.get('label', 'Object1')} is {rel} {obj2.get('label', 'Object2')}")
        
        return "; ".join(relationships) if relationships else "Multiple objects detected"
    
    def _determine_spatial_relation(self, bbox1: List, bbox2: List) -> str:
        """Determine spatial relation between two bounding boxes"""
        try:
            # Simple spatial analysis
            x1_center = (bbox1[0] + bbox1[2]) / 2
            y1_center = (bbox1[1] + bbox1[3]) / 2
            x2_center = (bbox2[0] + bbox2[2]) / 2
            y2_center = (bbox2[1] + bbox2[3]) / 2
            
            if abs(x1_center - x2_center) > abs(y1_center - y2_center):
                return "left of" if x1_center < x2_center else "right of"
            else:
                return "above" if y1_center < y2_center else "below"
        except:
            return "near"
    
    def _determine_scene_context(self, faces: List, objects: List, overview: Dict) -> str:
        """Determine scene context based on detected elements"""
        context_indicators = []
        
        # Analyze objects for context clues
        for obj in objects:
            label = obj.get("label", "").lower()
            if any(word in label for word in ["laptop", "computer", "desk", "office"]):
                context_indicators.append("office")
            elif any(word in label for word in ["chair", "table", "sofa", "tv"]):
                context_indicators.append("home")
            elif any(word in label for word in ["phone", "camera", "book"]):
                context_indicators.append("personal")
        
        # Analyze faces for context
        if faces:
            context_indicators.append("social")
        
        # Analyze overview description
        description = overview.get("description", "").lower()
        if any(word in description for word in ["outdoor", "nature", "park", "street"]):
            context_indicators.append("outdoor")
        
        return ", ".join(set(context_indicators)) if context_indicators else "general"
    
    def _determine_mood_atmosphere(self, faces: List, objects: List, overview: Dict) -> str:
        """Determine mood/atmosphere of the scene"""
        mood_indicators = []
        
        # Analyze face expressions
        for face in faces:
            # This would need to be enhanced with actual emotion detection
            mood_indicators.append("neutral")  # Default
        
        # Analyze scene description
        description = overview.get("description", "").lower()
        if any(word in description for word in ["bright", "sunny", "happy", "cheerful"]):
            mood_indicators.append("positive")
        elif any(word in description for word in ["dark", "gloomy", "sad", "serious"]):
            mood_indicators.append("negative")
        
        return ", ".join(set(mood_indicators)) if mood_indicators else "neutral"
    
    def _get_recent_conversation_history(self) -> str:
        """Get recent conversation history"""
        try:
            recent_history = self.interaction_history[-3:]  # Last 3 interactions
            history_text = []
            for interaction in recent_history:
                history_text.append(f"User: {interaction.get('user_message', '')}")
                history_text.append(f"AI: {interaction.get('ai_response', '')}")
            return "\n".join(history_text)
        except:
            return "No recent conversation history"
    
    def _determine_user_intent(self, user_message: str) -> str:
        """Determine user intent from message"""
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        elif any(word in message_lower for word in ["who", "identify", "recognize"]):
            return "identification"
        elif any(word in message_lower for word in ["what", "detect", "see", "objects"]):
            return "detection"
        elif any(word in message_lower for word in ["describe", "analyze", "explain"]):
            return "analysis"
        elif any(word in message_lower for word in ["help", "how", "can you"]):
            return "help_request"
        else:
            return "general_inquiry"
    
    def update_knowledge_base(self, user_message: str, ai_response: str, image_context: Dict = None):
        """Update knowledge base with new interaction"""
        try:
            # Add to interaction history
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "ai_response": ai_response,
                "image_context": image_context,
                "context_type": self.determine_context_type(user_message, image_context)
            }
            
            self.interaction_history.append(interaction)
            
            # Keep only last 1000 interactions
            if len(self.interaction_history) > 1000:
                self.interaction_history = self.interaction_history[-1000:]
            
            # Update knowledge base patterns
            self._update_face_patterns(image_context)
            self._update_object_patterns(image_context)
            self._update_conversation_patterns(user_message, ai_response)
            
            # Save updated data
            self.save_knowledge_base()
            self.save_interaction_history()
            
        except Exception as e:
            print(f"❌ Error updating knowledge base: {e}")
    
    def _update_face_patterns(self, image_context: Dict):
        """Update face recognition patterns"""
        try:
            if not image_context or "faces" not in image_context:
                return
            
            faces = image_context.get("faces", {}).get("faces", [])
            for face in faces:
                name = face.get("name", "Unknown")
                confidence = face.get("confidence", 0)
                
                if name != "Unknown" and confidence > 0.8:
                    # Update known faces
                    if name not in self.knowledge_base["face_patterns"]["known_faces"]:
                        self.knowledge_base["face_patterns"]["known_faces"][name] = {
                            "appearances": 1,
                            "contexts": [],
                            "attributes": [],
                            "last_seen": datetime.now().isoformat(),
                            "confidence_history": [confidence]
                        }
                    else:
                        face_data = self.knowledge_base["face_patterns"]["known_faces"][name]
                        face_data["appearances"] += 1
                        face_data["last_seen"] = datetime.now().isoformat()
                        face_data["confidence_history"].append(confidence)
                        
                        # Keep only last 10 confidence scores
                        if len(face_data["confidence_history"]) > 10:
                            face_data["confidence_history"] = face_data["confidence_history"][-10:]
        except Exception as e:
            print(f"❌ Error updating face patterns: {e}")
    
    def _update_object_patterns(self, image_context: Dict):
        """Update object detection patterns"""
        try:
            if not image_context or "products" not in image_context:
                return
            
            objects = image_context.get("products", {}).get("objects", [])
            for obj in objects:
                label = obj.get("label", "")
                confidence = obj.get("confidence", 0)
                
                if confidence > 0.7:
                    # Update object patterns
                    if "object_patterns" not in self.knowledge_base:
                        self.knowledge_base["object_patterns"] = {}
                    
                    category = self._categorize_object(label)
                    if category not in self.knowledge_base["object_patterns"]:
                        self.knowledge_base["object_patterns"][category] = {}
                    
                    if label not in self.knowledge_base["object_patterns"][category]:
                        self.knowledge_base["object_patterns"][category][label] = {
                            "appearances": 1,
                            "contexts": [],
                            "confidence_history": [confidence],
                            "last_seen": datetime.now().isoformat()
                        }
                    else:
                        obj_data = self.knowledge_base["object_patterns"][category][label]
                        obj_data["appearances"] += 1
                        obj_data["last_seen"] = datetime.now().isoformat()
                        obj_data["confidence_history"].append(confidence)
                        
                        # Keep only last 10 confidence scores
                        if len(obj_data["confidence_history"]) > 10:
                            obj_data["confidence_history"] = obj_data["confidence_history"][-10:]
        except Exception as e:
            print(f"❌ Error updating object patterns: {e}")
    
    def _categorize_object(self, label: str) -> str:
        """Categorize object into main categories"""
        label_lower = label.lower()
        
        if any(word in label_lower for word in ["laptop", "computer", "phone", "tablet", "camera"]):
            return "electronics"
        elif any(word in label_lower for word in ["chair", "table", "sofa", "bed", "desk"]):
            return "furniture"
        elif any(word in label_lower for word in ["book", "paper", "document", "magazine"]):
            return "documents"
        elif any(word in label_lower for word in ["bottle", "cup", "glass", "plate", "food"]):
            return "consumables"
        else:
            return "miscellaneous"
    
    def _update_conversation_patterns(self, user_message: str, ai_response: str):
        """Update conversation patterns"""
        try:
            # Extract patterns from user message
            user_words = user_message.lower().split()
            for word in user_words:
                if len(word) > 3:  # Only meaningful words
                    # Ensure conversation_patterns structure exists
                    if "conversation_patterns" not in self.knowledge_base:
                        self.knowledge_base["conversation_patterns"] = {
                            "questions": [],
                            "requests": [],
                            "responses": {"positive": [], "negative": [], "helpful": []}
                        }
                    
                    if word not in self.knowledge_base["conversation_patterns"]["questions"]:
                        # Add to appropriate category
                        if word in ["who", "what", "where", "when", "why", "how"]:
                            self.knowledge_base["conversation_patterns"]["questions"].append(word)
                        elif word in ["please", "can", "could", "would"]:
                            if "requests" not in self.knowledge_base["conversation_patterns"]:
                                self.knowledge_base["conversation_patterns"]["requests"] = []
                            self.knowledge_base["conversation_patterns"]["requests"].append(word)
            
            # Update response patterns
            if "positive" in ai_response.lower() or "great" in ai_response.lower():
                if "conversation_patterns" not in self.knowledge_base:
                    self.knowledge_base["conversation_patterns"] = {
                        "questions": [],
                        "requests": [],
                        "responses": {"positive": [], "negative": [], "helpful": []}
                    }
                if "positive_responses" not in self.knowledge_base["conversation_patterns"]:
                    self.knowledge_base["conversation_patterns"]["positive_responses"] = []
                self.knowledge_base["conversation_patterns"]["positive_responses"].append(ai_response[:50])
            
        except Exception as e:
            print(f"❌ Error updating conversation patterns: {e}")

if __name__ == "__main__":
    # Test the RAG system
    rag = EnhancedRAGSystem()
    print("✅ Enhanced RAG System created successfully!")
    
    # Test context retrieval
    contexts = rag.retrieve_relevant_context("Who is in this image?", "face_analysis")
    print(f"Retrieved {len(contexts)} relevant contexts")
    
    # Test instruction prompt building
    prompt = rag.build_instruction_prompt("face_analysis", None, contexts)
    print("✅ Instruction prompt built successfully!")
