import os
import json
import pickle
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from PIL import Image
import io

class GlobalFaceEncodings:
    """
    Global Face Encodings for Universal Human Face Detection
    """
    
    def __init__(self, encodings_dir: str = "../models/face_encodings"):
        self.encodings_dir = encodings_dir
        self.global_encodings_file = os.path.join(encodings_dir, "global_face_encodings.pkl")
        self.human_traits_file = os.path.join(encodings_dir, "human_traits.json")
        
        # Create directory if it doesn't exist
        os.makedirs(encodings_dir, exist_ok=True)
        
        # Initialize face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Global face encodings for different human traits
        self.global_encodings = {
            "male_adult": [],
            "female_adult": [],
            "child": [],
            "elderly": [],
            "asian": [],
            "caucasian": [],
            "african": [],
            "hispanic": [],
            "generic_human": []
        }
        
        # Human traits database
        self.human_traits = {
            "age_groups": ["child", "young_adult", "middle_aged", "elderly"],
            "genders": ["male", "female"],
            "ethnicities": ["asian", "caucasian", "african", "hispanic", "mixed"],
            "expressions": ["neutral", "happy", "sad", "angry", "surprised"],
            "face_shapes": ["oval", "round", "square", "heart", "diamond"],
            "eye_colors": ["brown", "blue", "green", "hazel", "gray"],
            "hair_colors": ["black", "brown", "blonde", "red", "gray", "white"]
        }
        
        # Load existing encodings
        self.load_global_encodings()
        self.load_human_traits()
        
        print("🌍 Global Face Encodings System Initialized")
    
    def load_global_encodings(self):
        """Load global face encodings"""
        try:
            if os.path.exists(self.global_encodings_file):
                with open(self.global_encodings_file, 'rb') as f:
                    self.global_encodings = pickle.load(f)
                print(f"✅ Loaded global face encodings")
            else:
                print("ℹ️ No global encodings found, will create new ones")
        except Exception as e:
            print(f"❌ Error loading global encodings: {e}")
    
    def save_global_encodings(self):
        """Save global face encodings"""
        try:
            with open(self.global_encodings_file, 'wb') as f:
                pickle.dump(self.global_encodings, f)
            print("✅ Global face encodings saved")
        except Exception as e:
            print(f"❌ Error saving global encodings: {e}")
    
    def load_human_traits(self):
        """Load human traits database"""
        try:
            if os.path.exists(self.human_traits_file):
                with open(self.human_traits_file, 'r', encoding='utf-8') as f:
                    self.human_traits = json.load(f)
                print("✅ Human traits database loaded")
            else:
                self.create_default_human_traits()
        except Exception as e:
            print(f"❌ Error loading human traits: {e}")
            self.create_default_human_traits()
    
    def save_human_traits(self):
        """Save human traits database"""
        try:
            with open(self.human_traits_file, 'w', encoding='utf-8') as f:
                json.dump(self.human_traits, f, indent=2, ensure_ascii=False)
            print("✅ Human traits database saved")
        except Exception as e:
            print(f"❌ Error saving human traits: {e}")
    
    def create_default_human_traits(self):
        """Create default human traits database"""
        self.human_traits = {
            "age_groups": ["child", "young_adult", "middle_aged", "elderly"],
            "genders": ["male", "female"],
            "ethnicities": ["asian", "caucasian", "african", "hispanic", "mixed"],
            "expressions": ["neutral", "happy", "sad", "angry", "surprised"],
            "face_shapes": ["oval", "round", "square", "heart", "diamond"],
            "eye_colors": ["brown", "blue", "green", "hazel", "gray"],
            "hair_colors": ["black", "brown", "blonde", "red", "gray", "white"],
            "facial_features": {
                "beard": ["none", "light", "medium", "heavy"],
                "mustache": ["none", "light", "medium", "heavy"],
                "glasses": ["none", "reading", "sunglasses", "prescription"],
                "hat": ["none", "cap", "hat", "helmet"]
            }
        }
        self.save_human_traits()
    
    def download_sample_faces(self):
        """Download sample faces for training (placeholder URLs)"""
        sample_urls = {
            "male_adult": [
                "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face",
                "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face"
            ],
            "female_adult": [
                "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150&h=150&fit=crop&crop=face",
                "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop&crop=face"
            ],
            "child": [
                "https://images.unsplash.com/photo-1503454537195-1dcabb73ffb9?w=150&h=150&fit=crop&crop=face",
                "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&h=150&fit=crop&crop=face"
            ]
        }
        
        print("📥 Downloading sample faces for global encodings...")
        
        for category, urls in sample_urls.items():
            for i, url in enumerate(urls):
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        image = Image.open(io.BytesIO(response.content))
                        image_path = f"../data/sample_faces/{category}_{i}.jpg"
                        os.makedirs(os.path.dirname(image_path), exist_ok=True)
                        image.save(image_path)
                        
                        # Extract features
                        features = self.extract_face_features_from_path(image_path)
                        if features is not None:
                            self.global_encodings[category].append(features)
                        
                        print(f"✅ Downloaded and processed {category} sample {i+1}")
                    
                except Exception as e:
                    print(f"⚠️ Error downloading {url}: {e}")
        
        self.save_global_encodings()
        print("✅ Sample faces downloaded and processed")
    
    def extract_face_features_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face features from image path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            return self.extract_face_features(image)
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return None
    
    def extract_face_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract comprehensive face features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Resize to standard size
            face_resized = cv2.resize(face_roi, (128, 128))
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            equalized_face = cv2.equalizeHist(gray_face)
            
            # Extract multiple feature types
            features = []
            
            # 1. Pixel features
            features.extend(equalized_face.flatten())
            
            # 2. Histogram features
            hist = cv2.calcHist([equalized_face], [0], None, [64], [0, 256])
            features.extend(hist.flatten())
            
            # 3. LBP features (simplified)
            lbp_features = self.compute_simple_lbp(equalized_face)
            features.extend(lbp_features)
            
            # 4. Edge features
            edges = cv2.Canny(equalized_face, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
            features.extend(edge_hist.flatten())
            
            # 5. Texture features
            texture_features = self.compute_texture_features(equalized_face)
            features.extend(texture_features)
            
            # Convert to numpy array and normalize
            features = np.array(features, dtype=np.float32)
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting face features: {e}")
            return None
    
    def compute_simple_lbp(self, image: np.ndarray) -> np.ndarray:
        """Compute simplified Local Binary Pattern features"""
        try:
            h, w = image.shape
            lbp = np.zeros_like(image)
            
            # Simple LBP for center pixels
            for i in range(1, h-1, 4):  # Sample every 4th pixel
                for j in range(1, w-1, 4):
                    center = image[i, j]
                    binary_string = ''
                    
                    # 8-neighborhood
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    
                    lbp[i, j] = int(binary_string, 2)
            
            # Compute histogram
            hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
            return hist.astype(np.float32)
            
        except:
            return np.zeros(64, dtype=np.float32)
    
    def compute_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Compute texture features using Gabor filters"""
        try:
            features = []
            
            # Create Gabor kernels for different orientations
            for theta in [0, 45, 90, 135]:
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 10, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                features.extend([np.mean(filtered), np.std(filtered)])
            
            return np.array(features, dtype=np.float32)
            
        except:
            return np.zeros(8, dtype=np.float32)
    
    def classify_face_traits(self, face_features: np.ndarray) -> Dict[str, str]:
        """Classify face traits based on features"""
        try:
            traits = {}
            
            # Age classification (simplified)
            age_score = self.classify_age(face_features)
            traits["age_group"] = age_score
            
            # Gender classification (simplified)
            gender_score = self.classify_gender(face_features)
            traits["gender"] = gender_score
            
            # Ethnicity classification (simplified)
            ethnicity_score = self.classify_ethnicity(face_features)
            traits["ethnicity"] = ethnicity_score
            
            return traits
            
        except Exception as e:
            print(f"❌ Error classifying face traits: {e}")
            return {"age_group": "unknown", "gender": "unknown", "ethnicity": "unknown"}
    
    def classify_age(self, features: np.ndarray) -> str:
        """Classify age group based on features"""
        try:
            # Simple age classification based on feature patterns
            # This is a placeholder - in real implementation, you'd use trained models
            
            # Use feature statistics for age estimation
            feature_mean = np.mean(features)
            feature_std = np.std(features)
            
            if feature_mean < 0.3:
                return "child"
            elif feature_mean < 0.5:
                return "young_adult"
            elif feature_mean < 0.7:
                return "middle_aged"
            else:
                return "elderly"
                
        except:
            return "unknown"
    
    def classify_gender(self, features: np.ndarray) -> str:
        """Classify gender based on features"""
        try:
            # Simple gender classification based on feature patterns
            # This is a placeholder - in real implementation, you'd use trained models
            
            # Use feature variance for gender estimation
            feature_variance = np.var(features)
            
            if feature_variance > 0.1:
                return "male"
            else:
                return "female"
                
        except:
            return "unknown"
    
    def classify_ethnicity(self, features: np.ndarray) -> str:
        """Classify ethnicity based on features"""
        try:
            # Simple ethnicity classification based on feature patterns
            # This is a placeholder - in real implementation, you'd use trained models
            
            # Use feature distribution for ethnicity estimation
            feature_skew = np.mean((features - np.mean(features))**3) / (np.std(features)**3)
            
            if feature_skew > 0.1:
                return "asian"
            elif feature_skew > -0.1:
                return "caucasian"
            elif feature_skew > -0.3:
                return "african"
            else:
                return "hispanic"
                
        except:
            return "unknown"
    
    def recognize_face_with_traits(self, image_path: str) -> Dict:
        """Recognize face with trait classification"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            # Detect faces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {"faces": [], "total_faces": 0, "message": "No faces detected"}
            
            recognized_faces = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = image[y:y+h, x:x+w]
                
                # Extract features
                face_features = self.extract_face_features(face_roi)
                
                if face_features is None:
                    continue
                
                # Classify traits
                traits = self.classify_face_traits(face_features)
                
                # Compare with global encodings
                best_match = self.find_best_global_match(face_features)
                
                recognized_faces.append({
                    "location": {"top": y, "right": x + w, "bottom": y + h, "left": x},
                    "traits": traits,
                    "global_match": best_match,
                    "confidence": best_match.get("confidence", 0.0)
                })
            
            return {
                "faces": recognized_faces,
                "total_faces": len(recognized_faces),
                "image_path": image_path,
                "model": "Global Face Recognition with Traits"
            }
            
        except Exception as e:
            return {"error": str(e), "faces": [], "total_faces": 0}
    
    def find_best_global_match(self, face_features: np.ndarray) -> Dict:
        """Find best match from global encodings"""
        try:
            best_match = {"category": "unknown", "confidence": 0.0}
            
            for category, encodings in self.global_encodings.items():
                if not encodings:
                    continue
                
                # Calculate distances
                distances = []
                for encoding in encodings:
                    distance = np.linalg.norm(face_features - encoding)
                    distances.append(distance)
                
                # Find minimum distance
                min_distance = min(distances)
                confidence = max(0, 1 - (min_distance / 0.5))  # Normalize to 0-1
                
                if confidence > best_match["confidence"]:
                    best_match = {
                        "category": category,
                        "confidence": confidence,
                        "distance": min_distance
                    }
            
            return best_match
            
        except Exception as e:
            print(f"❌ Error finding global match: {e}")
            return {"category": "unknown", "confidence": 0.0}
    
    def add_face_to_global_encodings(self, image_path: str, category: str) -> bool:
        """Add face to global encodings"""
        try:
            features = self.extract_face_features_from_path(image_path)
            
            if features is None:
                print(f"❌ Could not extract features from {image_path}")
                return False
            
            if category not in self.global_encodings:
                self.global_encodings[category] = []
            
            self.global_encodings[category].append(features)
            self.save_global_encodings()
            
            print(f"✅ Added face to {category} category")
            return True
            
        except Exception as e:
            print(f"❌ Error adding face to global encodings: {e}")
            return False
    
    def get_global_stats(self) -> Dict:
        """Get global encodings statistics"""
        stats = {}
        for category, encodings in self.global_encodings.items():
            stats[category] = len(encodings)
        
        return {
            "total_categories": len(self.global_encodings),
            "total_encodings": sum(stats.values()),
            "category_breakdown": stats,
            "human_traits": self.human_traits
        }

if __name__ == "__main__":
    # Test the global face encodings system
    gfe = GlobalFaceEncodings()
    
    print("🌍 Global Face Encodings System Test")
    print("=" * 50)
    
    # Show stats
    stats = gfe.get_global_stats()
    print(f"Total categories: {stats['total_categories']}")
    print(f"Total encodings: {stats['total_encodings']}")
    print(f"Category breakdown: {stats['category_breakdown']}")
    
    # Test with sample image if available
    test_image = "../data/test/otherhumans/otherhumans.png"
    if os.path.exists(test_image):
        results = gfe.recognize_face_with_traits(test_image)
        print("\n🔍 Recognition Results:")
        print(json.dumps(results, indent=2))
    else:
        print("\nℹ️ No test image found. Please add an image to test the system.")
