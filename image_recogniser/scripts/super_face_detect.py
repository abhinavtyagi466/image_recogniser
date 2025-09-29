import os
import json
import pickle
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
from datetime import datetime

# Try to import MediaPipe for better face detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available for advanced face detection")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available, using OpenCV fallback")

class SuperFaceDetector:
    """
    SUPER ACCURATE Face Detection System with Multiple Methods
    """
    
    def __init__(self, encodings_dir: str = "../models/face_encodings"):
        self.encodings_dir = encodings_dir
        self.known_encodings = []
        self.known_names = []
        self.encodings_file = os.path.join(encodings_dir, "super_face_encodings.pkl")
        
        # Initialize multiple face detection methods
        self.setup_multiple_detectors()
        
        # Create directory if it doesn't exist
        os.makedirs(encodings_dir, exist_ok=True)
        
        # Load existing encodings
        self.load_encodings()
        
        print("🚀 SUPER FACE DETECTOR INITIALIZED!")
    
    def setup_multiple_detectors(self):
        """Setup multiple face detection methods for maximum accuracy"""
        try:
            # Method 1: MediaPipe Face Detection (Most Accurate)
            if MEDIAPIPE_AVAILABLE:
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_drawing = mp.solutions.drawing_utils
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=0,  # 0 for close-range, 1 for full-range
                    min_detection_confidence=0.5
                )
                print("✅ MediaPipe face detection initialized")
            else:
                self.face_detection = None
                print("⚠️ MediaPipe not available, using OpenCV only")
            
            # Method 2: OpenCV Haar Cascades (Fallback)
            self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.haar_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            # Method 3: Alternative cascade (if available)
            try:
                self.haar_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            except:
                self.haar_alt = None
            
            print("✅ Multiple face detectors initialized")
            
        except Exception as e:
            print(f"⚠️ Some detectors failed to load: {e}")
            self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.haar_profile = None
            self.haar_alt = None
    
    def load_encodings(self):
        """Load existing face encodings"""
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get('encodings', [])
                    self.known_names = data.get('names', [])
                print(f"✅ Loaded {len(self.known_names)} known faces")
            else:
                print("ℹ️ No existing face encodings found")
        except Exception as e:
            print(f"❌ Error loading encodings: {e}")
            self.known_encodings = []
            self.known_names = []
    
    def save_encodings(self):
        """Save face encodings"""
        try:
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names,
                'created_at': datetime.now().isoformat(),
                'version': '2.0'
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"✅ Saved {len(self.known_names)} face encodings")
        except Exception as e:
            print(f"❌ Error saving encodings: {e}")
    
    def detect_faces_super_accurate(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
        """
        ULTRA-ACCURATE face detection using MediaPipe + OpenCV fallback
        
        This method uses MediaPipe as primary detector (most accurate) and OpenCV as fallback
        to ensure reliable face detection with proper confidence scoring.
        
        Key improvements:
        - MediaPipe as primary detector (state-of-the-art accuracy)
        - OpenCV Haar cascade as fallback
        - Proper confidence scoring
        - Smart parameter tuning
        - Better NMS filtering
        
        Returns:
            List of (x, y, w, h, confidence, method) tuples
        """
        all_faces = []
        
        print(f"🔍 Starting ULTRA-ACCURATE face detection...")
        print(f"📏 Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Method 1: MediaPipe Face Detection (Primary - Most Accurate)
        if MEDIAPIPE_AVAILABLE and self.face_detection is not None:
            try:
                print("🎯 Using MediaPipe for face detection...")
                
                # Convert BGR to RGB for MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image
                results = self.face_detection.process(rgb_image)
                
                if results.detections:
                    print(f"🎯 MediaPipe detections: {len(results.detections)}")
                    
                    for detection in results.detections:
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = image.shape
                        
                        # Convert relative coordinates to absolute
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Get confidence score
                        confidence = detection.score[0]
                        
                        # Validate face size and position
                        if (width > 30 and height > 30 and 
                            x >= 0 and y >= 0 and 
                            x + width <= w and y + height <= h):
                            
                            all_faces.append((x, y, width, height, confidence, "mediapipe"))
                            print(f"   ✅ MediaPipe face: confidence={confidence:.2f}, size={width}x{height}")
                        else:
                            print(f"   ❌ MediaPipe face rejected: invalid size/position")
                else:
                    print("🎯 MediaPipe: No faces detected")
                    
            except Exception as e:
                print(f"❌ MediaPipe detection error: {e}")
        
        # Method 2: OpenCV Haar Cascade (Fallback)
        if len(all_faces) == 0:  # Only use OpenCV if MediaPipe found nothing
            try:
                print("🎯 Using OpenCV Haar cascade as fallback...")
                
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Use balanced parameters (not too strict, not too loose)
                faces = self.haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,        # Balanced scale
                    minNeighbors=4,         # Balanced threshold
                    minSize=(30, 30),       # Reasonable minimum size
                    maxSize=(400, 400),     # Reasonable maximum size
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                print(f"🎯 OpenCV Haar detections: {len(faces)}")
                
                for (x, y, w, h) in faces:
                    confidence = self.calculate_face_confidence(x, y, w, h, image.shape, "haar")
                    # Use lower threshold for OpenCV since it's fallback
                    if confidence > 0.2:  # Lower threshold for fallback
                        all_faces.append((x, y, w, h, confidence, "haar"))
                        print(f"   ✅ OpenCV face: confidence={confidence:.2f}, size={w}x{h}")
                    else:
                        print(f"   ❌ OpenCV face rejected: confidence={confidence:.2f}")
                        
            except Exception as e:
                print(f"❌ OpenCV detection error: {e}")
        
        print(f"🎯 Total faces before NMS: {len(all_faces)}")
        
        # Apply NMS to remove overlapping faces
        filtered_faces = self.advanced_nms(all_faces, overlap_threshold=0.3)
        
        print(f"🎯 Final accurate faces after NMS: {len(filtered_faces)}")
        for i, (x, y, w, h, conf, method) in enumerate(filtered_faces):
            print(f"   Face {i+1}: confidence={conf:.2f}, size={w}x{h}, method={method}")
        
        return filtered_faces
    
    def detect_faces_template_matching(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using template matching"""
        try:
            faces = []
            
            # Create a simple face template
            template_size = 50
            template = self.create_face_template(template_size)
            
            # Multi-scale template matching
            scales = [0.8, 1.0, 1.2, 1.5]
            
            for scale in scales:
                scaled_template = cv2.resize(template, (int(template_size * scale), int(template_size * scale)))
                
                # Template matching
                result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                # Find matches above threshold
                locations = np.where(result >= 0.4)
                
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    w, h = scaled_template.shape[::-1]
                    faces.append((x, y, w, h))
            
            return faces
            
        except Exception as e:
            print(f"⚠️ Template matching error: {e}")
            return []
    
    def create_face_template(self, size: int) -> np.ndarray:
        """Create a simple face template"""
        try:
            # Create a basic face-like template
            template = np.zeros((size, size), dtype=np.uint8)
            
            # Face outline (oval)
            cv2.ellipse(template, (size//2, size//2), (size//3, size//2), 0, 0, 360, 255, -1)
            
            # Eyes
            cv2.circle(template, (size//3, size//3), size//12, 0, -1)
            cv2.circle(template, (2*size//3, size//3), size//12, 0, -1)
            
            # Nose
            cv2.rectangle(template, (size//2-2, size//2-5), (size//2+2, size//2+5), 0, -1)
            
            # Mouth
            cv2.ellipse(template, (size//2, 2*size//3), (size//6, size//12), 0, 0, 180, 0, -1)
            
            return template
            
        except Exception as e:
            print(f"⚠️ Template creation error: {e}")
            return np.zeros((size, size), dtype=np.uint8)
    
    def calculate_face_confidence(self, x: int, y: int, w: int, h: int, image_shape: Tuple, method: str) -> float:
        """Calculate confidence based on face properties"""
        try:
            # For MediaPipe, we already have high-quality confidence scores
            if method == "mediapipe":
                # MediaPipe confidence is already very accurate, just validate basic properties
                img_h, img_w = image_shape[:2]
                size_ratio = (w * h) / (img_w * img_h)
                if 0.005 <= size_ratio <= 0.4:  # Basic size validation
                    return 0.9  # High confidence for MediaPipe
                else:
                    return 0.5  # Lower confidence for unusual sizes
            
            img_h, img_w = image_shape[:2]
            
            # Size confidence (prefer medium-sized faces) - MORE STRICT
            size_ratio = (w * h) / (img_w * img_h)
            if 0.01 <= size_ratio <= 0.2:  # Stricter range for realistic face sizes
                size_conf = 1.0
            elif 0.005 <= size_ratio < 0.01:  # Slightly too small
                size_conf = 0.5
            elif 0.2 < size_ratio <= 0.4:  # Slightly too large
                size_conf = 0.7
            else:  # Way too small or too large
                size_conf = 0.1
            
            # Position confidence (prefer center faces)
            center_x = x + w // 2
            center_y = y + h // 2
            img_center_x = img_w // 2
            img_center_y = img_h // 2
            
            distance_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_distance = np.sqrt(img_w**2 + img_h**2) / 2
            position_conf = 1.0 - (distance_from_center / max_distance)
            
            # Aspect ratio confidence - MORE STRICT for realistic faces
            aspect_ratio = w / h
            if 0.8 <= aspect_ratio <= 1.2:  # Stricter range for realistic face aspect ratios
                aspect_conf = 1.0
            elif 0.6 <= aspect_ratio < 0.8 or 1.2 < aspect_ratio <= 1.4:
                aspect_conf = 0.5
            else:
                aspect_conf = 0.1
            
            # Method confidence
            method_conf = {
                "mediapipe": 1.0,  # MediaPipe is most accurate
                "haar": 0.8,       # OpenCV Haar cascade
                "profile": 0.7,
                "alt": 0.6,
                "template": 0.5
            }.get(method, 0.5)
            
            # Overall confidence
            overall_conf = (
                size_conf * 0.3 +
                position_conf * 0.2 +
                aspect_conf * 0.2 +
                method_conf * 0.3
            )
            
            return min(overall_conf, 1.0)
            
        except:
            return 0.5
    
    def advanced_nms(self, faces: List[Tuple[int, int, int, int, float, str]], 
                    overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float, str]]:
        """Advanced Non-Maximum Suppression"""
        if len(faces) == 0:
            return []
        
        # Sort by confidence
        faces.sort(key=lambda x: x[4], reverse=True)
        
        filtered_faces = []
        
        for current_face in faces:
            x1, y1, w1, h1, conf1, method1 = current_face
            
            # Check overlap with already selected faces
            is_duplicate = False
            for selected_face in filtered_faces:
                x2, y2, w2, h2, conf2, method2 = selected_face
                
                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    union = w1 * h1 + w2 * h2 - intersection
                    overlap = intersection / union if union > 0 else 0
                    
                    if overlap > overlap_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_faces.append(current_face)
        
        return filtered_faces
    
    def extract_face_features(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract face features"""
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_roi, (100, 100))
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            equalized_face = cv2.equalizeHist(gray_face)
            
            # Extract multiple features
            features = []
            
            # 1. Flattened pixel values
            features.extend(equalized_face.flatten())
            
            # 2. Histogram features
            hist = cv2.calcHist([equalized_face], [0], None, [32], [0, 256])
            features.extend(hist.flatten())
            
            # 3. Edge features
            edges = cv2.Canny(equalized_face, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
            features.extend(edge_hist.flatten())
            
            # Convert to numpy array and normalize
            features = np.array(features, dtype=np.float32)
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting face features: {e}")
            return None
    
    def recognize_face(self, image_path: str, tolerance: float = 0.6) -> Dict:
        """Recognize faces with SUPER ACCURACY"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "faces": [],
                    "total_faces": 0,
                    "error": "Could not load image"
                }
            
            print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
            print(f"📏 Image size: {image.shape[1]}x{image.shape[0]}")
            
            # Detect faces with super accuracy
            faces = self.detect_faces_super_accurate(image)
            
            print(f"🎯 Raw detections: {len(faces)}")
            
            if len(faces) == 0:
                return {
                    "faces": [],
                    "total_faces": 0,
                    "message": "No faces detected with current parameters"
                }
            
            recognized_faces = []
            
            for (x, y, w, h, conf, method) in faces:
                # Extract face region
                face_roi = image[y:y+h, x:x+w]
                
                # Extract features
                face_features = self.extract_face_features(face_roi)
                
                if face_features is None:
                    continue
                
                # Compare with known faces
                if len(self.known_encodings) > 0:
                    # Calculate distances
                    distances = []
                    for known_encoding in self.known_encodings:
                        distance = np.linalg.norm(face_features - known_encoding)
                        distances.append(distance)
                    
                    # Find best match
                    min_distance = min(distances)
                    best_match_index = distances.index(min_distance)
                    
                    # Calculate confidence (inverse of distance)
                    confidence = max(0, 1 - (min_distance / tolerance))
                    
                    if min_distance < tolerance:
                        name = self.known_names[best_match_index]
                    else:
                        name = "Unknown"
                        confidence = 0.0
                else:
                    name = "Unknown"
                    confidence = 0.0
                
                recognized_faces.append({
                    "name": name,
                    "confidence": confidence,
                    "location": {
                        "top": y,
                        "right": x + w,
                        "bottom": y + h,
                        "left": x
                    },
                    "detection_confidence": conf,
                    "detection_method": method,
                    "face_size": f"{w}x{h}"
                })
            
            # Sort by confidence
            recognized_faces.sort(key=lambda x: x["detection_confidence"], reverse=True)
            
            print(f"✅ Final detections: {len(recognized_faces)}")
            
            return {
                "faces": recognized_faces,
                "total_faces": len(recognized_faces),
                "image_path": image_path,
                "model": "SUPER ACCURATE Face Detection",
                "detection_methods": ["haar", "profile", "alt", "template"]
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "faces": [],
                "total_faces": 0
            }
    
    def register_face(self, name: str, image_path: str) -> bool:
        """Register a new face"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return False
            
            # Detect faces
            faces = self.detect_faces_super_accurate(image)
            
            if len(faces) == 0:
                print(f"❌ No face found in {image_path}")
                return False
            
            # Use the best face
            best_face = max(faces, key=lambda x: x[4])  # Max confidence
            x, y, w, h, conf, method = best_face
            
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            face_features = self.extract_face_features(face_roi)
            
            if face_features is None:
                print(f"❌ Could not extract features from {image_path}")
                return False
            
            # Add to known faces
            self.known_encodings.append(face_features)
            self.known_names.append(name)
            
            # Save encodings
            self.save_encodings()
            
            print(f"✅ Successfully registered face for {name} (Confidence: {conf:.2f}, Method: {method})")
            return True
            
        except Exception as e:
            print(f"❌ Error registering face: {e}")
            return False
    
    def get_known_faces(self) -> List[str]:
        """Get list of known face names"""
        return self.known_names.copy()

# Convenience functions
def register_face(name: str, image_path: str) -> bool:
    """Register a new face in the database"""
    detector = SuperFaceDetector()
    return detector.register_face(name, image_path)

def recognize_face(image_path: str, tolerance: float = 0.6) -> Dict:
    """Recognize faces in an image"""
    detector = SuperFaceDetector()
    return detector.recognize_face(image_path, tolerance)

def get_known_faces() -> List[str]:
    """Get list of known face names"""
    detector = SuperFaceDetector()
    return detector.get_known_faces()

if __name__ == "__main__":
    # Test the super face detector
    detector = SuperFaceDetector()
    
    print("🚀 SUPER FACE DETECTOR TEST")
    print("=" * 50)
    
    # Test with a sample image if available
    test_image = "../data/test/otherhumans/otherhumans.png"
    if os.path.exists(test_image):
        results = detector.recognize_face(test_image)
        print("🔍 Recognition Results:")
        print(json.dumps(results, indent=2))
    else:
        print("ℹ️ No test image found. Please add an image to test the detector.")
