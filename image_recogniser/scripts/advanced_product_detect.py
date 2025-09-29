import os
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import torch
from PIL import Image
import requests
from io import BytesIO

# Import ultra advanced detector
try:
    from ultra_advanced_detector import UltraAdvancedDetector
    ULTRA_ADVANCED_AVAILABLE = True
    print("✅ Ultra Advanced Detector available")
except ImportError:
    ULTRA_ADVANCED_AVAILABLE = False
    print("⚠️ Ultra Advanced Detector not available, using standard models")

class AdvancedProductDetector:
    """
    Advanced product detection system with custom trained models
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.custom_classes = []
        
        # Initialize ultra advanced detector if available
        self.ultra_detector = None
        if ULTRA_ADVANCED_AVAILABLE:
            try:
                self.ultra_detector = UltraAdvancedDetector()
                print("🚀 Ultra Advanced Detector initialized!")
            except Exception as e:
                print(f"⚠️ Could not initialize Ultra Advanced Detector: {e}")
                self.ultra_detector = None
        
        self.load_model()
    
    def load_model(self):
        """Load the best available model"""
        try:
            # Try to load custom trained model first
            if self.model_path and os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Loaded custom model: {self.model_path}")
                
                # Get custom classes
                if hasattr(self.model, 'names'):
                    self.custom_classes = list(self.model.names.values())
                    print(f"📊 Custom classes: {self.custom_classes}")
                
            else:
                # Try advanced models in order of preference
                advanced_models = [
                    ('yolov11x.pt', 'YOLOv11 Extra Large - Highest Accuracy'),
                    ('yolov11l.pt', 'YOLOv11 Large - High Accuracy'),
                    ('yolov11m.pt', 'YOLOv11 Medium - Good Accuracy'),
                    ('yolov8x.pt', 'YOLOv8 Extra Large - High Accuracy'),
                    ('yolov8l.pt', 'YOLOv8 Large - Good Accuracy'),
                    ('yolov8m.pt', 'YOLOv8 Medium - Balanced'),
                    ('yolov8s.pt', 'YOLOv8 Small - Fast'),
                    ('yolov8n.pt', 'YOLOv8 Nano - Fastest')
                ]
                
                model_loaded = False
                for model_name, description in advanced_models:
                    try:
                        print(f"🔄 Trying to load {description}...")
                        self.model = YOLO(model_name)
                        print(f"✅ Successfully loaded {description}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️ Could not load {model_name}: {e}")
                        continue
                
                if not model_loaded:
                    # Final fallback
                    self.model = YOLO('yolov8n.pt')
                    print("⚠️ Using basic YOLOv8n as final fallback")
                
                # Enhanced class list with more detailed categories
                self.custom_classes = [
                    # People and Animals
                    'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
                    'zebra', 'giraffe', 'monkey', 'panda', 'lion', 'tiger', 'rabbit', 'mouse',
                    
                    # Vehicles
                    'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'helicopter', 'scooter', 'skateboard', 'surfboard',
                    
                    # Electronics and Technology
                    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'tablet', 'camera',
                    'headphones', 'microphone', 'speaker', 'monitor', 'printer', 'scanner',
                    
                    # Furniture and Household
                    'chair', 'couch', 'bed', 'dining table', 'toilet', 'sink', 'refrigerator',
                    'microwave', 'oven', 'toaster', 'washing machine', 'dryer', 'dishwasher',
                    
                    # Food and Kitchen Items
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'plate',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'cookie', 'bread', 'cheese', 'egg', 'milk',
                    
                    # Personal Items
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'wallet', 'watch',
                    'glasses', 'hat', 'shirt', 'pants', 'shoes', 'socks', 'belt',
                    
                    # Sports and Recreation
                    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'tennis racket', 'golf club', 'basketball', 'football',
                    'soccer ball', 'volleyball', 'tennis ball',
                    
                    # Tools and Equipment
                    'scissors', 'hammer', 'screwdriver', 'wrench', 'pliers', 'drill', 'saw',
                    'ladder', 'toolbox', 'flashlight', 'battery', 'cable', 'wire',
                    
                    # Office and Stationery
                    'book', 'notebook', 'pen', 'pencil', 'eraser', 'ruler', 'calculator',
                    'stapler', 'paper clip', 'folder', 'envelope', 'stamp', 'calendar',
                    
                    # Decorative and Miscellaneous
                    'clock', 'vase', 'teddy bear', 'doll', 'toy', 'puzzle', 'game', 'card',
                    'flower', 'plant', 'tree', 'leaf', 'grass', 'rock', 'stone', 'crystal',
                    'jewelry', 'ring', 'necklace', 'bracelet', 'earring', 'brooch'
                ]
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def detect_products(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect products in an image using ultra advanced models
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary with detection results
        """
        # Try ultra advanced detector first
        if self.ultra_detector:
            try:
                print("🚀 Using Ultra Advanced Detector for maximum accuracy...")
                results = self.ultra_detector.detect_objects_ensemble(image_path, confidence_threshold)
                
                # Add model info
                results["detection_method"] = "Ultra Advanced Ensemble"
                results["models_used"] = results.get("models_used", [])
                results["ensemble_accuracy"] = results.get("ensemble_accuracy", 0.0)
                results["detection_quality"] = results.get("detection_quality", "Unknown")
                
                print(f"✅ Ultra Advanced Detection complete: {results['total_detections']} objects, Quality: {results['detection_quality']}")
                return results
                
            except Exception as e:
                print(f"⚠️ Ultra Advanced Detector failed: {e}")
                print("🔄 Falling back to standard models...")
        
        # Fallback to standard model
        if self.model is None:
            return {
                "objects": [],
                "total_detections": 0,
                "error": "No models loaded",
                "detection_method": "None"
            }
        
        try:
            # Run inference
            results = self.model(image_path, conf=confidence_threshold)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        if hasattr(self.model, 'names') and class_id in self.model.names:
                            class_name = self.model.names[class_id]
                        else:
                            class_name = f"class_{class_id}"
                        
                        # Filter for product-like objects
                        if self.is_product_like(class_name):
                            detections.append({
                                "label": class_name,
                                "confidence": float(confidence),
                                "bbox": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2)
                                },
                                "area": float((x2 - x1) * (y2 - y1))
                            })
            
            # Sort by confidence
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "objects": detections,
                "total_detections": len(detections),
                "image_path": image_path,
                "detection_method": "YOLO Advanced",
                "model_used": "YOLOv8/YOLOv11 Advanced",
                "confidence_threshold": confidence_threshold,
                "ensemble_accuracy": 0.0,
                "detection_quality": "Standard"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "objects": [],
                "total_detections": 0,
                "detection_method": "Error",
                "detection_quality": "Failed"
            }
    
    def is_product_like(self, class_name: str) -> bool:
        """Check if a detected object is product-like"""
        product_keywords = [
            'bottle', 'cup', 'bowl', 'laptop', 'cell phone', 'mouse', 'keyboard',
            'remote', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'backpack', 'handbag', 'suitcase', 'sports ball', 'frisbee',
            'skateboard', 'tennis racket', 'wine glass', 'fork', 'knife', 'spoon',
            'microwave', 'oven', 'toaster', 'refrigerator', 'tv', 'couch', 'chair',
            'bed', 'dining table', 'potted plant', 'toilet', 'sink'
        ]
        
        return any(keyword in class_name.lower() for keyword in product_keywords)
    
    def detect_specific_products(self, image_path: str, product_names: List[str]) -> Dict:
        """
        Detect specific products in an image
        
        Args:
            image_path: Path to input image
            product_names: List of product names to detect
            
        Returns:
            Dictionary with detection results
        """
        # Get all detections
        all_results = self.detect_products(image_path, confidence_threshold=0.3)
        
        # Filter for specific products
        specific_detections = []
        for detection in all_results.get("objects", []):
            if any(product.lower() in detection["label"].lower() for product in product_names):
                specific_detections.append(detection)
        
        return {
            "objects": specific_detections,
            "total_detections": len(specific_detections),
            "searched_products": product_names,
            "image_path": image_path
        }
    
    def get_product_analysis(self, image_path: str) -> Dict:
        """
        Get detailed product analysis
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with detailed analysis
        """
        detections = self.detect_products(image_path)
        
        if not detections.get("objects"):
            return {
                "analysis": "No products detected",
                "recommendations": ["Try different image", "Lower confidence threshold"],
                "detections": detections
            }
        
        # Analyze detected products
        products = detections["objects"]
        high_confidence = [p for p in products if p["confidence"] > 0.7]
        medium_confidence = [p for p in products if 0.5 <= p["confidence"] <= 0.7]
        low_confidence = [p for p in products if p["confidence"] < 0.5]
        
        analysis = {
            "total_products": len(products),
            "high_confidence": len(high_confidence),
            "medium_confidence": len(medium_confidence),
            "low_confidence": len(low_confidence),
            "product_types": list(set([p["label"] for p in products])),
            "largest_product": max(products, key=lambda x: x["area"]) if products else None,
            "most_confident": max(products, key=lambda x: x["confidence"]) if products else None,
            "detections": detections
        }
        
        # Generate recommendations
        recommendations = []
        if len(high_confidence) == 0:
            recommendations.append("Consider retraining model with more data")
        if len(low_confidence) > len(high_confidence):
            recommendations.append("Many low-confidence detections - check image quality")
        if len(products) == 0:
            recommendations.append("No products detected - try different image or model")
        
        analysis["recommendations"] = recommendations
        
        return analysis
    
    def draw_detections(self, image_path: str, output_path: str, 
                       detection_results: Dict) -> bool:
        """Draw detection boxes on image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Draw detection boxes
            for detection in detection_results.get("objects", []):
                bbox = detection["bbox"]
                label = detection["label"]
                confidence = detection["confidence"]
                
                # Draw rectangle
                cv2.rectangle(image, 
                             (int(bbox["x1"]), int(bbox["y1"])),
                             (int(bbox["x2"]), int(bbox["y2"])),
                             (0, 255, 0), 2)
                
                # Draw label
                label_text = f"{label}: {confidence:.2f}"
                cv2.putText(image, label_text,
                           (int(bbox["x1"]), int(bbox["y1"]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save image
            cv2.imwrite(output_path, image)
            return True
            
        except Exception as e:
            print(f"❌ Error drawing detections: {e}")
            return False
    
    def download_pretrained_model(self, model_url: str, save_path: str) -> bool:
        """Download a pretrained model"""
        try:
            print(f"📥 Downloading model from {model_url}")
            response = requests.get(model_url)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Model downloaded to {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False

# Convenience functions
def detect_products(image_path: str, confidence_threshold: float = 0.5) -> Dict:
    """Detect products in an image"""
    detector = AdvancedProductDetector()
    return detector.detect_products(image_path, confidence_threshold)

def get_product_analysis(image_path: str) -> Dict:
    """Get detailed product analysis"""
    detector = AdvancedProductDetector()
    return detector.get_product_analysis(image_path)

if __name__ == "__main__":
    # Test the advanced product detector
    detector = AdvancedProductDetector()
    
    # Test with a sample image if available
    test_image = "../data/test/bottle/bottle.png"
    if os.path.exists(test_image):
        results = detector.detect_products(test_image)
        print("🔍 Product Detection Results:")
        print(json.dumps(results, indent=2))
        
        analysis = detector.get_product_analysis(test_image)
        print("\n📊 Product Analysis:")
        print(json.dumps(analysis, indent=2))
    else:
        print("ℹ️ No test image found. Please add an image to test the detector.")
