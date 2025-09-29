#!/usr/bin/env python3
"""
🎥 LIVE CAMERA FACE & OBJECT DETECTION SYSTEM
- Real-time face and object detection
- Interactive labeling system
- Automatic data collection for training
- Live training capabilities
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk

# Import our existing models
from super_face_detect import SuperFaceDetector
from advanced_product_detect import AdvancedProductDetector
from data_mediator import DataMediator

class LiveCameraDetector:
    """
    LIVE CAMERA FACE & OBJECT DETECTION SYSTEM
    - Real-time face and object detection
    - Interactive labeling system
    - Automatic data collection for training
    - Live training capabilities
    """
    
    def __init__(self):
        # Initialize models
        self.face_detector = SuperFaceDetector()
        self.object_detector = AdvancedProductDetector()
        self.data_mediator = DataMediator()
        
        # Camera settings
        self.camera = None
        self.is_running = False
        self.current_frame = None
        
        # Detection settings
        self.detect_faces = True
        self.detect_objects = True
        self.show_labels = True
        self.save_data = True
        
        # Data collection
        self.collected_data = []
        self.labeling_mode = False
        self.current_labels = {}
        
        # GUI components
        self.root = None
        self.video_label = None
        self.control_frame = None
        
        # Callback functions for integration with mega_runengine
        self.detection_callback = None
        self.frame_callback = None
        self.latest_detections = []
        
        print("🎥 LIVE CAMERA DETECTOR INITIALIZED!")
    
    def set_detection_callback(self, callback):
        """Set callback function for detections (used by mega_runengine)"""
        self.detection_callback = callback
        print("✅ Detection callback set")
    
    def set_frame_callback(self, callback):
        """Set callback function for frames (used by mega_runengine)"""
        self.frame_callback = callback
        print("✅ Frame callback set")
    
    def get_latest_detections(self):
        """Get latest detections (used by mega_runengine)"""
        return self.latest_detections.copy()
    
    def start_camera(self, camera_index: int = 0):
        """Start camera capture with multiple camera index attempts"""
        try:
            # Try different camera indices to find working camera
            for idx in [0, 1, 2]:  # Try camera 0, 1, 2
                print(f"🔍 Trying camera index {idx}...")
                self.camera = cv2.VideoCapture(idx)
                
                if self.camera.isOpened():
                    # Test if camera actually works by reading a frame
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        print(f"✅ Camera {idx} started successfully")
                        break
                    else:
                        print(f"⚠️ Camera {idx} opened but no frame received")
                        self.camera.release()
                        self.camera = None
                else:
                    print(f"⚠️ Camera {idx} could not be opened")
                    if self.camera:
                        self.camera.release()
                        self.camera = None
            
            if not self.camera or not self.camera.isOpened():
                raise Exception("No working camera found on indices 0, 1, 2")
            
            # Set camera properties with smaller resolution for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"❌ Camera start error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        print("🛑 Camera stopped")
    
    def detect_and_label_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces and objects in frame with labels"""
        annotated_frame = frame.copy()
        detections = []
        
        # Face Detection with Enhanced Traits
        if self.detect_faces:
            try:
                face_results = self.face_detector.detect_faces_super_accurate(frame)
                
                for i, (x, y, w, h, conf, method) in enumerate(face_results):
                    # Draw face bounding box
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Extract face region for trait analysis
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Enhanced face traits analysis
                    traits = self.analyze_face_traits(face_roi)
                    
                    # Face label with traits
                    trait_text = f"{traits['gender']} {traits['age_range']}"
                    face_label = f"Face {i+1}: {conf:.2f} - {trait_text}"
                    cv2.putText(annotated_frame, face_label, (x, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Additional traits label
                    cv2.putText(annotated_frame, f"Emotion: {traits['emotion']}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Store detection data with traits
                    detections.append({
                        "type": "face",
                        "bbox": [x, y, w, h],
                        "confidence": conf,
                        "method": method,
                        "traits": traits,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                print(f"⚠️ Face detection error: {e}")
        
        # Enhanced Object Detection
        if self.detect_objects:
            try:
                # Save frame temporarily for object detection
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Use both YOLO and custom object detection
                object_results = self.object_detector.detect_products(temp_path)
                
                # Also try direct YOLO detection for better results
                yolo_results = self.detect_yolo_objects(frame)
                
                # Combine results
                all_objects = object_results.get("objects", []) + yolo_results
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                for i, obj in enumerate(all_objects):
                    if "box" in obj:
                        x1, y1, x2, y2 = obj["box"]
                        label = obj["label"]
                        conf = obj["confidence"]
                        
                        # Enhanced object analysis
                        obj_traits = self.analyze_object_traits(frame[y1:y2, x1:x2], label)
                        
                        # Draw object bounding box with different colors for different objects
                        color = self.get_object_color(label)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Enhanced object label
                        object_label = f"{label}: {conf:.2f}"
                        cv2.putText(annotated_frame, object_label, (x1, y1 - 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Additional object traits
                        if obj_traits:
                            trait_label = f"Size: {obj_traits['size']} | Color: {obj_traits['color']}"
                            cv2.putText(annotated_frame, trait_label, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Store detection data
                        detections.append({
                            "type": "object",
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "label": label,
                            "confidence": conf,
                            "traits": obj_traits,
                            "timestamp": datetime.now().isoformat()
                        })
                    
            except Exception as e:
                print(f"⚠️ Object detection error: {e}")
        
        # Add frame info
        face_count = len([d for d in detections if d['type'] == 'face'])
        object_count = len([d for d in detections if d['type'] == 'object'])
        info_text = f"Faces: {face_count} | Objects: {object_count}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp_text = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated_frame, timestamp_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save data if enabled
        if self.save_data and detections:
            self.collect_training_data(frame, detections)
        
        # Store latest detections for callbacks
        self.latest_detections = detections.copy()
        
        # Call detection callback if set (for mega_runengine integration)
        if self.detection_callback and detections:
            try:
                self.detection_callback(detections)
            except Exception as e:
                print(f"⚠️ Detection callback error: {e}")
        
        # Call frame callback if set (for mega_runengine integration)
        if self.frame_callback:
            try:
                self.frame_callback({
                    "frame": frame,
                    "annotated_frame": annotated_frame,
                    "detections": detections,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"⚠️ Frame callback error: {e}")
        
        return annotated_frame
    
    def analyze_face_traits(self, face_roi: np.ndarray) -> Dict:
        """Analyze face traits like age, gender, emotion"""
        try:
            traits = {
                "age_range": "Unknown",
                "gender": "Unknown", 
                "emotion": "Neutral",
                "ethnicity": "Unknown"
            }
            
            # Basic face analysis using OpenCV
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Age estimation based on face size and features
            face_area = face_roi.shape[0] * face_roi.shape[1]
            if face_area > 10000:
                traits["age_range"] = "Adult"
            elif face_area > 5000:
                traits["age_range"] = "Young Adult"
            else:
                traits["age_range"] = "Child/Teen"
            
            # Gender estimation based on face features (basic)
            # This is a simplified approach - in real implementation, you'd use ML models
            face_ratio = face_roi.shape[1] / face_roi.shape[0]
            if face_ratio > 0.8:
                traits["gender"] = "Male"
            else:
                traits["gender"] = "Female"
            
            # Emotion detection (simplified)
            # In real implementation, you'd use emotion recognition models
            brightness = np.mean(gray_face)
            if brightness > 150:
                traits["emotion"] = "Happy"
            elif brightness < 80:
                traits["emotion"] = "Sad"
            else:
                traits["emotion"] = "Neutral"
            
            return traits
            
        except Exception as e:
            print(f"⚠️ Face traits analysis error: {e}")
            return {"age_range": "Unknown", "gender": "Unknown", "emotion": "Unknown"}
    
    def analyze_object_traits(self, object_roi: np.ndarray, label: str) -> Dict:
        """Analyze object traits like size, color"""
        try:
            traits = {
                "size": "Medium",
                "color": "Unknown",
                "shape": "Unknown"
            }
            
            if object_roi.size == 0:
                return traits
            
            # Size analysis
            area = object_roi.shape[0] * object_roi.shape[1]
            if area > 50000:
                traits["size"] = "Large"
            elif area < 10000:
                traits["size"] = "Small"
            else:
                traits["size"] = "Medium"
            
            # Color analysis
            mean_color = np.mean(object_roi, axis=(0, 1))
            b, g, r = mean_color
            
            if r > g and r > b:
                traits["color"] = "Red"
            elif g > r and g > b:
                traits["color"] = "Green"
            elif b > r and b > g:
                traits["color"] = "Blue"
            elif np.mean(mean_color) > 180:
                traits["color"] = "White/Light"
            elif np.mean(mean_color) < 50:
                traits["color"] = "Black/Dark"
            else:
                traits["color"] = "Mixed"
            
            return traits
            
        except Exception as e:
            print(f"⚠️ Object traits analysis error: {e}")
            return {"size": "Unknown", "color": "Unknown"}
    
    def detect_yolo_objects(self, frame: np.ndarray) -> List[Dict]:
        """Direct YOLO object detection for better results"""
        try:
            from ultralytics import YOLO
            
            # Load YOLO model
            model_path = "yolov8n.pt"
            if not os.path.exists(model_path):
                return []
            
            model = YOLO(model_path)
            results = model(frame)
            
            objects = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = model.names[cls]
                        
                        # Filter for relevant objects
                        if class_name.lower() in ['bottle', 'phone', 'laptop', 'person', 'cup', 'book', 'cell phone']:
                            objects.append({
                                "box": [int(x1), int(y1), int(x2), int(y2)],
                                "label": class_name,
                                "confidence": float(conf)
                            })
            
            return objects
            
        except Exception as e:
            print(f"⚠️ YOLO detection error: {e}")
            return []
    
    def get_object_color(self, label: str) -> Tuple[int, int, int]:
        """Get color for object bounding box based on label"""
        color_map = {
            "bottle": (255, 0, 0),      # Red
            "phone": (0, 255, 0),       # Green
            "laptop": (0, 0, 255),      # Blue
            "person": (255, 255, 0),    # Cyan
            "cup": (255, 0, 255),       # Magenta
            "book": (0, 255, 255),      # Yellow
            "cell phone": (128, 0, 128) # Purple
        }
        return color_map.get(label.lower(), (255, 255, 255))  # White default
    
    def collect_training_data(self, frame: np.ndarray, detections: List[Dict]):
        """Collect data for training"""
        try:
            # Save frame with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_path = f"../data/live_capture/frame_{timestamp}.jpg"
            
            # Ensure directory exists
            os.makedirs("../data/live_capture", exist_ok=True)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            
            # Store detection metadata
            data_entry = {
                "frame_path": frame_path,
                "timestamp": timestamp,
                "detections": detections,
                "frame_size": frame.shape[:2]
            }
            
            self.collected_data.append(data_entry)
            
            # Save to data mediator
            for detection in detections:
                if detection["type"] == "face":
                    # Add face data
                    self.data_mediator.add_face_data(
                        name="Unknown",  # Will be labeled later
                        image_path=frame_path,
                        description=f"Live capture - {detection['method']}"
                    )
                elif detection["type"] == "object":
                    # Add object data
                    self.data_mediator.add_product_data(
                        category=detection["label"],
                        image_path=frame_path,
                        description=f"Live capture - {detection['confidence']:.2f}"
                    )
            
            print(f"📊 Collected data: {len(self.collected_data)} frames")
            
        except Exception as e:
            print(f"❌ Data collection error: {e}")
    
    def start_labeling_mode(self):
        """Start interactive labeling mode"""
        self.labeling_mode = True
        print("🏷️ Labeling mode activated!")
        print("Click on detected faces/objects to label them")
    
    def stop_labeling_mode(self):
        """Stop labeling mode"""
        self.labeling_mode = False
        print("🏷️ Labeling mode deactivated")
    
    def label_detection(self, detection_type: str, detection_id: int, label: str):
        """Label a specific detection"""
        try:
            if detection_type == "face":
                # Update face label in data mediator
                self.data_mediator.add_face_data(
                    name=label,
                    image_path=self.collected_data[-1]["frame_path"],
                    description=f"Labeled: {label}"
                )
                print(f"✅ Face labeled as: {label}")
                
            elif detection_type == "object":
                # Update object label
                self.data_mediator.add_product_data(
                    category=label,
                    image_path=self.collected_data[-1]["frame_path"],
                    description=f"Labeled: {label}"
                )
                print(f"✅ Object labeled as: {label}")
                
        except Exception as e:
            print(f"❌ Labeling error: {e}")
    
    def start_training(self):
        """Start training with collected data"""
        try:
            print("🎓 Starting training with collected data...")
            
            # Prepare data for training
            self.data_mediator.prepare_retraining()
            
            # Import and run training
            from train import retrain_with_new_data
            success = retrain_with_new_data()
            
            if success:
                print("✅ Training completed successfully!")
                messagebox.showinfo("Training Complete", "Model has been retrained with new data!")
            else:
                print("❌ Training failed")
                messagebox.showerror("Training Failed", "Training failed. Check logs for details.")
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            messagebox.showerror("Training Error", f"Training error: {e}")
    
    def create_gui(self):
        """Create GUI for live camera system"""
        self.root = tk.Tk()
        self.root.title("🎥 Live Camera Face & Object Detector")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # Main video display
        self.video_label = tk.Label(self.root, bg='black', text="Camera Feed", 
                                   font=('Arial', 16), fg='white')
        self.video_label.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Control panel
        self.control_frame = tk.Frame(self.root, bg='#2a2a2a')
        self.control_frame.pack(fill='x', padx=10, pady=5)
        
        # Camera controls
        camera_frame = tk.Frame(self.control_frame, bg='#2a2a2a')
        camera_frame.pack(side='left', padx=10)
        
        tk.Button(camera_frame, text="📹 Start Camera", command=self.start_camera_gui,
                 bg='#4CAF50', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        tk.Button(camera_frame, text="🛑 Stop Camera", command=self.stop_camera_gui,
                 bg='#f44336', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        # Detection controls
        detection_frame = tk.Frame(self.control_frame, bg='#2a2a2a')
        detection_frame.pack(side='left', padx=10)
        
        self.face_var = tk.BooleanVar(value=True)
        tk.Checkbutton(detection_frame, text="👤 Detect Faces", variable=self.face_var,
                      command=self.toggle_face_detection, bg='#2a2a2a', fg='white',
                      selectcolor='#4CAF50').pack(side='left', padx=5)
        
        self.object_var = tk.BooleanVar(value=True)
        tk.Checkbutton(detection_frame, text="🛍️ Detect Objects", variable=self.object_var,
                      command=self.toggle_object_detection, bg='#2a2a2a', fg='white',
                      selectcolor='#4CAF50').pack(side='left', padx=5)
        
        # Data collection controls
        data_frame = tk.Frame(self.control_frame, bg='#2a2a2a')
        data_frame.pack(side='left', padx=10)
        
        tk.Button(data_frame, text="🏷️ Start Labeling", command=self.start_labeling_mode,
                 bg='#FF9800', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        tk.Button(data_frame, text="🎓 Train Model", command=self.start_training,
                 bg='#9C27B0', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        # Status display
        self.status_label = tk.Label(self.control_frame, text="Status: Ready", 
                                   bg='#2a2a2a', fg='white', font=('Arial', 10))
        self.status_label.pack(side='right', padx=10)
        
        # Start video loop
        self.update_video()
        
        # Start GUI
        self.root.mainloop()
    
    def start_camera_gui(self):
        """Start camera from GUI"""
        if self.start_camera():
            self.status_label.config(text="Status: Camera Active")
        else:
            self.status_label.config(text="Status: Camera Error")
    
    def stop_camera_gui(self):
        """Stop camera from GUI"""
        self.stop_camera()
        self.status_label.config(text="Status: Camera Stopped")
    
    def toggle_face_detection(self):
        """Toggle face detection"""
        self.detect_faces = self.face_var.get()
        print(f"👤 Face detection: {'ON' if self.detect_faces else 'OFF'}")
    
    def toggle_object_detection(self):
        """Toggle object detection"""
        self.detect_objects = self.object_var.get()
        print(f"🛍️ Object detection: {'ON' if self.detect_objects else 'OFF'}")
    
    def update_video(self):
        """Update video display with better error handling"""
        try:
            if self.is_running and self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    # Process frame
                    processed_frame = self.detect_and_label_frame(frame)
                    
                    # Convert to PIL Image
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Resize for display - smaller size to avoid memory issues
                    display_size = (640, 480)
                    frame_pil = frame_pil.resize(display_size, Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage - keep reference to prevent garbage collection
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Update display - store reference in instance variable
                    if not hasattr(self, 'current_frame'):
                        self.current_frame = None
                    self.current_frame = frame_tk  # Keep reference
                    
                    if self.video_label:
                        self.video_label.config(image=frame_tk)
                        self.video_label.image = frame_tk
                    
                    # Update status
                    if hasattr(self, 'status_label') and self.status_label:
                        self.status_label.config(text=f"Status: Active | Data: {len(self.collected_data)} frames")
                else:
                    print("⚠️ Camera read failed - trying to restart...")
                    self.restart_camera()
            else:
                if self.is_running:
                    print("⚠️ Camera not available - trying to restart...")
                    self.restart_camera()
        except Exception as e:
            print(f"❌ Video update error: {e}")
            if self.is_running:
                print("🔄 Attempting to restart camera...")
                self.restart_camera()
        
        # Schedule next update
        if self.root:
            self.root.after(33, self.update_video)  # ~30 FPS
    
    def restart_camera(self):
        """Restart camera if it fails"""
        try:
            print("🔄 Restarting camera...")
            self.stop_camera()
            time.sleep(1)  # Wait a bit
            if self.start_camera():
                print("✅ Camera restarted successfully")
            else:
                print("❌ Failed to restart camera")
        except Exception as e:
            print(f"❌ Camera restart error: {e}")
    
    def run(self):
        """Run the live camera system"""
        print("🚀 Starting Live Camera Face & Object Detector...")
        self.create_gui()

if __name__ == "__main__":
    detector = LiveCameraDetector()
    detector.run()
