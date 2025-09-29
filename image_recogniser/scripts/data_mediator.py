import os
import json
import shutil
import pickle
from typing import Dict, List, Optional
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

class DataMediator:
    """
    Mediator class to transfer user data to training datasets
    and manage model retraining
    """
    
    def __init__(self, base_dir: str = "../data"):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, "train")
        self.test_dir = os.path.join(base_dir, "test")
        self.user_data_dir = os.path.join(base_dir, "user_input")
        self.face_encodings_dir = "../models/face_encodings"
        
        # Create directories if they don't exist
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.train_dir,
            self.test_dir,
            self.user_data_dir,
            self.face_encodings_dir,
            os.path.join(self.user_data_dir, "faces"),
            os.path.join(self.user_data_dir, "products"),
            os.path.join(self.user_data_dir, "text_data")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def add_user_face_data(self, name: str, image_path: str, description: str = "") -> bool:
        """
        Add user face data to training dataset
        
        Args:
            name: Person's name
            image_path: Path to face image
            description: Optional description
            
        Returns:
            True if successful
        """
        try:
            # Create person's directory in train folder
            person_train_dir = os.path.join(self.train_dir, name.lower().replace(" ", "_"))
            os.makedirs(person_train_dir, exist_ok=True)
            
            # Copy image to training directory
            ext = os.path.splitext(image_path)[1]
            existing_files = [f for f in os.listdir(person_train_dir) if f.endswith(ext)]
            filename = f"{name.lower().replace(' ', '_')}_{len(existing_files) + 1}{ext}"
            
            train_image_path = os.path.join(person_train_dir, filename)
            shutil.copy2(image_path, train_image_path)
            
            # Save metadata
            metadata = {
                "name": name,
                "original_path": image_path,
                "train_path": train_image_path,
                "description": description,
                "added_date": datetime.now().isoformat(),
                "type": "face"
            }
            
            self.save_metadata(name, metadata)
            
            print(f"✅ Added face data for {name} to training dataset")
            return True
            
        except Exception as e:
            print(f"❌ Error adding face data: {e}")
            return False
    
    def add_user_product_data(self, product_name: str, image_path: str, description: str = "") -> bool:
        """
        Add user product data to training dataset
        
        Args:
            product_name: Name of the product
            image_path: Path to product image
            description: Optional description
            
        Returns:
            True if successful
        """
        try:
            # Create product directory in train folder
            product_train_dir = os.path.join(self.train_dir, product_name.lower().replace(" ", "_"))
            os.makedirs(product_train_dir, exist_ok=True)
            
            # Copy image to training directory
            ext = os.path.splitext(image_path)[1]
            existing_files = [f for f in os.listdir(product_train_dir) if f.endswith(ext)]
            filename = f"{product_name.lower().replace(' ', '_')}_{len(existing_files) + 1}{ext}"
            
            train_image_path = os.path.join(product_train_dir, filename)
            shutil.copy2(image_path, train_image_path)
            
            # Save metadata
            metadata = {
                "product_name": product_name,
                "original_path": image_path,
                "train_path": train_image_path,
                "description": description,
                "added_date": datetime.now().isoformat(),
                "type": "product"
            }
            
            self.save_metadata(product_name, metadata)
            
            print(f"✅ Added product data for {product_name} to training dataset")
            return True
            
        except Exception as e:
            print(f"❌ Error adding product data: {e}")
            return False
    
    def add_text_data(self, text: str, category: str = "general") -> bool:
        """
        Add text data for training
        
        Args:
            text: Text content
            category: Category of the text
            
        Returns:
            True if successful
        """
        try:
            text_data = {
                "text": text,
                "category": category,
                "added_date": datetime.now().isoformat(),
                "type": "text"
            }
            
            # Save to text data directory
            text_file = os.path.join(self.user_data_dir, "text_data", f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(text_file, 'w', encoding='utf-8') as f:
                json.dump(text_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Added text data to category: {category}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding text data: {e}")
            return False
    
    def save_metadata(self, name: str, metadata: Dict):
        """Save metadata for tracking"""
        metadata_file = os.path.join(self.user_data_dir, "metadata.json")
        
        # Load existing metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = []
        
        # Add new metadata
        all_metadata.append(metadata)
        
        # Save updated metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    def get_training_stats(self) -> Dict:
        """Get statistics about training data"""
        stats = {
            "total_categories": 0,
            "total_images": 0,
            "categories": {},
            "last_updated": None
        }
        
        try:
            # Count training data
            for category in os.listdir(self.train_dir):
                category_path = os.path.join(self.train_dir, category)
                if os.path.isdir(category_path):
                    image_count = len([f for f in os.listdir(category_path) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
                    
                    stats["categories"][category] = image_count
                    stats["total_images"] += image_count
                    stats["total_categories"] += 1
            
            # Get last updated time
            if os.path.exists(os.path.join(self.user_data_dir, "metadata.json")):
                with open(os.path.join(self.user_data_dir, "metadata.json"), 'r') as f:
                    metadata = json.load(f)
                    if metadata:
                        stats["last_updated"] = metadata[-1]["added_date"]
            
        except Exception as e:
            print(f"Error getting training stats: {e}")
        
        return stats
    
    def prepare_for_retraining(self) -> bool:
        """
        Prepare data for retraining and update face encodings
        
        Returns:
            True if successful
        """
        try:
            print("🔄 Preparing data for retraining...")
            
            # Update face encodings if face recognition is available
            try:
                from face_detect_fallback import FaceRecognizer
                recognizer = FaceRecognizer()
                
                # Re-register all faces from training data
                for category in os.listdir(self.train_dir):
                    category_path = os.path.join(self.train_dir, category)
                    if os.path.isdir(category_path):
                        for image_file in os.listdir(category_path):
                            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                image_path = os.path.join(category_path, image_file)
                                recognizer.register_face(category, image_path)
                
                print("✅ Face encodings updated")
                
            except ImportError:
                print("⚠️ Face recognition not available - skipping face encoding update")
            
            # Create training summary
            stats = self.get_training_stats()
            summary_file = os.path.join(self.user_data_dir, "training_summary.json")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            print("✅ Data preparation complete")
            print(f"📊 Training data summary:")
            print(f"   - Total categories: {stats['total_categories']}")
            print(f"   - Total images: {stats['total_images']}")
            print(f"   - Last updated: {stats['last_updated']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error preparing for retraining: {e}")
            return False
    
    def cleanup_old_data(self, days_old: int = 30) -> bool:
        """
        Clean up old user data
        
        Args:
            days_old: Remove data older than this many days
            
        Returns:
            True if successful
        """
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            removed_count = 0
            
            # Clean up old text data
            text_dir = os.path.join(self.user_data_dir, "text_data")
            if os.path.exists(text_dir):
                for file in os.listdir(text_dir):
                    file_path = os.path.join(text_dir, file)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_date:
                            os.remove(file_path)
                            removed_count += 1
            
            print(f"✅ Cleaned up {removed_count} old files")
            return True
            
        except Exception as e:
            print(f"❌ Error cleaning up old data: {e}")
            return False

# Convenience functions
def add_face_data(name: str, image_path: str, description: str = "") -> bool:
    """Add face data to training dataset"""
    mediator = DataMediator()
    return mediator.add_user_face_data(name, image_path, description)

def add_product_data(product_name: str, image_path: str, description: str = "") -> bool:
    """Add product data to training dataset"""
    mediator = DataMediator()
    return mediator.add_user_product_data(product_name, image_path, description)

def add_text_data(text: str, category: str = "general") -> bool:
    """Add text data for training"""
    mediator = DataMediator()
    return mediator.add_text_data(text, category)

def prepare_retraining() -> bool:
    """Prepare data for retraining"""
    mediator = DataMediator()
    return mediator.prepare_for_retraining()

def get_training_stats() -> Dict:
    """Get training data statistics"""
    mediator = DataMediator()
    return mediator.get_training_stats()

if __name__ == "__main__":
    # Test the data mediator
    mediator = DataMediator()
    
    print("📊 Current Training Data Statistics:")
    stats = mediator.get_training_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n🔄 Preparing for retraining...")
    mediator.prepare_for_retraining()
