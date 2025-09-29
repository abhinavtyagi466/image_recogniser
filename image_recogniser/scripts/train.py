import os
import json
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_cnn_transfer
from ultralytics import YOLO
import shutil
from data_mediator import DataMediator, prepare_retraining, get_training_stats

def train_cnn_model():
    """Train CNN model for general classification"""
    print("=" * 50)
    print("TRAINING CNN MODEL")
    print("=" * 50)
    
    train_dir = '../data/train'
    test_dir  = '../data/test'
    model_path = "../models/cnn_classifier.h5"
    labels_path = "../labels.json"

    IMG_SIZE = (224, 224)
    BATCH = 16
    EPOCHS_FULL = 10

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode='categorical',
        shuffle=True
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode='categorical',
        shuffle=False
    )

    classes = list(train_generator.class_indices.keys())
    num_classes = len(classes)
    print("Detected classes:", classes)

    with open(labels_path, 'w') as f:
        json.dump(classes, f)

    model = build_cnn_transfer(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes, base_trainable=False)
    model.fit(train_generator, epochs=EPOCHS_FULL, validation_data=test_generator)

    model.save(model_path)
    print("Saved CNN model to", model_path)

def prepare_yolo_dataset():
    """Prepare YOLO dataset format from existing data"""
    print("=" * 50)
    print("PREPARING YOLO DATASET")
    print("=" * 50)
    
    # Create YOLO dataset structure
    yolo_dir = "../data/yolo_dataset"
    os.makedirs(f"{yolo_dir}/images/train", exist_ok=True)
    os.makedirs(f"{yolo_dir}/images/val", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/val", exist_ok=True)
    
    # Get class names from existing data
    train_dir = '../data/train'
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes = [c for c in classes if c not in ['faces', 'products']]  # Exclude special folders
    
    # Create class mapping
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    # Process training images
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_id = class_to_id[class_name]
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Copy image
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(f"{yolo_dir}/images/train", img_file)
                shutil.copy2(src_path, dst_path)
                
                # Create label file (assuming full image is the object)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(f"{yolo_dir}/labels/train", label_file)
                
                with open(label_path, 'w') as f:
                    # YOLO format: class_id center_x center_y width height (normalized)
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    # Process test images for validation
    test_dir = '../data/test'
    if os.path.exists(test_dir):
        for class_name in classes:
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_id = class_to_id[class_name]
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Copy image
                    src_path = os.path.join(class_dir, img_file)
                    dst_path = os.path.join(f"{yolo_dir}/images/val", img_file)
                    shutil.copy2(src_path, dst_path)
                    
                    # Create label file
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(f"{yolo_dir}/labels/val", label_file)
                    
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath(yolo_dir)}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""
    
    with open(f"{yolo_dir}/dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"YOLO dataset prepared at: {yolo_dir}")
    print(f"Classes: {classes}")
    return f"{yolo_dir}/dataset.yaml"

def train_yolo_model(dataset_yaml=None, epochs=50):
    """Train YOLO model for object detection"""
    print("=" * 50)
    print("TRAINING YOLO MODEL")
    print("=" * 50)
    
    if dataset_yaml is None:
        dataset_yaml = prepare_yolo_dataset()
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # Start with nano version
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        name='yolo_products',
        project='../models',
        save=True,
        plots=True
    )
    
    # Save the trained model
    model_path = "../models/yolo_products.pt"
    model.save(model_path)
    print(f"Saved YOLO model to {model_path}")
    
    return model_path

def retrain_with_new_data():
    """Retrain models with new user data"""
    print("🔄 RETRAINING WITH NEW DATA")
    print("=" * 50)
    
    # Get current training stats
    stats = get_training_stats()
    print(f"📊 Current training data:")
    print(f"   - Total categories: {stats['total_categories']}")
    print(f"   - Total images: {stats['total_images']}")
    print(f"   - Last updated: {stats['last_updated']}")
    
    # Prepare data for retraining
    if prepare_retraining():
        print("✅ Data preparation complete")
        
        # Train models with new data
        print("\n🚀 Starting model retraining...")
        
        # Train CNN model
        print("\n1️⃣ Training CNN model...")
        train_cnn_model()
        
        # Train YOLO model
        print("\n2️⃣ Training YOLO model...")
        dataset_yaml = prepare_yolo_dataset()
        train_yolo_model(dataset_yaml, epochs=30)  # Fewer epochs for retraining
        
        print("\n✅ RETRAINING COMPLETED!")
        print("Models updated with new data")
        
        # Show final stats
        final_stats = get_training_stats()
        print(f"\n📊 Final training data:")
        print(f"   - Total categories: {final_stats['total_categories']}")
        print(f"   - Total images: {final_stats['total_images']}")
        
    else:
        print("❌ Data preparation failed")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train image recognition models')
    parser.add_argument('--model', choices=['cnn', 'yolo', 'both'], default='both',
                       help='Which model to train (default: both)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for YOLO training (default: 50)')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain with new user data')
    parser.add_argument('--stats', action='store_true',
                       help='Show training data statistics')
    
    args = parser.parse_args()
    
    if args.stats:
        print("📊 TRAINING DATA STATISTICS")
        print("=" * 50)
        stats = get_training_stats()
        print(json.dumps(stats, indent=2))
        return
    
    if args.retrain:
        retrain_with_new_data()
        return
    
    print("🚀 ADVANCED IMAGE RECOGNIZER TRAINING")
    print("=" * 60)
    
    if args.model in ['cnn', 'both']:
        train_cnn_model()
    
    if args.model in ['yolo', 'both']:
        dataset_yaml = prepare_yolo_dataset()
        train_yolo_model(dataset_yaml, args.epochs)
    
    print("=" * 60)
    print("✅ TRAINING COMPLETED!")
    print("Models saved in ../models/ directory")

if __name__ == "__main__":
    main()