from tensorflow.keras.models import load_model
from preprocess import preprocess_image
import os
import numpy as np

# 1️⃣ Load trained model
model = load_model("../image_recogniser_model.h5")

# 2️⃣ Class labels (same order as train_generator)
class_labels = ['bhains', 'bottle', 'laptop', 'otherhumans', 'phone']

# 3️⃣ Test folder path
test_dir = "../data/test"

# 4️⃣ Test loop
total = 0
correct = 0

for category in os.listdir(test_dir):
    cat_path = os.path.join(test_dir, category)
    for img_file in os.listdir(cat_path):
        img_path = os.path.join(cat_path, img_file)
        img = preprocess_image(img_path)
        pred = model.predict(img)
        pred_class = class_labels[np.argmax(pred)]
        total += 1
        if pred_class == category:
            correct += 1
        print(f"{img_file}: Actual={category}, Predicted={pred_class}")

# 5️⃣ Overall accuracy
accuracy = correct / total
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
