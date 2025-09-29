import os, shutil, json, threading
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
import numpy as np
from hf_infer import image_caption_hf
from PIL import Image, ImageStat

try:
    from face_detect_fallback import FaceRecognizer
except ImportError:
    from face_detect_fallback import FaceRecognizer

MODEL_PATH = "../image_recogniser_model.h5"
TRAIN_DIR = "../data/train"
LABELS_PATH = "../labels.json"
IMG_SIZE = (224,224)

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r') as f:
        existing_classes = json.load(f)
else:
    existing_classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR,d))]

os.makedirs(os.path.join(TRAIN_DIR, "unknown"), exist_ok=True)

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def save_image(img_path, class_name):
    folder = os.path.join(TRAIN_DIR, class_name)
    os.makedirs(folder, exist_ok=True)
    idx = len(os.listdir(folder)) + 1
    ext = os.path.splitext(img_path)[1]
    new_name = f"{class_name}{idx}{ext}"
    new_path = os.path.join(folder, new_name)
    shutil.copy(img_path, new_path)
    return new_name, new_path

def predict_image(img_path):
    global model
    if model is None: return None, 0.0
    img = preprocess_image(img_path, target_size=IMG_SIZE)
    preds = model.predict(img)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    return existing_classes[idx] if idx < len(existing_classes) else None, conf

def overview(img_path):
    try:
        im = Image.open(img_path).convert("RGB")
        w,h = im.size
        avg = tuple(int(x) for x in ImageStat.Stat(im).mean)
        aspect = round(w/h, 2)
        caption = image_caption_hf(img_path)
        return {"size": (w,h), "aspect": aspect, "avg_color": avg, "caption": caption}
    except Exception as e:
        return {"error": str(e)}

# === GUI ===
root = tk.Tk()
root.title("Image Recogniser Advanced (Sci-Fi)")
root.geometry("600x400")

# Dark sci-fi style
root.configure(bg="#0A0F1E")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton",
                font=("Consolas", 12, "bold"),
                foreground="#00FFCC",
                background="#101826",
                borderwidth=2,
                focusthickness=3,
                focuscolor="none")
style.map("TButton",
          background=[("active","#1E2A40")],
          foreground=[("active","#00FFFF")])

# Sci-fi text area
txt = tk.Text(root, height=12, width=70, bg="#101826", fg="#00FFCC",
              insertbackground="#00FFCC", font=("Consolas", 10))
txt.pack(pady=10)

def handle(img_path):
    pred, conf = predict_image(img_path)
    if pred and conf >= 0.6:
        name, path = save_image(img_path, pred)
        ov = overview(path)
        txt.insert(tk.END, f"Predicted {pred} ({conf:.2f})\nSaved as {name}\n{ov}\n\n")
    else:
        user_label = simpledialog.askstring("Label Image", "Uncertain. Enter class:")
        if not user_label: user_label = "unknown"
        if user_label not in existing_classes:
            existing_classes.append(user_label)
            with open(LABELS_PATH,'w') as f: json.dump(existing_classes,f)
        name, path = save_image(img_path, user_label)
        ov = overview(path)
        txt.insert(tk.END, f"Labeled {user_label}\nSaved as {name}\n{ov}\n\n")

def upload():
    path = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg")])
    if path: threading.Thread(target=handle, args=(path,), daemon=True).start()

upload_btn = ttk.Button(root, text="🚀 Upload & Predict", command=upload)
upload_btn.pack(pady=5)

def retrain():
    txt.insert(tk.END, "⚙️ Retraining...\n")
    os.system("python train.py")
    global model, existing_classes
    model = load_model(MODEL_PATH)
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f: existing_classes = json.load(f)
    txt.insert(tk.END, "✅ Retraining done.\n")

retrain_btn = ttk.Button(root, text="🔄 Retrain Model",
                         command=lambda: threading.Thread(target=retrain,daemon=True).start())
retrain_btn.pack(pady=5)

root.mainloop()
