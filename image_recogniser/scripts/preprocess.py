from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path, target_size=(224,224), color_mode='rgb'):
    """
    Preprocess image for prediction or training.
    
    Args:
        img_path (str): Path to image.
        target_size (tuple): Target size for resizing (default 224x224).
        color_mode (str): 'rgb' or 'grayscale'. Default 'rgb'.
        
    Returns:
        np.array: Preprocessed image ready for model.predict(), shape (1, H, W, C)
    """
    # Load image with color mode
    img = image.load_img(img_path, target_size=target_size, color_mode=color_mode)
    
    # Convert to array and normalize
    img_array = image.img_to_array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
