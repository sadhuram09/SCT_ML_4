import tensorflow as tf
import numpy as np
import cv2
import os

MODELS_DIR = r'C:\Users\USER\OneDrive\Desktop\hand_gesture_recognition\models'
model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'final_model.h5'))

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {img_path}")
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 1)  # Shape for prediction
    return img

def predict_gesture(img_path):
    img = preprocess_image(img_path)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    return class_idx, confidence

if __name__ == "__main__":
    # Replace this path with an actual image location on your system
    test_img_path = r'C:\Users\USER\OneDrive\Desktop\hand_gesture_recognition\src\image.jpg'
    
    try:
        pred_class, conf = predict_gesture(test_img_path)
        print(f"Predicted class: {pred_class}, Confidence: {conf:.2f}")
    except Exception as e:
        print("Error:", e)
