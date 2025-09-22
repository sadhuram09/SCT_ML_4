import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Paths with leapGestRecog included as subfolder
RAW_DATA_DIR = r'C:\Users\USER\OneDrive\Desktop\hand_gesture_recognition\data\raw\leapGestRecog'
PROCESSED_DATA_DIR = r'C:\Users\USER\OneDrive\Desktop\hand_gesture_recognition\data\processed'
IMG_SIZE = 64  # Resize images to 64x64

def preprocess_and_save():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    images = []
    labels = []
    class_names = sorted(os.listdir(RAW_DATA_DIR))  # Get sorted gesture class folders

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_folder):
            continue
        # Recursively walk through all subfolders to find images
        for root, dirs, files in os.walk(class_folder):
            for img_file in files:
                img_path = os.path.join(root, img_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0  # Normalize to [0, 1]
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Save processed numpy arrays
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)

    print(f"Data preprocessing complete. Processed files saved in {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    preprocess_and_save()
