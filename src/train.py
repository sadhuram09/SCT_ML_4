import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model
from visualize import plot_training_history
import os

# Paths
PROCESSED_DATA_DIR = r'C:\Users\USER\OneDrive\Desktop\hand_gesture_recognition\data\processed'
MODELS_DIR = r'C:\Users\USER\OneDrive\Desktop\hand_gesture_recognition\models'

os.makedirs(MODELS_DIR, exist_ok=True)

# Load preprocessed data
X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))

# Determine number of classes and input shape
num_classes = len(np.unique(y_train))
input_shape = X_train.shape[1:]  # e.g. (64, 64, 1)

# Build the CNN model
model = build_model(input_shape=input_shape, num_classes=num_classes)

# Setup callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=os.path.join(MODELS_DIR, 'best_model.h5'), save_best_only=True)
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save final model
model.save(os.path.join(MODELS_DIR, 'final_model.h5'))

print("Model training complete and saved.")

# Plot training history
plot_training_history(history)
