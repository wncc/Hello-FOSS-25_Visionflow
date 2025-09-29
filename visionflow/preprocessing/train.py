import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os


from preprocess.py import preprocess_image

# --- Constants and Configuration ---
MODEL_SAVE_PATH = 'cifar10_model.h5'
BATCH_SIZE = 64
EPOCHS = 10 # For a real training, you might want 20-50 epochs

def build_model(input_shape=(64, 64, 3), num_classes=10):
    """Builds a simple Convolutional Neural Network."""
    model = models.Sequential([
        # We expect input images of size 64x64 with 3 color channels
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes) # No softmax needed due to from_logits=True
    ])
    return model

def main():
    """Main function to run the training pipeline."""
    # --- 1. Load the CIFAR-10 Dataset ---
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    print(f"Data loaded. Training images: {x_train.shape[0]}, Test images: {x_test.shape[0]}")

    # --- 2. Preprocess the Data using your pipeline ---
    # We will apply a standard pipeline for training: resize and normalize.
    # We will also use data augmentation to make the model more robust.
    print("Preprocessing training data with augmentation...")
    # Using a list comprehension to apply the function to all images
    x_train_processed = np.array([
        preprocess_image(img, size=(64, 64), normalize=True, augment=True)
        for img in x_train
    ])

    print("Preprocessing test data...")
    x_test_processed = np.array([
        preprocess_image(img, size=(64, 64), normalize=True, augment=False) # No augmentation for test set
        for img in x_test
    ])
    print(f"Data preprocessed. New shape: {x_train_processed.shape}")

    # --- 3. Build and Compile the Model ---
    print("Building the CNN model...")
    model = build_model()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    # --- 4. Train the Model ---
    print("\nStarting model training...")
    history = model.fit(
        x_train_processed, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test_processed, y_test)
    )

    # --- 5. Save the Trained Model ---
    print(f"\nTraining complete. Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully!")

if __name__ == '__main__':
    # Ensure the script is being run directly
    main()
