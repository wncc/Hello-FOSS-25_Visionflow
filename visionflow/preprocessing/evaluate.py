import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
import os

# Import all your custom preprocessing functions from the other file
from preprocess.py import (
    load_image,
    resize,
    grayscale,
    normalize_img,
    median_filter,
    gaussian_blur,
    flip_horizontal,
    adjust_brightness,
    adjust_contrast,
    add_gaussian_noise,
    add_salt_pepper
)

# --- Configuration ---
MODEL_PATH = 'cifar10_model.h5'

def evaluate_transformation(model, x_test, y_test, function, description, **kwargs):
    """
    Applies a specific transformation to the test set and evaluates the model's performance.

    Args:
        model: The trained Keras model.
        x_test: The original test images.
        y_test: The original test labels.
        function: The image processing function to apply.
        description (str): A description of the transformation for printing.
        **kwargs: Additional arguments to pass to the transformation function.
    """
    print(f"\n--- Evaluating: {description} ---")

    # Apply the transformation to a copy of the test images
    x_test_transformed = x_test.copy()
    
    # Process each image with the specified function
    processed_images = []
    for img in x_test_transformed:
        # The core logic: apply the function
        transformed_img = function(img, **kwargs)
        
        # We must always resize and normalize to match the model's expected input
        resized_img = resize(transformed_img, 64, 64)

        # Handle grayscale images that need 3 channels for the model
        if resized_img.ndim == 2:
            resized_img = np.stack([resized_img] * 3, axis=-1)
            
        normalized_img = normalize_img(resized_img, new_min=0, new_max=1) # Normalize to [0,1]
        processed_images.append(normalized_img)

    # Evaluate the model
    loss, accuracy = model.evaluate(np.array(processed_images), y_test, verbose=0)
    print(f"  -> Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    """Main function to run the evaluation."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please run train.py first to train and save the model.")
        return

    # --- 1. Load the Dataset and the Trained Model ---
    print("Loading test data and the pre-trained model...")
    (_, _), (x_test, y_test) = datasets.cifar10.load_data()
    model = load_model(MODEL_PATH)
    print("Model and data loaded.")

    # --- 2. Evaluate Baseline Performance ---
    # The baseline is performance on data processed exactly like the validation set in training
    print("\n--- Evaluating: Baseline (Resize + Normalize) ---")
    x_test_baseline = np.array([normalize_img(resize(img, 64, 64), 0, 1) for img in x_test])
    baseline_loss, baseline_acc = model.evaluate(x_test_baseline, y_test, verbose=0)
    print(f"  -> Baseline Accuracy: {baseline_acc:.4f}")

    # --- 3. Evaluate Each Transformation ---
    # Now, we test each of your functions individually.
    evaluate_transformation(model, x_test, y_test, grayscale, "Grayscale")
    evaluate_transformation(model, x_test, y_test, median_filter, "Median Filter", ksize=3)
    evaluate_transformation(model, x_test, y_test, gaussian_blur, "Gaussian Blur", sigma=1)
    evaluate_transformation(model, x_test, y_test, adjust_brightness, "Brightness (+50)", value=50)
    evaluate_transformation(model, x_test, y_test, adjust_contrast, "Contrast (x1.5)", factor=1.5)
    evaluate_transformation(model, x_test, y_test, add_gaussian_noise, "Gaussian Noise (sigma=25)", sigma=25)
    evaluate_transformation(model, x_test, y_test, add_salt_pepper, "Salt & Pepper Noise", prob=0.02)
    evaluate_transformation(model, x_test, y_test, flip_horizontal, "Horizontal Flip")

if __name__ == '__main__':
    main()
