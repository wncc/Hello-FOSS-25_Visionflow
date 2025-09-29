import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets
import os

# Import only the functions we need now
from image_utils import (
    resize,
    normalize_img,
    preprocess_image # Import the main pipeline function
)

# --- Configuration ---
MODEL_PATH = 'cifar10_model.h5'

def evaluate_pipeline(model, x_test, y_test, description, **pipeline_kwargs):
    """
    Evaluates the model's performance using the full preprocess_image pipeline.
    
    Args:
        model: The trained Keras model.
        x_test: Original test images.
        y_test: Original test labels.
        description (str): A description of the pipeline being tested.
        **pipeline_kwargs: The arguments to pass to the preprocess_image function.
    """
    print(f"\n--- Evaluating Pipeline: {description} ---")

    # The preprocess_image function handles resizing and normalization internally.
    # We must ensure the final output is normalized for the model.
    pipeline_kwargs['normalize'] = True
    pipeline_kwargs['size'] = (64, 64)
    
    processed_images = [preprocess_image(img, **pipeline_kwargs) for img in x_test]

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
    # We still need resize and normalize_img for the baseline comparison
    x_test_baseline = np.array([normalize_img(resize(img, 64, 64), 0, 1) for img in x_test])
    baseline_loss, baseline_acc = model.evaluate(x_test_baseline, y_test, verbose=0)
    print(f"  -> Baseline Accuracy: {baseline_acc:.4f}")

    # --- 3. Evaluate Full preprocess_image Pipelines ---
    # This section now contains all our evaluation tests.
    evaluate_pipeline(
        model, x_test, y_test,
        description="Just Augmentation",
        pipeline_kwargs={'augment': True}
    )

    evaluate_pipeline(
        model, x_test, y_test,
        description="Just Grayscale",
        pipeline_kwargs={'to_grayscale': True}
    )

    evaluate_pipeline(
        model, x_test, y_test,
        description="Just Median Filter",
        pipeline_kwargs={'noise_reduction': 'median'}
    )

    evaluate_pipeline(
        model, x_test, y_test,
        description="Grayscale with Median Filter",
        pipeline_kwargs={'to_grayscale': True, 'noise_reduction': 'median'}
    )
    
    evaluate_pipeline(
        model, x_test, y_test,
        description="Gaussian Blur with Augmentation",
        pipeline_kwargs={'noise_reduction': 'gaussian', 'augment': True}
    )


if __name__ == '__main__':
    main()

