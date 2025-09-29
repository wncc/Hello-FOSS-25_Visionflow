import tensorflow as tf
import os
import numpy as np

# ==============================================================================
# CORE DATASET CREATION PIPELINE
# This is the main entry point for your pipeline.py
# ==============================================================================

def create_dataset_from_directory(
    data_path: str,
    img_size: tuple,
    batch_size: int,
    augment: bool = True
):
    """
    Builds a complete, performance-optimized tf.data.Dataset pipeline.

    This function finds all images in the directory, loads them, applies
    preprocessing and optional augmentation, and batches them for training.

    Args:
        data_path (str): Path to the dataset directory.
        img_size (tuple): Target image size (height, width).
        batch_size (int): Batch size for training.
        augment (bool): If True, applies random augmentations to the training data.

    Returns:
        tf.data.Dataset: A configured dataset ready for model.fit().
    """
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    
    # Create a dataset of all file paths and their corresponding labels
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    # --- Create the preprocessing function to be mapped ---
    def preprocess_fn(image, label):
        # Normalize pixel values to the [0, 1] range.
        # This is a standard and crucial step for most models.
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Apply the preprocessing function to every image in the dataset
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # --- Apply augmentation if specified ---
    if augment:
        augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        # Note: Augmentation is applied AFTER batching for performance
        dataset = dataset.map(lambda x, y: (augmentation_layers(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    # --- Optimize for performance ---
    # Prefetching overlaps data preprocessing and model execution, improving throughput.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"✅ Dataset created from '{data_path}' with batch size {batch_size}.")
    print(f"   - Augmentation enabled: {augment}")
    return dataset


