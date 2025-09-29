# visionflow/model_zoo.py

import tensorflow as tf
from tensorflow.keras import layers, models, applications

def get_model(model_name: str, input_shape: tuple, num_classes: int, weights: str = "imagenet"):
    """
    Creates a pre-trained model from Keras Applications with a new classification head.

    This function supports transfer learning by loading a well-known architecture,
    freezing its layers, and adding a new trainable output layer for your specific task.

    Args:
        model_name (str): The name of the model architecture (e.g., "MobileNetV2", "ResNet50").
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes for the new classification head.
        weights (str): Weights to load for the base model. Can be "imagenet" (pre-trained) 
                       or None (random initialization).

    Returns:
        tf.keras.Model: The constructed Keras model ready for training.
        
    Raises:
        ValueError: If the provided model_name is not supported.
    """
    # A dictionary mapping user-friendly names to their Keras Application classes
    MODEL_APPLICATIONS = {
        "MobileNetV2": applications.MobileNetV2,
        "ResNet50": applications.ResNet50,
        "EfficientNetB0": applications.EfficientNetB0,
        "VGG16": applications.VGG16,
        "InceptionV3": applications.InceptionV3,
    }

    if model_name not in MODEL_APPLICATIONS:
        raise ValueError(f"Unknown model name: '{model_name}'. "
                         f"Available models are: {list(MODEL_APPLICATIONS.keys())}")

    # 1. Load the base model from tf.keras.applications without the top classification layer.
    base_model = MODEL_APPLICATIONS[model_name](
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )

    # 2. Freeze the base model's layers. This is the key to transfer learning, as it
    # prevents the pre-trained weights from being modified during initial training.
    base_model.trainable = False

    # 3. Create a new model on top by adding a custom classification head.
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Flattens the feature maps to a vector
        layers.Dropout(0.2),               # Adds regularization to prevent overfitting
        layers.Dense(num_classes)          # The final output layer for our specific number of classes
    ])

    print(f"✅ Successfully created model '{model_name}' with {num_classes} classes.")
    return model

