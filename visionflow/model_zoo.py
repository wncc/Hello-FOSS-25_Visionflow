# visionflow/model_zoo.py

import tensorflow as tf
from tensorflow.keras import layers, models, applications

def get_model(model_name: str, input_shape: tuple, num_classes: int, weights: str = "imagenet"):

    """
    Choosing a pre-trained model from Keras Applications with a new classfication head.
    Returns tf.Keras.Model
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

    print(f"Successfully created model '{model_name}' with {num_classes} classes.")
    return model

