import tensorflow as tf
from tensorflow.keras import layers, models, applications


def _build_simple_cnn(input_shape, num_classes):
    """Builds a basic, small CNN. Good for simple datasets."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # -- Feature Extraction Block 1 --
        # Conv2D: The core building block. It acts as a feature detector, learning to
        # recognize simple patterns like edges, corners, and textures.
        layers.Conv2D(32, (3, 3), activation='relu'),
        # MaxPooling2D: This layer downsamples the image, making it smaller. It reduces
        # computational cost and helps the model generalize by focusing on the most
        # important features in a region.
        layers.MaxPooling2D((2, 2)),

        # -- Feature Extraction Block 2 --
        # We add another block to learn more complex patterns by combining the
        # features learned in the previous block.
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # -- Classification Head --
        # Flatten: This layer takes the 2D grid of features from the convolutional
        # blocks and converts it into a single, long 1D vector.
        layers.Flatten(),
        # Dense: A standard, fully-connected neural network layer. It takes the
        # flattened features and learns how to combine them to make a final decision.
        layers.Dense(64, activation='relu'),
        # Output Layer: A final Dense layer with a neuron for each class. It outputs
        # the raw scores (logits) for each possible class.
        layers.Dense(num_classes)
    ])
    print(" Successfully built a 'Simple CNN' model.")
    return model

def _build_from_keras_applications(model_name, input_shape, num_classes, weights):
    """Builds a model using a powerful, pre-trained backbone."""
    MODEL_APPLICATIONS = {
        "MobileNetV2": applications.MobileNetV2, "ResNet50": applications.ResNet50,
        "EfficientNetB0": applications.EfficientNetB0, "VGG16": applications.VGG16,
    }
    # 1. Load the Base Model: This is the powerful, pre-trained "engine" (e.g., ResNet50).
    # `include_top=False` means we are only loading the feature extraction part,
    # not its original classification layer for 1000 ImageNet classes.
    base_model = MODEL_APPLICATIONS[model_name](
        input_shape=input_shape, include_top=False, weights=weights
    )

    # 2. Freeze the Base Model: We lock the weights of the base model so they don't
    # get changed during the initial training. We want to keep all the powerful
    # knowledge it learned from the ImageNet dataset.
    base_model.trainable = False

    # 3. Create a new classification head for our specific task.
    model = models.Sequential([
        base_model, # The frozen feature extraction engine
        
        # GlobalAveragePooling2D: A modern and efficient alternative to Flatten.
        # It takes the average of each feature map, creating a compact feature vector.
        layers.GlobalAveragePooling2D(),
        
        # Dropout: A regularization technique. During training, it randomly sets a fraction
        # of its input units to 0, which helps prevent overfitting.
        layers.Dropout(0.2),
        
        # Output Layer: Our new, trainable classification layer that learns to map
        # the extracted features to our specific number of classes.
        layers.Dense(num_classes)
    ])
    print(f" Successfully created transfer learning model '{model_name}'.")
    return model




def get_model(model_name: str, input_shape: tuple, num_classes: int, weights: str = "imagenet"):

    # List of models that are built entirely from scratch within this file
    CUSTOM_MODELS = {
        "simple_cnn": lambda: _build_simple_cnn(input_shape, num_classes),
    }

    if model_name in CUSTOM_MODELS:
        # If the user asks for a 'simple_cnn' or 'vit_small', call the corresponding
        # builder function.
        return CUSTOM_MODELS[model_name]()
    else:
        # Otherwise, assume the user is asking for a standard model from Keras
        # Applications (like 'MobileNetV2') for transfer learning.
        try:
            return _build_from_keras_applications(model_name, input_shape, num_classes, weights)
        except KeyError:
            # If the name is not a custom model and not a valid Keras model, raise an error.
            raise ValueError(
                f"Unknown model_name: '{model_name}'. "
                f"Choose from {list(CUSTOM_MODELS.keys())} or a valid Keras Applications model."
            )

