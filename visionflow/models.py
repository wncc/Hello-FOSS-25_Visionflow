import tensorflow as tf
from tensorflow.keras import layers, models, applications

# ==============================================================================
# Helper Components for Vision Transformer (ViT)
# These are custom layers that define the building blocks of a Transformer.
# ==============================================================================

class Patches(layers.Layer):
    """
    This layer acts like a pair of scissors. It takes a full image and
    chops it up into a grid of smaller, non-overlapping square patches.
    For example, a 128x128 image with a patch size of 8 would be cut into
    (128/8) * (128/8) = 16 * 16 = 256 patches.
    
    """
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        # Get the batch size from the input tensor's shape
        batch_size = tf.shape(images)[0]
        # Use a built-in TensorFlow function to efficiently extract patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Reshape the patches into a sequence for the Transformer
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    """
    This layer teaches the model *where* each patch came from.
    After chopping up the image, we lose the original spatial information.
    This layer adds a unique "positional number" (embedding) to each patch
    so the model knows if a patch was from the top-left corner or the center.
    """
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        # A Dense layer to project the patch data into a consistent vector size
        self.projection = layers.Dense(units=projection_dim)
        # The embedding layer that stores the unique "positional number" for each patch
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        # Create a tensor representing the position of each patch (0, 1, 2, ...)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # Add the positional embedding to the patch data
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# ==============================================================================
# Private Model Builder Functions
# These functions define the specific architectures for each model type.
# ==============================================================================

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
    print("✅ Successfully built a 'Simple CNN' model.")
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
    print(f"✅ Successfully created transfer learning model '{model_name}'.")
    return model

def _build_vit(input_shape, num_classes, patch_size=8, projection_dim=64, num_heads=4, transformer_layers=4):
    """Builds a small Vision Transformer model for advanced pattern recognition."""
    num_patches = (input_shape[0] // patch_size) ** 2
    inputs = layers.Input(shape=input_shape)
    
    # 1. Create patches from the input image.
    patches = Patches(patch_size)(inputs)
    
    # 2. Encode patches with positional information.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # 3. Create the Transformer Encoder Blocks. This is the core of the ViT.
    for _ in range(transformer_layers):
        # LayerNormalization: Stabilizes training by normalizing the inputs to each block.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # MultiHeadAttention: The key layer. It allows the model to weigh the importance
        # of every patch in relation to every other patch. It learns the context and
        # relationships between different parts of the image.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip Connection: Adds the output back to the original input. This helps
        # prevent gradients from vanishing during training in deep networks.
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Feed-Forward Network part of the Transformer block.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(projection_dim * 2, activation="relu")(x3)
        x3 = layers.Dense(projection_dim)(x3)
        
        # Second skip connection.
        encoded_patches = layers.Add()([x3, x2])

    # 4. Create the final classification head.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Final Dense layer to output the class logits.
    logits = layers.Dense(num_classes)(representation)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    print("✅ Successfully built a 'Vision Transformer' model.")
    return model

# ==============================================================================
# Main get_model Factory Function (This function decides which builder to call)
# ==============================================================================

def get_model(model_name: str, input_shape: tuple, num_classes: int, weights: str = "imagenet"):
    """
    Acts as a master factory for creating different types of models based on the
    `model_name` parameter provided by the user.
    """
    # List of models that are built entirely from scratch within this file
    CUSTOM_MODELS = {
        "simple_cnn": lambda: _build_simple_cnn(input_shape, num_classes),
        "vit_small": lambda: _build_vit(input_shape, num_classes),
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

