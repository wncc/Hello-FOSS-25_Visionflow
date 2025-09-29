import tensorflow as tf
import cv2
import numpy as np
import os

# --- Optimized Helper Functions ---

def load_image(path):
    """Loads an image and converts it from BGR (OpenCV default) to RGB."""
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize(img, new_height, new_width, interpolation=cv2.INTER_LINEAR):
    """Efficiently resizes an image using OpenCV."""
    # Note: OpenCV's resize expects the size as a (width, height) tuple.
    return cv2.resize(img, (new_width, new_height), interpolation=interpolation)

def grayscale(img):
    """Efficiently converts an image to grayscale using OpenCV."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def normalize_img(img, new_min=0.0, new_max=1.0):
    """Efficiently normalizes an image using NumPy's vectorized operations."""
    old_min, old_max = img.min(), img.max()
    # Handle the edge case where max equals min to avoid division by zero
    if old_max == old_min:
        return np.full(img.shape, new_min, dtype=np.float32)
    
    img_norm = (img - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    return img_norm.astype(np.float32)

def median_filter(img, ksize=3):
    """Applies a median filter using OpenCV."""
    # Ensure the kernel size is odd, as required by the function
    if ksize % 2 == 0:
        raise ValueError("Kernel size for median filter must be odd.")
    
    # cv2.medianBlur handles each channel of a color image automatically.
    return cv2.medianBlur(img, ksize)

def flip_horizontal(img):
    """This function is already efficient as it uses NumPy's slicing."""
    return img[:, ::-1]

# ==============================================================================
# The New, Configurable Preprocessor Class
# ==============================================================================

class Preprocessor:
    """
    Applies a sequence of custom preprocessing steps defined in a config.
    """
    def __init__(self, config: dict):
        self.config = config

    def process(self, image_path_tensor):
        """
        The main processing function that will be wrapped by tf.py_function.
        It takes a file path, loads the image, and applies transformations.
        """
        # Decode the tensor to a string
        image_path = image_path_tensor.numpy().decode('utf-8')
        
        # Load the image using your function
        img = load_image(image_path)

        # Apply transformations sequentially based on the config
        if "resize" in self.config:
            params = self.config["resize"]
            img = resize(img, new_height=params["height"], new_width=params["width"])

        if self.config.get("grayscale", False):
            img = grayscale(img)
            # Ensure it still has 3 channels if the model expects it
            img = np.stack([img]*3, axis=-1)

        if "median_filter" in self.config:
            img = median_filter(img, ksize=self.config["median_filter"].get("ksize", 3))

        if "flip_horizontal" in self.config and self.config["flip_horizontal"]:
             if np.random.rand() > 0.5: # Apply augmentation randomly
                img = flip_horizontal(img)

        if self.config.get("normalize", False):
            img = normalize_img(img)

        return img.astype(np.float32)

# ==============================================================================
# The Main Dataset Creation Function (Updated)
# ==============================================================================

def create_dataset_from_directory(data_path: str, batch_size: int, preprocessor: Preprocessor):
    """
    Builds a tf.data.Dataset pipeline using your custom Preprocessor.
    """
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_map = {name: i for i, name in enumerate(class_names)}

    filepaths = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(data_path, class_name)
        for filename in os.listdir(class_dir):
            filepaths.append(os.path.join(class_dir, filename))
            labels.append(class_map[class_name])

    # 1. Create a dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.shuffle(buffer_size=len(filepaths))

    # 2. Use `tf.py_function` to wrap your custom processing logic
    def apply_custom_preprocessing(path, label):
        # Define the output shape and type for TensorFlow
        img_h = preprocessor.config["resize"]["height"]
        img_w = preprocessor.config["resize"]["width"]
        
        # The wrapped function takes a tensor, so we call the preprocessor's method
        image = tf.py_function(func=preprocessor.process, inp=[path], Tout=tf.float32)
        
        # Set the shape explicitly, which is required after py_function
        image.set_shape((img_h, img_w, 3))
        return image, label

    # 3. Map the wrapped function across the dataset
    dataset = dataset.map(apply_custom_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 4. Batch and prefetch for performance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"✅ Dataset created using your custom NumPy/OpenCV preprocessor.")
    return dataset
