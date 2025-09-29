import tensorflow as tf
import cv2
import numpy as np
import os

# ==============================================================================
# All of your custom, from-scratch image processing functions
# ==============================================================================

def load_image(path):
    img = cv2.imread(path)
    # OpenCV loads in BGR, convert to RGB for consistency with other libraries
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize(img, new_height, new_width):
    # This is your slow implementation. As you noted, it should be optimized later.
    resized_image = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
    original_height, original_width = img.shape[:2]
    height_ratio = original_height / new_height
    width_ratio = original_width / new_width
    for i in range(new_height):
        for j in range(new_width):
            x = int(i * height_ratio)
            y = int(j * width_ratio)
            resized_image[i, j] = img[x, y]
    return resized_image

def grayscale(img):
    height, width = img.shape[:2]
    gray_img = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            # Using standard RGB channel order
            R, G, B = img[i, j]
            gray_value = 0.2989 * R + 0.5870 * G + 0.1140 * B
            gray_img[i, j] = gray_value
    return gray_img

def normalize_img(img, new_min=0.0, new_max=1.0):
    old_min, old_max = img.min(), img.max()
    img_norm = (img - old_min) * (new_max - new_min) / (old_max - old_min + 1e-8) + new_min
    return img_norm.astype(np.float32)

def median_filter(img, ksize=3):
    # This function expects a 2D or 3D image and will be slow with loops
    pad = ksize // 2
    padded_img = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+ksize, j:j+ksize]
            # Median for each channel
            for c in range(img.shape[2]):
                 out[i, j, c] = np.median(region[:,:,c])
    return out.astype(img.dtype)

# ... (and so on for all your other custom functions like flip, rotate, etc.)
def flip_horizontal(img):
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
