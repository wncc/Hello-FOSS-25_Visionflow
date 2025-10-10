import tensorflow as tf
import cv2
import numpy as np
import os
from . import augmentation 


def load_image(path):
    img = cv2.imread(path)
    # OpenCV loads in BGR, convert to RGB for consistency with other library 
    # testung
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize(img, new_height, new_width):
    # This is very slow, needs to be optimized
    resized_image = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
    original_height, original_width = img.shape[:2]
    height_ratio = original_height / new_height
    width_ratio = original_width / new_width
    #coordinate grids for all pixels
    i_coord = np.arrange(new_height) #[0, 1, ..., new_height - 1]
    j_coord = np.arrange(new_width)

    x = (i_coord * height_ratio).astype(int)
    y = (j_coord * width_ratio).astype(int)

    resized_image = img[np.ix_(x, y)] # np.ix_ fits by indexing into the grid
    return resized_image

def grayscale(img):
    #Need to optimize
    height, width = img.shape[:2]
    gray_img = np.zeros((height, width), dtype=np.float32)

    # think of the constants as weights in a NN then,
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray_img = np.dot(img, weights)
    return gray_img.astype(img.dtype)

def normalize_img(img):
    img = img.astype(np.float32)
    return img / 255.0

#taking median of its surrounding k*k box and updating
def median_filter(img, ksize=3):
    pad = ksize // 2
    #Only takes rgb images
    #border edge cases are not handled
    #for the border cases issue_2, mode symmetric can be used instead of reflect 
    padded_img = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode='symmetric')
    '''
    Now its a CNN related problem where we look into strides.
    like a window sliding across all the dim, extract the subsets 
    '''
    from numpy.lib.stride_tricks import sliding_window_view as win_view
    windows = win_view(padded_img, (ksize, ksize, img.shape[2])) # img.shape[2] got through the third nested loop ka range
    out = np.median(windows, axis=(2, 3))
    return out.astype(img.dtype)

#creating gaussian kernel
def Gaussian_kernel(ksize, sigma):
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

#
def Gaussian_blur(img, sigma, ksize = 3):
    #Only takes rgb images
    #border edge cases are not handled
    kernel = Gaussian_kernel(ksize, sigma)
    pad = ksize // 2
    # standard problem can be handeled using scipy for convolution
    #issue_2 the opencv's reflect doesn't take edges while scipy's mode = reflect takes the whole thing so...
    padded_img = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode = 'reflect')
    from scipy.ndimage import convolve
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape(2)):
        #convolve our padded_image then crop it to original size
        convolved = convolve(padded_img[:, :, c], kernel, mode = 'constant', cval = 0)
        #extract centre region i.e. remove padding 
        out[:, :, c] = convolved[pad: -pad, pad: -pad]
    return out.astype(img.dtype)


class Preprocessor:
    # Let a user preprocess with custom configurations given by them 
    def __init__(self, config: dict):
        self.config = config

    def process(self, image_path_tensor):
        image_path = image_path_tensor.numpy().decode('utf-8')
        img = load_image(image_path)

        if "resize" in self.config:
            params = self.config["resize"]
            img = resize(img, new_height=params["height"], new_width=params["width"])

        if self.config.get("flip_horizontal", False) and np.random.rand() > 0.5:
            img = augmentation.flip_horizontal(img)
        
        if "adjust_brightness" in self.config:
            params = self.config["adjust_brightness"]
            value = np.random.randint(-params.get("value", 30), params.get("value", 30))
            img = augmentation.adjust_brightness(img, value=value)

        if "median_filter" in self.config:
            params = self.config["median_filter"]
            img = median_filter(img, ksize=params.get("ksize", 3))
        
        if "gaussian_blur" in self.config:
            params = self.config["gaussian_blur"]
            img = Gaussian_blur(img, ksize=params.get("ksize", 3), sigma=params.get("sigma", 1.0))

        if self.config.get("grayscale", False):
            img = grayscale(img)
            img = np.stack([img]*3, axis=-1)

        if self.config.get("normalize", False):
            img = normalize_img(img)

        return img.astype(np.float32)

#Create a dataset from a directory path with images already placed in folders. Eg- directory has 2 folders cats and dogs with images of each.
def create_dataset_from_directory(data_path: str, batch_size: int, preprocessor: Preprocessor):
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_map = {name: i for i, name in enumerate(class_names)}

    filepaths = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(data_path, class_name)
        for filename in os.listdir(class_dir):
            filepaths.append(os.path.join(class_dir, filename))
            labels.append(class_map[class_name])

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.shuffle(buffer_size=len(filepaths))

    def apply_custom_preprocessing(path, label):
        img_h = preprocessor.config["resize"]["height"]
        img_w = preprocessor.config["resize"]["width"]
        
        image = tf.py_function(func=preprocessor.process, inp=[path], Tout=tf.float32)
        
        image.set_shape((img_h, img_w, 3))
        return image, label

    dataset = dataset.map(apply_custom_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"Dataset created using preprocessor.")
    return dataset

