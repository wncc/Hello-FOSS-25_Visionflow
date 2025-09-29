import cv2
import numpy as np 

def preprocess_image(img_path, size=(64,64), normalize = True, augment = False, noise_reduction = None):
    pass
    
def load_image(path):
    img = cv2.imread(path)
    return img

#Suppose you want to resize 735 to 600, then every pixel in the new image searches for a pixel at a height 735/600 multiplied by it's new height.
def resize(img, new_height, new_width):
    resized_image = np.zeros((new_height,new_width, img.shape[2]), dtype=img.dtype)   #dtype is important... shud be same as image np.float32 or np.float64 doesn't work
    original_height, original_width = img.shape[:2]
    height_ratio = original_height/new_height
    width_ratio = original_width/new_width
    #Suppose i,j pixel searches for x,y pixel in the orginal image
    for i in range(new_height):
        for j in range(new_width):
            x = int(i * height_ratio)
            y = int(j * width_ratio)
            resized_image[i,j] = img[x,y]
    return resized_image
    #very slow - need to make it faster

def grayscale(img):
    height, width = img.shape[:2]
    grayscale = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            #Extract R, B, G values from each pixel. Pay attention to order.
            R, G, B = img[i,j][::-1]
            gray_value = 0.2989 * R + 0.5870 * G + 0.1140 * B 
            grayscale[i,j] = gray_value
    return grayscale

def normalize_img(img, new_min = 0, new_max = 255):
    old_min, old_max = img.min(), img.max()
    #normalizing to new min, new max. Default from 0 to 1
    #1e-8 to avoid dividing by zero
    #there are more normalization methods we can check later
    img_norm = (img - old_min) * (new_max - new_min) / (old_max - old_min + 1e-8) + new_min 
    return img_norm.astype(np.float32)
    
    #Noise reduction functions



    #Average filtering - taking mean of neighbours 
def mean_filter(img, ksize=3):
    pad = ksize // 2 #size of padding
    padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect') #padding image to take care of border cases- only for grayscale, for rgb theres an extra dimension
    out = np.zeros_like(img, dtype=np.float32) #output 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+ksize, j:j+ksize]
            out[i, j] = region.mean(axis=(0,1))
    return out

#Median filtering - taking median of neighbours \
def median_filter(img, ksize=3):
    pad = ksize // 2
    padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+ksize, j:j+ksize]
            out[i, j] = np.median(region, axis=(0,1))
    return out


#making a gaussian kernel for each pixel
def Gaussian_kernel(ksize, sigma):
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

#2nd problem- implemeting this nicely 
#again- making this faster
def Gaussian_blur(img, sigma, ksize = 3):
    kernel = Gaussian_kernel(ksize, sigma)
    pad = ksize // 2
    padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+ksize, j:j+ksize]
            out[i, j] = np.sum(region * kernel, axis=(0,1))
    return out


# Thresholding
def custom_thresholding_binary(img, threshold):
    binary = np.where(img>=threshold, 255, 0).astype(img.dtype)
    return binary

def custom_thresholding_binary_inv(img, threshold):
    binary_inv = np.where(img>=threshold, 0, 255).astype(img.dtype)
    return binary_inv

def custom_thresholding_tozero(img, threshold):
    tozero = np.where(img>=threshold, img, 0).astype(img.dtype)
    return tozero

def custom_thresholding_tozero_inv(img, threshold):
    tozero_inv = np.where(img>=threshold, 0, img).astype(img.dtype)
    return tozero_inv

def custom_thresholding_trunc(img, threshold):
    trunc = np.where(img>=threshold, threshold, img).astype(img.dtype)
    return trunc

# Adaptive Mean
def adaptive_mean(img, ksize, c):
    pad = ksize//2
    padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    out = np.zeros(img.shape, dtype=img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+ksize, j:j+ksize]
            threshold = np.mean(region, axis=(0,1)) - c
            out[i,j] = (img[i,j]>=threshold)*255
    return out

# Adaptive Gaussian        
def adaptive_gaussian(img, ksize, c):
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  #Sigma is calculated inetrnally, so for large blockSize, sigma will be quite large, making it a smooth Gaussian. The effect is: the threshold is more influenced by nearby pixels than far-away ones.
    kernel = Gaussian_kernel(ksize, sigma)
    pad = ksize//2
    padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    out = np.zeros(img.shape, dtype=img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+ksize, j:j+ksize]
            threshold = np.sum(region * kernel, axis=(0,1)) - c
            out[i,j] = (img[i,j]>=threshold)*255
    return out


#Augmentation
def flip_horizontal(img):
    return img[:, ::-1]

def flip_vertical(img):
    return img[::-1, :]

def rotate_90(img):  #Rotating by any other angle is trickier to implement
    return np.rot90(img)  # counter-clockwise

def adjust_brightness(img, value=50):
    return np.clip(img + value, 0, 255).astype(np.uint8)  #Increase value by 50 and ensure that pixel values lie between 0 to 255

def adjust_contrast(img, factor=1.2):
    mean = img.mean()   #Find the mean of all pixel values. This is now the reference
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8) #First center it around 0, and then multiply by a factor. If factor is greater than 1, then pixels spread farther away from mean and if it's lesser then they come closer to mean. Then shift it back so that the brightness is not affected.

def add_gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape) #Creates the noise. A mean of 0 means that the entire image is not made brighter or darker. Higher the SD wider is the curve and thus more intense will be the noise.
    noisy = img.astype(np.float32) + noise #Adds the noise to the image. Here cannot convert noise directly as unit and add since noise has negative values as well. So do both using int.
    return np.clip(noisy, 0, 255).astype(np.uint8) #Then clip negatives to 0 and convert to uint.

def add_salt_pepper(img, prob=0.01):  #Greater prob means more of the pixels are turned into noise
    output = img.copy()
    rnd = np.random.rand(*img.shape[:2])  #Unpacks the shape tuple and returns a random array of same shape as image with integers between 0.0 and 1.0 representing probabilities
    output[rnd < prob/2] = 0       # pepper
    output[rnd > 1 - prob/2] = 255 # salt
    return output

def random_crop(img, crop_h, crop_w):
    h, w = img.shape[:2]
    y = np.random.randint(0, h - crop_h + 1) #Selects a random row from 0(top row) to h-crop_h+1. This is done so that the range is withing limits and +1 is because randint include both upper and lower bounds.
    x = np.random.randint(0, w - crop_w + 1)
    return img[y:y+crop_h, x:x+crop_w]

def rotate_image(img, angle):  #Suppose angle is in degrees
    # Convert angle to radians
    theta = np.deg2rad(angle)

    # Get image shape
    h, w = img.shape[:2]
    # Center of the image
    cx, cy = w // 2, h // 2    #The image rotates about center. So we will define the center as orgin

    # Create output image
    rotated = np.zeros_like(img)

    # Precompute sin and cos
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # For each pixel in the output
    for i in range(h):
        for j in range(w):
            # Coordinates relative to center. Initially, the origin is the top left corner. We now define our new origin as center so we get new cordinates for each pixel.
            x = j - cx  #The column or width
            y = i - cy   #The row or height
            # Rotate. For this, take any pixel i,j in the new image. We should find what image from the source is getting used. So we find the x and y cordinates of this pixel in the original image.
            #Suppose a point is rotated by theta due to rotation of axes. Then the new cordinates of the point is given by xcos(theta)-ysin(theta), xsin(theta)+ycos(theta)
            #And the original points are given by, xcos(theta)+ysin(theta), -xsin(theta)+ycos(theta)
            #So we find the location of these original points. And then shift back to the original cordinate system.
            x_new = int(cos_theta * x + sin_theta * y + cx)
            y_new = int(-sin_theta * x + cos_theta * y + cy)
            # Assign pixel if inside bounds
            if 0 <= x_new < w and 0 <= y_new < h:
                rotated[i, j] = img[y_new, x_new]
    return rotated

def random_rotate(img, max_angle=30):
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate_image(img, angle)

# Pipeline that comines all augmenation functions

def augment_image(img, methods=["flip_h", "flip_v", "random_rotate", "brightness", "contrast", "gaussian", "salt_pepper"]):
    augmented = []

    for method in methods:
        if method == "flip_h":
            augmented.append(flip_horizontal(img))
        elif method == "flip_v":
            augmented.append(flip_vertical(img))
        elif method == "random_rotate":
            augmented.append(random_rotate(img, max_angle=30))  # rotate a random angle between -30 and 30
        elif method == "brightness":
            augmented.append(adjust_brightness(img, value=50))
        elif method == "contrast":
            augmented.append(adjust_contrast(img, factor=1.2))
        elif method == "gaussian":
            augmented.append(add_gaussian_noise(img, sigma=25))
        elif method == "salt_pepper":
            augmented.append(add_salt_pepper(img, prob=0.01))

    return augmented

    



        
