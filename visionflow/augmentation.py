import numpy as np



def flip_horizontal(img):
    #Flips an image horizontally.
    return img[:, ::-1]

def flip_vertical(img):
    #Flips an image vertically.
    return img[::-1, :]

def rotate_image(img, angle):
    #Rotates an image by a given angle.
    theta = np.deg2rad(angle)
    h, w = img.shape[:2]
    rotated = np.zeros_like(img)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    for i in range(h):
        for j in range(w):
            x_new = int(j * cos_theta - i * sin_theta)
            y_new = int(j * sin_theta + i * cos_theta)
            if 0 <= x_new < w and 0 <= y_new < h:
                rotated[y_new, x_new] = img[i, j]
                
    return rotated

def adjust_brightness(img, value=50):
    #Adjusts image brightness by adding a value to pixel intensities.
    return np.clip(img.astype(np.int16) + value, 0, 255).astype(np.uint8)

def adjust_contrast(img, factor=1.2):
    #Adjusts image contrast by multiplying pixel intensities by a factor.
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

def add_gaussian_noise(img, mean=0, sigma=25):
    #Adds Gaussian noise to an image.
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(img, prob=0.01):
    #Adds salt and pepper noise to an image.
    output = img.copy()
    rnd = np.random.rand(*img.shape[:2])
    output[rnd < prob / 2] = 0   # pepper
    output[rnd > 1 - prob / 2] = 255 # salt
    return output
