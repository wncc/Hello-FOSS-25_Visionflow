import numpy as np

# ==============================================================================
# All of your custom, from-scratch image augmentation functions
# ==============================================================================

def flip_horizontal(img):
    """Flips an image horizontally."""
    return img[:, ::-1]

def flip_vertical(img):
    """Flips an image vertically."""
    return img[::-1, :]

def rotate_image(img, angle):
    """Rotates an image by a given angle in degrees."""
    theta = np.deg2rad(angle)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    rotated = np.zeros_like(img)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    for i in range(h):
        for j in range(w):
            x = j - cx
            y = i - cy
            x_new = int(cos_theta * x + sin_theta * y + cx)
            y_new = int(-sin_theta * x + cos_theta * y + cy)
            if 0 <= x_new < w and 0 <= y_new < h:
                rotated[i, j] = img[y_new, x_new]
    return rotated

def adjust_brightness(img, value=50):
    """Adjusts image brightness by adding a value to pixel intensities."""
    # Cast to a larger integer type to prevent overflow before clipping
    return np.clip(img.astype(np.int16) + value, 0, 255).astype(np.uint8)

def adjust_contrast(img, factor=1.2):
    """Adjusts image contrast by multiplying pixel intensities by a factor."""
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

def add_gaussian_noise(img, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(img, prob=0.01):
    """Adds salt and pepper noise to an image."""
    output = img.copy()
    rnd = np.random.rand(*img.shape[:2])
    output[rnd < prob / 2] = 0   # pepper
    output[rnd > 1 - prob / 2] = 255 # salt
    return output
