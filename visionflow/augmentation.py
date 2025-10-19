import numpy as np



def flip_horizontal(img):
    #Flips an image horizontally.
    return img[:, ::-1]

def flip_vertical(img):
    #Flips an image vertically.
    return img[::-1, :]

def rotate_image(img, angle):
    #Rotates an image by a specified angle (in degrees counter-clockwise).
    theta = np.deg2rad(angle) 
    h, w = img.shape[:2]
    r = np.sqrt((w/2)**2 + (h/2)**2)
    rotated = np.zeros((int(h + 2*r*(1-np.cos(theta))), int(w + 2*r*(1-np.cos(theta))), img.shape[2]), dtype=img.dtype)
    center_x, center_y = h // 2, w // 2
    rot_center_x, rot_center_y = rotated.shape[0] // 2, rotated.shape[1] // 2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Change in axis
            x = i - center_x
            y = j - center_y

            # Rotated coordinates
            x_new = int(x * np.cos(theta) - y * np.sin(theta)) + rot_center_x 
            y_new = int(x * np.sin(theta) + y * np.cos(theta)) + rot_center_y

            # Project to new image
            if 0 <= x_new < rotated.shape[0] and 0 <= y_new < rotated.shape[1]:
                rotated[x_new, y_new] = img[i, j]

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
