import cv2
import numpy as np

# ----------------- Preprocessing Functions -----------------

def resize(img, new_height, new_width):
    if img.ndim == 3:
        resized_image = np.zeros((new_height,new_width,img.shape[2]), dtype=img.dtype)
    else:
        resized_image = np.zeros((new_height,new_width), dtype=img.dtype)
    
    original_height, original_width = img.shape[:2]
    height_ratio = original_height / new_height
    width_ratio = original_width / new_width

    for i in range(new_height):
        for j in range(new_width):
            x = int(i * height_ratio)
            y = int(j * width_ratio)
            resized_image[i,j] = img[x,y]
    return resized_image

def grayscale(img):
    height, width = img.shape[:2]
    gray = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            R, G, B = img[i,j][::-1]  # OpenCV loads BGR
            gray[i,j] = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

def normalize_img(img, new_min=0, new_max=255):
    old_min, old_max = img.min(), img.max()
    img_norm = (img - old_min) * (new_max - new_min) / (old_max - old_min + 1e-8) + new_min
    return img_norm.astype(np.float32)

def mean_filter(img, ksize=3):
    pad = ksize // 2
    if img.ndim == 3:
        padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    else:
        padded = np.pad(img, ((pad,pad),(pad,pad)), mode='reflect')

    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+ksize, j:j+ksize]
            out[i,j] = region.mean(axis=(0,1))
    return out

def median_filter(img, ksize=3):
    pad = ksize // 2
    if img.ndim == 3:
        padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    else:
        padded = np.pad(img, ((pad,pad),(pad,pad)), mode='reflect')

    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+ksize, j:j+ksize]
            out[i,j] = np.median(region, axis=(0,1))
    return out

def Gaussian_kernel(ksize, sigma):
    ax = np.linspace(-(ksize//2), ksize//2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    return kernel / np.sum(kernel)

def Gaussian_blur(img, sigma, ksize=3):
    kernel = Gaussian_kernel(ksize, sigma)
    pad = ksize // 2
    if img.ndim == 3:
        padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    else:
        padded = np.pad(img, ((pad,pad),(pad,pad)), mode='reflect')

    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+ksize, j:j+ksize]
            if img.ndim == 3:
                out[i,j] = np.sum(region * kernel[..., None], axis=(0,1))
            else:
                out[i,j] = np.sum(region * kernel, axis=(0,1))
    return out

# ----------------- Testing -----------------

if __name__ == "__main__":
    img_path = "T1.jpg"  # Change to your test image
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found!")
        exit()

    gray = grayscale(img)
    resized = resize(img, 128, 128)
    meaned = mean_filter(img, ksize=3)
    medianed = median_filter(img, ksize=3)
    blurred = Gaussian_blur(img, sigma=1.0, ksize=5)

    # Display results
    cv2.imshow("Original", img)
    cv2.imshow("Grayscale", gray.astype(np.uint8))
    cv2.imshow("Resized", resized.astype(np.uint8))
    cv2.imshow("Mean Filter", meaned.astype(np.uint8))
    cv2.imshow("Median Filter", medianed.astype(np.uint8))
    cv2.imshow("Gaussian Blur", blurred.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
