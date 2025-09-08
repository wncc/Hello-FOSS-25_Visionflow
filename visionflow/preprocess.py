import cv2
import numpy as np 

def preprocess(img_path, size=(64,64), normalize = True, augment = False, noise_reduction):
    def load_image(path):
        img = cv2.imread(path)
        return img

    #Suppose you want to resize 735 to 600, then every pixel in the new image searches for a pixel at a height 735/600 multiplied by it's new height.
    def resize(img, new_height, new_width):
        resized_image = np.zeros((new_height,new_width, img.shape[2]), dtype=img.dtype):   #dtype is important... shud be same as image np.float32 or np.float64 doesn't work
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
        grayscale = np.zeros((height, width), dtype=img.dtype)
        for i in range(height):
            for j in range(width):
                #Extract R, B, G values from each pixel. Pay attention to order.
                R, G, B = img[i,j][::-1]
                gray_value = 0.2989 * R + 0.5870 * G + 0.1140 * B 
                grayscale[i,j] = gray_value
        return grayscale

    def normalize(img, new_min = 0, mew_max = 1):
        old_min, old_max = img.min(), img.max()
        #normalizing to new min, new max. Default from 0 to 1
        #1e-8 to avoid dividing by zero
        #there are more normalization methods we can check later
        img_norm = (img - old_min) * (new_max - new_min) / (old_max - old_min + 1e-8) + new_min 
        return img_norm

    img = load_image(img_path)
    img = resize(img)
    img = grayscale(img)
    img = normalize(img)
    

    



        
