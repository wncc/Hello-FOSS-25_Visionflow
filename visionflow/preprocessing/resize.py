#Suppose you want to resize 735 to 600, then every pixel in the new image searches for a pixel at a height 735/600 multiplied by it's new height.
def custom_resize(img, new_height, new_width):
    resized_image = np.zeros((new_height,new_width, img.shape[2]), dtype=img.dtype)    #dtype is important... shud be same as image np.float32 or np.float64 doesn't work
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
