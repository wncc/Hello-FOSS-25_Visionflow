import numpy as np
import cv2
from visionflow.preprocess import median_filter

# Small test image
test_img = np.array([
    [[100, 100, 100], [150, 150, 150], [200, 200, 200]],
    [[110, 110, 110], [160, 160, 160], [210, 210, 210]],
    [[120, 120, 120], [170, 170, 170], [220, 220, 220]]
], dtype=np.uint8)

# Your implementation
our_result = median_filter(test_img, ksize=3)

# OpenCV's implementation
opencv_result = cv2.medianBlur(test_img, 3)

print("Our result:")
print(our_result[:,:,0])  # Just R channel
print("\nOpenCV result:")
print(opencv_result[:,:,0])
print("\nDifference (should be 0 if identical):")
print(np.abs(our_result.astype(int) - opencv_result.astype(int))[:,:,0])