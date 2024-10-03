import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(img, None)

# Convert the descriptors to a PyTorch tensor
descriptors_tensor = torch.tensor(descriptors, dtype=torch.float32)

# Visualize the keypoints on the image (optional)
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show the image with keypoints
plt.imshow(img_with_keypoints, cmap='gray')
plt.show()

# Now, `descriptors_tensor` can be used in your PyTorch model.
print("SIFT Descriptors Tensor Shape: ", descriptors_tensor.shape)
