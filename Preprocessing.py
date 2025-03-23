import cv2
import numpy as np

# Apply Gaussian blur to remove noise
blurred_image = cv2.GaussianBlur(image_data, (5, 5), 0)

plt.imshow(blurred_image, cmap="gray", origin="lower")
plt.title("Denoised NASA Image")
plt.show()
# Normalize pixel intensity values
norm_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

plt.imshow(norm_image, cmap="gray", origin="lower")
plt.title("Contrast Enhanced Image")
plt.show()

