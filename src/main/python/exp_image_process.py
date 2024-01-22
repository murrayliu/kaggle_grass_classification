import cv2
import numpy as np
from matplotlib import pyplot as plt

# === read image ===
image_path = "C:/Users/murray/Desktop/kaggle_grass_classification/src/main/python/data/train/Black-grass/37d85d833.png"
bgr_image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2HSV)

# === extract green area ===
lower = (25,40,50)
upper = (75,255,255)
mask_image = cv2.inRange(hsv_image,lower,upper)
struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
blur_mask_image = cv2.morphologyEx(mask_image,cv2.MORPH_CLOSE,struc)

# === mask raw image ===
boolean = blur_mask_image == 0
rgb_image[boolean] = (0,0,0)


# === show extracted part ===
show_image = np.vstack([rgb_image])
plt.imshow(show_image)
plt.show()