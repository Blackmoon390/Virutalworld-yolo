import cv2
import numpy as np


base = cv2.imread('test2.png')       
overlay = cv2.imread('white_mask.jpg') 
overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))


lower_black = np.array([0, 0, 0])
upper_black = np.array([60, 10, 10])  # adjust tolerance if needed
mask = cv2.inRange(base, lower_black, upper_black)


mask = cv2.GaussianBlur(mask, (5,5), 0)


mask_3ch = cv2.merge([mask, mask, mask])
mask_3ch = mask_3ch / 255.0  # normalize to 0-1 for blending

# Blend images only at black regions
blended = base * (1 - mask_3ch) + overlay * mask_3ch
blended = blended.astype(np.uint8)

cv2.imwrite('Blended.jpg', blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
