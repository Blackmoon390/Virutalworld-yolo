import cv2
import numpy as np

# --- your RGB glow color variable ---
r, g, b =125, 143, 255



mask = cv2.imread(r"C:\python\computer_vision\virtual_world\upated.png", cv2.IMREAD_GRAYSCALE)

# 1. Bigger dilation for wider glow
kernel = np.ones((35,35), np.uint8)   # bigger = stronger glow
dilated = cv2.dilate(mask, kernel, iterations=1)

# 2. Outer region
outline = cv2.subtract(dilated, mask)

# 3. Blur
glow = cv2.GaussianBlur(outline, (0,0), sigmaX=40, sigmaY=40)

# 4. ***Boost brightness*** (super glow)
glow = cv2.normalize(glow, None, 0, 255, cv2.NORM_MINMAX)
glow = cv2.addWeighted(glow, 2.5, glow, 0, 0)  # <- increase intensity (2.5 = super bright)

# 5. Convert glow to BGR
glow_bgr = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)

# 6. Apply RGB color
colored_glow = np.zeros_like(glow_bgr)
colored_glow[:,:,0] = glow * (b / 255)
colored_glow[:,:,1] = glow * (g / 255)
colored_glow[:,:,2] = glow * (r / 255)


cv2.imshow("frame",colored_glow)
cv2.waitkey(0)
