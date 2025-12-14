import cv2
import numpy as np


bluecolour=np.full((640,480,3),(24, 34, 200),dtype='uint8')
bluecolour=cv2.cvtColor(bluecolour,cv2.COLOR_BGR2RGB)

def apply_blue_black_noise(frame=bluecolour, mask=None, block=1):
    h, w = mask.shape[:2]  

    # --- Fix mask shape mismatch ----
    if frame.shape != (h, w):
        frame = cv2.resize(frame, (w, h))

    # Make binary mask
    mask = (mask > 0).astype(np.uint8) * 255

    # Convert mask to 3-channel
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Noise generation
    small_h = h // block
    small_w = w // block

    rnd = np.random.randint(0, 2, (small_h, small_w), dtype=np.uint8)

    noise_small = np.zeros((small_h, small_w, 3), dtype=np.uint8)
    noise_small[rnd == 1] = frame[0, 0]

    big_noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Apply mask (only noise inside)
    inside = cv2.bitwise_and(big_noise, mask3)

    return inside



def glow(mask):
    
    r, g, b = 45, 58, 175 

 
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # --- expand area for outer glow ---
    kernel = np.ones((10, 10), np.uint8)   # increase size = bigger glow base 15
    expanded = cv2.dilate(mask, kernel, iterations=1)

    # --- outer-only area (this prevents inner glow!) ---
    outer = cv2.subtract(expanded, mask)

    # --- blur to create smooth glow ---
    glow = cv2.GaussianBlur(outer, (0, 0), sigmaX= 10, sigmaY=10)

    # --- normalize for brightness ---
    glow = cv2.normalize(glow, None, 0, 255, cv2.NORM_MINMAX)

    # --- color the glow ---
    glow_f = glow.astype(float) / 255.0
    glow_bgr = np.zeros((*glow.shape, 3), dtype=np.uint8)
    glow_bgr[:, :, 0] = glow_f * b
    glow_bgr[:, :, 1] = glow_f * g
    glow_bgr[:, :, 2] = glow_f * r

    return glow_bgr







def resize_mask_height_only(mask, bbox, y1, y2, gap=30):
    H, W = mask.shape[:2]
    x1, y_top, x2, y_bottom = bbox

    # Extract original object mask
    obj_mask = mask[y_top:y_bottom, x1:x2].copy()
    oh, ow = obj_mask.shape[:2]

    # Final mask height = y1 → y2
    full_h = y2 - y1

    # Available height after applying gap
    available_h = full_h - gap
    if available_h <= 0:
        available_h = 1

    # Scale height to fill available space exactly
    scale_h = available_h / oh
    new_h = available_h
    new_w = int(round(ow * scale_h))  # round to nearest pixel

    # Resize mask
    resized_obj = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Create final mask
    final_mask = np.zeros((full_h, W), dtype=mask.dtype)

    # Determine horizontal placement
    start_x = x1 + max(0, (x2 - x1 - new_w)//2)

    # Clip to prevent overflow
    end_x = min(start_x + new_w, W)
    resized_obj = resized_obj[:, :end_x-start_x]  # match slice width

    # Place mask
    final_mask[gap:gap+new_h, start_x:end_x] = resized_obj

    return final_mask
