import cv2
import numpy as np

def apply_blue_black_noise(frame, mask, block=2):
    h, w = mask.shape[:2]   # frame = (640, 480, 3)

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
    # Glow color
    r, g, b = 125, 143, 255  # orange, like your example

 
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # --- expand area for outer glow ---
    kernel = np.ones((35, 35), np.uint8)   # increase size = bigger glow base 15
    expanded = cv2.dilate(mask, kernel, iterations=1)

    # --- outer-only area (this prevents inner glow!) ---
    outer = cv2.subtract(expanded, mask)

    # --- blur to create smooth glow ---
    glow = cv2.GaussianBlur(outer, (0, 0), sigmaX=20, sigmaY=20)

    # --- normalize for brightness ---
    glow = cv2.normalize(glow, None, 0, 255, cv2.NORM_MINMAX)

    # --- color the glow ---
    glow_f = glow.astype(float) / 255.0
    glow_bgr = np.zeros((*glow.shape, 3), dtype=np.uint8)
    glow_bgr[:, :, 0] = glow_f * b
    glow_bgr[:, :, 1] = glow_f * g
    glow_bgr[:, :, 2] = glow_f * r

    return glow_bgr

def resize_yolo_object_only2(base_img, bbox, y1=300, y2=600):
    x1, y_top, x2, y_bottom = bbox

    obj = base_img[y_top:y_bottom, x1:x2].copy()
    oh, ow = obj.shape[:2]

    target_h = y2 - y1  # = 300 px

    scale = target_h / oh
    new_w = int(ow * scale)

    resized = cv2.resize(obj, (new_w, target_h))
    return resized


def resize_yolo_object_only3(base_img, bbox, y1=300):
    x1, y_top, x2, y_bottom = bbox

    # Crop original object
    obj = base_img[y_top:y_bottom, x1:x2].copy()
    oh, ow = obj.shape[:2]

    # Fixed height
    target_h = 500

    # Resize based on height only
    scale = target_h / oh
    new_w = int(ow * scale)

    resized = cv2.resize(obj, (new_w, target_h))

    # Updated bbox (height always 500)
    new_x1 = x1
    new_x2 = x1 + new_w
    new_y1 = y1
    new_y2 = y1 + target_h   # always y1 + 500

    return resized, (new_x1, new_y1, new_x2, new_y2)

def resize_mask_only_crop(mask, bbox, y1, y2):
    x1, y_top, x2, y_bottom = bbox

    # Crop object
    obj_mask = mask[y_top:y_bottom, x1:x2].copy()
    oh, ow = obj_mask.shape[:2]

    target_h = y2 - y1
    scale = target_h / oh
    new_w = int(ow * scale)

    resized_mask = cv2.resize(obj_mask, (new_w, target_h))
    return resized_mask  # only the resized object



def resize_mask_height_only(mask, bbox, y1, y2):
   
    H, W = mask.shape[:2]
    x1, y_top, x2, y_bottom = bbox

    # Crop object mask
    obj_mask = mask[y_top:y_bottom, x1:x2].copy()
    oh, ow = obj_mask.shape[:2]

    # Target height
    target_h = y2 - y1
    if target_h <= 0:
        return np.zeros((0, W), dtype=mask.dtype)

    # Resize height only, keep width
    resized_mask = cv2.resize(obj_mask, (ow, target_h))

    # Create new mask with target height, full frame width
    final_mask = np.zeros((target_h, W), dtype=mask.dtype)

    # Paste resized object at same x position
    final_mask[:, x1:x1+ow] = resized_mask

    return final_mask