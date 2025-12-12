import cv2
import numpy as np

def apply_blue_black_noise(frame, mask, block=1):
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
    # Glow color
    r, g, b = 45, 58, 175  # orange, like your example

 
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





def resize_mask_height_only3main(mask, bbox, y1, y2, gap=30):
    H, W = mask.shape[:2]
    x1, y_top, x2, y_bottom = bbox

    obj_mask = mask[y_top:y_bottom, x1:x2].copy()
    oh, ow = obj_mask.shape[:2]

    # Normal target height
    target_h = y2 - y1

    # Resize object to new height
    resized_mask = cv2.resize(obj_mask, (ow, target_h), interpolation=cv2.INTER_NEAREST)

    # Now add EMPTY SPACE (gap) at the TOP
    final_h = target_h + gap
    final_mask = np.zeros((final_h, W), dtype=mask.dtype)

    # Paste mask starting at row "gap"
    final_mask[gap:gap+target_h, x1:x1+ow] = resized_mask

    return final_mask

def resize_mask_height_only2(mask, bbox, y1, y2, gap=30):
    H, W = mask.shape[:2]
    x1, y_top, x2, y_bottom = bbox

    # Crop YOLO mask region
    obj = mask[y_top:y_bottom, x1:x2].copy()
    oh, ow = obj.shape[:2]

    # AREA where final mask must fit
    target_h = y2 - y1
    available_h = target_h - gap
    if available_h <= 1:
        available_h = 1

    # Scale height to available_h, keep aspect
    scale = available_h / oh
    new_h = int(oh * scale)
    new_w = int(ow * scale)

    # ---- CLAMP WIDTH to frame ----
    if x1 + new_w > W:
        new_w = W - x1    # shrink width to fit
        if new_w < 1:
            new_w = 1

    # Resize object mask
    resized = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Create EMPTY output only for y1→y2 region
    final_mask = np.zeros_like(mask)

    # ---- FIT INSIDE THE BOX (no overflows) ----
    y_start = y1 + gap
    y_end   = y_start + new_h

    if y_end > y2:
        y_end = y2
        new_h = y_end - y_start
        resized = resized[:new_h]  # crop if needed

    # Place resized mask
    final_mask[y_start:y_start+new_h, x1:x1+new_w] = resized

    return final_mask


def resize_mask_height_onlydumb(mask, bbox, y1, y2, gap=30):
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

    # ----- KEY FIX: WIDTH FIRST -----
    # Preserve original bbox width
    target_w = x2 - x1

    # Scale height proportionally
    scale = target_w / ow
    new_h = int(oh * scale)

    # If new height is bigger than available, scale down
    if new_h > available_h:
        scale = available_h / oh
        new_h = available_h
        target_w = int(ow * scale)

    # Resize object (no horizontal compression)
    resized_obj = cv2.resize(obj_mask, (target_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Create final mask
    final_mask = np.zeros((full_h, W), dtype=mask.dtype)

    # Place object with gap on top
    final_mask[gap:gap+new_h, x1:x1+target_w] = resized_obj

    return final_mask




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
