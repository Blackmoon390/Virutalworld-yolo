import cv2
import numpy as np
from ultralytics import YOLO
import torch


cpu= "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {cpu}")

model=YOLO("yolo11s-seg.pt")

bluecolour=np.full((640,480,3),(24, 34, 200),dtype='uint8')

def apply_blue_black_noise(frame, mask, block=2):
    h, w = frame.shape[:2]   # frame = (640, 480, 3)

    # --- Fix mask shape mismatch ----
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h))

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


cam=cv2.VideoCapture(0)

while cam.isOpened():
    _,frame=cam.read()

    results=model(frame,verbose=False,stream=True)
    frame2=np.zeros(frame.shape[:2],dtype='uint8')
    # backgroud_texture=cv2.resize(backgroud_texture,(frame.shape[1],frame.shape[0]))

    for r in results:
        if r.masks is not None:
            
            for cls,data in zip(r.names,r.masks.data):
                if int(cls) == 0:
                   mask = data.cpu().numpy().astype('uint8')*255
                   mask=cv2.resize(mask,(640,480))
    mask_not=cv2.bitwise_not(mask)
    # backgroud_texture_mask=cv2.bitwise_and(backgroud_texture,backgroud_texture,mask=mask)
    print(mask.shape,bluecolour.shape)
    person_mask_colour=apply_blue_black_noise(bluecolour, mask)

    cv2.imshow('mask',person_mask_colour)       



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
