import cv2
import numpy as np



def getvalues():
    with open("settings.txt","r") as lines:
        for line in lines:
            if line.startswith("Animationframe="):
                coordinates=line.strip and line.split("=")
                frame_ratio=coordinates[1]
                return frame_ratio
                

getvalues()


base = cv2.imread('main2.png')       



def blender(overlay):
    base2=cv2.resize(base,None,fx=0.70,fy=0.70,interpolation=cv2.INTER_AREA)
    overlay = cv2.resize(overlay, (640, 500))
    h,w=overlay.shape[:2]
    y=300
    x=500

    base = base2[y:y+h, x:x+w]   # FIXED

    

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([70, 10, 10], dtype=np.uint8)

    mask = cv2.inRange(base, lower_black, upper_black)



    mask = cv2.GaussianBlur(mask, (5,5), 0)


    mask_3ch = cv2.merge([mask, mask, mask])
    mask_3ch = mask_3ch / 255.0  # normalize to 0-1 for blending

# Blend images only at black regions
    blended = base * (1 - mask_3ch) + overlay * mask_3ch
    blended = blended.astype(np.uint8)

