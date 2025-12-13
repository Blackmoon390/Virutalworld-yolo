import cv2
import numpy as np



def getcoordinatevalues():
    with open("settings.txt","r") as lines:
        for line in lines:
            if line.startswith("Animationframesize="):
                coordinates=line.split("=")
                frame_ratio=coordinates[1].strip()
                frame_ratio=int(frame_ratio)/100
            if line.startswith("key="):
                key=line.split("=")[1]      
    return frame_ratio,key
            

def resizer(framesize,fitvalue,newframe):
    new_pointer=int(fitvalue*(newframe/framesize))
    return new_pointer
    
            
def size_ratio_fitter(newframesize):
    frameX,frameY=672, 1084  # default animation frame size 50% for fit person
    y1fit,y2fit=270,515
    xstart=372
    newframeX,newframeY=newframesize
    y1=resizer(frameY,y1fit,newframeY)
    y2=resizer(frameY,y2fit,newframeY)
    x1=resizer(frameX,xstart,newframeX)
    return y1,y2,x1

                

base = cv2.imread('Backgroud-image.png')  
coordinate=getcoordinatevalues()   
base2=cv2.resize(base,None,fx=coordinate,fy=coordinate,interpolation=cv2.INTER_AREA)
y1,y2,xstartpointer=size_ratio_fitter(base2.shape[:2])




def blender(overlay):
    h, w = overlay.shape[:2]

    y = y1
    x = xstartpointer

    basemain = base2.copy()
    H, W = base2.shape[:2]

    
    if x + w > W:
        w = W - x
        overlay = overlay[:, :w]

    if y + h > H:
        h = H - y
        overlay = overlay[:h, :]

    # crop base
    base = basemain[y:y+h, x:x+w]

    # black detection
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([70, 10, 10], dtype=np.uint8)

    mask = cv2.inRange(base, lower_black, upper_black)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    mask_3ch = (mask / 255.0)[..., None]

    blended = base * (1 - mask_3ch) + overlay * mask_3ch
    blended = blended.astype(np.uint8)

    # write back safely
    basemain[y:y+h, x:x+w] = blended

    return basemain
