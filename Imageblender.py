import cv2
import numpy as np
import Variables as vr


def verify_key(key):
        return True if key == vr.clamp[237:291] else False

def print_error():
    print(f"\033[91mcredentials missing please replace a settings.txt file as default\033[0m")
    exit()

            

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
key=verify_key(vr.key) 
bluecolour=np.full((640,480,3),(24, 34, 200),dtype='uint8')
bluecolour=cv2.cvtColor(bluecolour,cv2.COLOR_BGR2RGB)  

base2=cv2.resize(base,None,fx=vr.frame_ratio,fy=vr.frame_ratio,interpolation=cv2.INTER_AREA)
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

 
    base = basemain[y:y+h, x:x+w]

  
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([70, 10, 10], dtype=np.uint8)

    mask = cv2.inRange(base, lower_black, upper_black)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    mask_3ch = (mask / 255.0)[..., None]

    blended = base * (1 - mask_3ch) + overlay * mask_3ch
    blended = blended.astype(np.uint8)

    
    basemain[y:y+h, x:x+w] = blended

    return basemain
