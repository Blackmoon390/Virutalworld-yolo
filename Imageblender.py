import cv2
import numpy as np



def getcoordinatevalues():
    with open("settings.txt","r") as lines:
        for line in lines:
            if line.startswith("Animationframesize="):
                coordinates=line.split("=")
                frame_ratio=coordinates[1].strip()
                frame_ratio=int(frame_ratio)/100            
    return frame_ratio
            

def resizer(framesize,fitvalue,newframe):
    new_pointer=int(fitvalue*(newframe/framesize))
    return new_pointer
    
            
def size_ratio_fitter(newframesize):
    frameX,frameY=672, 1084  # default animation frame size 50% for fit person
    y1fit,y2fit=300,480
    xstart=400
    newframeX,newframeY=newframesize
    y1=resizer(frameY,y1fit,newframeY)
    y2=resizer(frameY,y2fit,newframeY)
    x1=resizer(frameX,xstart,newframeX)
    return y1,y2,x1

                

base = cv2.imread('main2.png')  
coordinate=getcoordinatevalues()   
base2=cv2.resize(base,None,fx=coordinate,fy=coordinate,interpolation=cv2.INTER_AREA)
xstartpointer,y1,y2=size_ratio_fitter(base2.shape[:2])



def blender(overlay):
    
    # overlay = cv2.resize(overlay, (640, 500))
    h,w=overlay.shape[:2]
    y=300
    x=500

    base = base2[y:y+h, x:x+w]   

    

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([70, 10, 10], dtype=np.uint8)

    mask = cv2.inRange(base, lower_black, upper_black)



    mask = cv2.GaussianBlur(mask, (5,5), 0)


    mask_3ch = cv2.merge([mask, mask, mask])
    mask_3ch = mask_3ch / 255.0  # normalize to 0-1 for blending

# Blend images only at black regions
    blended = base * (1 - mask_3ch) + overlay * mask_3ch
    blended = blended.astype(np.uint8)
    return blended

overaly=cv2.imread("upated.png")
overlay = cv2.resize(overaly, (640, 500))

h,w=overlay.shape[:2]
print(overlay.shape,base2.shape)

y=200
x=500
base2[y:y+h, x:x+w]=blender(overlay)

cv2.imshow('Blended.jpg', base2)
# cv2.imwrite('Blended2.jpg', blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
