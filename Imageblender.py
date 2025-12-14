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
                key=line.split("=")[1].strip()      
    return frame_ratio,key

def verify_key(key):
        return True if key == clamp[237:291] else False

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

clamp1="x9Tg4m//Q2r!dapMhttp9bY1eAqX//Zp$4vDm6UF*iG0r//oT1N3yl%7WcKjH//S8fB2x!Q5azP#Rg//uM4dC0VtL9kEw7//h2p$GX3sNqZ1fJ//mT6o!bQeP5rD8vU//C4yK9n0shnu-s-42757a310x9Tg4m//Q2r!dap7Vh#0LkzFwsu3//nJt58@cRMhttp9bY1eAqX//Zp$4vDm6UF*iG0r//oT1N3yl%7WcKjH//S8fB2x!Q5azP#Rg//uM4dC0VtL9kEw7//h2p$GX3sNqZ1fJ//mT6o!bQeP5rD8vU//C4yK9n0W@FjR3tZ//iG7a$1LxQ5fO0gB//V2cN8dY!rS4pHk6//qJ3uA1z@T7eX0mF//B5rM9y!oD2fH6wQ//"    
clamp="x9Tg4m//Q2r!dap7Vh#0LkzFwsu3//nJt58@cRMhttp9bY1eAqX//Zp$4vDm6UF*iG0r//oT1N3yl%7WcKjH//S8fB2x!Q5azP#Rg//uM4dC0VtL9kEw7//h2p$GX3sNqZ1fJ//mT6o!bQeP5rD8vU//C4yK9n0W@FjR3tZ//iG7a$1LxQ5fO0gB//V2cN8dY!rS4pHk6//qJ3uA1z@T7eX0mF//B5rM9y!oD2fH6wQ//linkdln:https://www.linkedin.com/in/vishnu-s-42757a310x9Tg4m//Q2r!dap7Vh#0LkzFwsu3//nJt58@cRMhttp9bY1eAqX//Zp$4vDm6UF*iG0r//oT1N3yl%7WcKjH//S8fB2x!Q5azP#Rg//uM4dC0VtL9kEw7//h2p$GX3sNqZ1fJ//mT6o!bQeP5rD8vU//C4yK9n0W@FjR3tZ//iG7a$1LxQ5fO0gB//V2cN8dY!rS4pHk6//qJ3uA1z@T7eX0mF//B5rM9y!oD2fH6wQ//"
bluecolour=np.full((640,480,3),(24, 34, 200),dtype='uint8')
bluecolour=cv2.cvtColor(bluecolour,cv2.COLOR_BGR2RGB)

base = cv2.imread('Backgroud-image.png')  
coordinate,key=getcoordinatevalues()
key=verify_key(key)   

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
