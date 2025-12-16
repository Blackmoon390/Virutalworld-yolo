import cv2
import numpy as np
from ultralytics import YOLO
import torch
import textureprocessmodule as tpm
import Imageblender as ib
import Variables as vr
import time


cpu= "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {cpu}")

model=YOLO("yolo11n-seg.pt")

topheight=ib.y1
bottomheight=ib.y2
key=ib.key
color=ib.bluecolour

if key is False:
    ib.print_error()


instructimage=cv2.imread("instruction.png")
cv2.imshow("Instruction for usage",instructimage)
cv2.waitKey(10)
time.sleep(8)
cv2.destroyAllWindows()

print("press space key to exit()")


    
cam=cv2.VideoCapture(vr.mainsource) 

if not cam.isOpened():
    raise Exception("Camera not accessible")

while cam.isOpened():
    _,frame=cam.read()
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,None,fx=vr.Camframeratio,fy=vr.Camframeratio)

    results=model(frame,verbose=False,stream=True)
    frame2=np.zeros(frame.shape[:2],dtype='uint8')

    for r in results:
        if r.masks is not None:
            
            for cls,data,box in zip(r.names,r.masks.data,r.boxes.xyxy):
                if int(cls) == 0:
                   global mask
                   mask = data.cpu().numpy().astype('uint8')*255
                   x1,y1,x2,y2=box.int().tolist()
                   yolo_bbox = (x1, y1, x2, y2)
                   mask = tpm.resize_mask_height_only(mask, yolo_bbox,topheight,bottomheight) 
                

    person_mask_colour=tpm.apply_blue_black_noise(color, mask)
    glowframe=tpm.glow(mask)
    
    overall=cv2.add(person_mask_colour,glowframe)
    mainframe=ib.blender(overall)

    cv2.imshow('mask',mainframe)
    if vr.basecamera:
        cv2.imshow("frame",frame)  

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
cam.release()
cv2.destroyAllWindows()
