import cv2
import numpy as np
from ultralytics import YOLO
import torch
import textureprocessmodule as tpm


cpu= "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {cpu}")

# model=YOLO("yolo11n-seg.pt")

bluecolour=np.full((640,480,3),(24, 34, 200),dtype='uint8')
bluecolour=cv2.cvtColor(bluecolour,cv2.COLOR_BGR2RGB)



cam=cv2.VideoCapture(0)

while cam.isOpened():
    _,frame=cam.read()

    results=model(frame,verbose=False,stream=True)
    frame2=np.zeros(frame.shape[:2],dtype='uint8')

    for r in results:
        if r.masks is not None:
            
            for cls,data in zip(r.names,r.masks.data):
                if int(cls) == 0:
                   global mask
                   mask = data.cpu().numpy().astype('uint8')*255
                   mask=cv2.resize(mask,(480,640))

                   
    mask_not=cv2.bitwise_not(mask)
    person_mask_colour=tpm.apply_blue_black_noise(bluecolour, mask)
    person_mask_colour=cv2.resize(person_mask_colour,(1280,480))
    glowframe=tpm.glow(mask)
    glowframe=cv2.resize(glowframe,(1280,480))
    print(person_mask_colour.shape,glowframe.shape)
    overall=cv2.add(person_mask_colour,glowframe)

    cv2.imshow('mask',overall)       

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
